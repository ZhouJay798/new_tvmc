"""Shared groupwise quantizer for packed evaluation and simulated INT4 writing."""
from contextlib import contextmanager

import torch
from torch import nn


def quantize_groups(weight, bits, size, importance=None):
    if bits not in (4, 8) or size not in (16, 32, 64) or weight.shape[1] % size:
        raise ValueError("invalid weight quantization layout")
    limit = (1 << (bits - 1)) - 1
    groups = weight.detach().float().reshape(weight.shape[0], -1, size)
    scale = groups.abs().amax(-1, keepdim=True).clamp_min(1e-12) / limit
    levels = (groups / scale).round().clamp(-limit, limit)
    if importance is not None:
        # Diagonal input second moments approximate projection-output squared error.
        importance = importance.reshape(1, -1, size)
        best_error = ((groups - levels * scale).square() * importance).sum(-1, keepdim=True)
        base_scale = scale
        for fraction in (.95, .90, .85, .80):
            candidate = base_scale * fraction
            codes = (groups / candidate).round().clamp(-limit, limit)
            error = ((groups - codes * candidate).square() * importance).sum(-1, keepdim=True)
            better = error < best_error
            scale = torch.where(better, candidate, scale)
            best_error = torch.minimum(best_error, error)
        levels = (groups / scale).round().clamp(-limit, limit)
    return levels, scale


def simulate_int4(weight, size):
    levels, scale = quantize_groups(weight, 4, size)
    return (levels * scale).reshape_as(weight).to(weight.dtype)


class QuantizedWriter:
    """Switch the frozen projections and the merged adapter to the deployment grid."""
    def __init__(self, lab, size):
        self.size = size
        self.adapters = [adapter for _, adapter, _ in lab._lora]
        self.projections = []
        for block in lab.model.model.layers:
            for parent in (block.self_attn, block.mlp):
                for layer in parent.children():
                    if isinstance(layer, nn.Linear):
                        fake = nn.Parameter(simulate_int4(layer.weight, size), requires_grad=False)
                        self.projections.append((layer, layer.weight, fake))

    @contextmanager
    def enabled(self):
        try:
            for layer, _, fake in self.projections:
                layer.weight = fake
            for adapter in self.adapters:
                adapter.quant_group = self.size
            yield
        finally:
            for layer, weight, _ in self.projections:
                layer.weight = weight
            for adapter in self.adapters:
                adapter.quant_group = None

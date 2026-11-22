"""Post-delivery transforms. No fingerprint probes, labels or scores enter this module."""
import json
import math
import random

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .lora_carrier import LoRALinear
from .shared_carriers import CarrierLab


class EvaluationModel(CarrierLab):
    """Reuse the stage-3 tokenizer, utility and probe readers without owner snapshots."""
    def __init__(self, path):
        self.device = torch.device("cuda:0")
        self.repeats = 3
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        packed = path / "packed.json"
        if packed.exists():
            bits = json.loads(packed.read_text())["bits"]
            config = AutoConfig.from_pretrained(path, local_files_only=True)
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)
            replace_linears(model, bits, initialize=False)
            state = torch.load(path / "packed.pt", map_location="cpu", weights_only=True, mmap=True)
            model.load_state_dict(state, assign=True)
            model.tie_weights()
            self.model = model.to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                path, device_map={"": self.device}, torch_dtype="auto", use_safetensors=True, local_files_only=True)
        self.model.requires_grad_(False).eval()


class PackedLinear(nn.Module):
    """Symmetric signed group-wise weights, 64 columns per scale, floating point accumulation."""
    def __init__(self, linear, bits, initialize):
        super().__init__()
        w = linear.weight.detach()
        rows, cols = w.shape
        if cols % 64:
            raise ValueError("group size 64 requires divisible input dimension")
        self.bits = bits
        self.bias = linear.bias
        self.register_buffer("scale", w.new_empty(rows, cols // 64, 1, dtype=torch.float32))
        self.register_buffer("codes", w.new_empty(rows, cols if bits == 8 else cols // 2,
                                                 dtype=torch.uint8))
        if initialize:
            limit = (1 << (bits - 1)) - 1
            grouped = w.float().reshape(rows, -1, 64)
            self.scale.copy_(grouped.abs().amax(-1, keepdim=True).clamp_min(1e-12) / limit)
            q = ((grouped / self.scale).round().clamp(-limit, limit) + limit).byte().reshape(rows, cols)
            self.codes.copy_(q if bits == 8 else q[:, ::2] | (q[:, 1::2] << 4))

    def forward(self, x):
        q = self.codes
        if self.bits == 4:
            q = torch.stack((q & 15, q >> 4), dim=-1).flatten(1)
        limit = (1 << (self.bits - 1)) - 1
        weight = (q.float().reshape(*self.scale.shape[:2], 64)
                  - limit) * self.scale
        return F.linear(x, weight.flatten(1).to(x.dtype), self.bias)


def projections(model):
    for block in model.model.layers:
        for parent in (block.self_attn, block.mlp):
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear):
                    yield parent, name, child


def replace_linears(model, bits, initialize):
    for parent, name, layer in projections(model):
        setattr(parent, name, PackedLinear(layer, bits, initialize))


def save_variant(lab, dest, op):
    dest.mkdir()
    if op in ("int4", "int8"):
        bits = int(op[3:])
        replace_linears(lab.model, bits, initialize=True)
        lab.model.config.save_pretrained(dest)
        torch.save(lab.model.state_dict(), dest / "packed.pt")
        (dest / "packed.json").write_text(json.dumps(dict(bits=bits, group_size=64)))
    else:
        if op == "bf16":
            lab.model.to(torch.bfloat16)
        lab.model.save_pretrained(dest, safe_serialization=True)
    lab.tokenizer.save_pretrained(dest)


@torch.no_grad()
def prune(model, percent):
    zeros = total = 0
    for _, _, layer in projections(model):
        w = layer.weight
        k = int(w.numel() * percent / 100)
        if k:
            threshold = w.abs().flatten().kthvalue(k).values
            w.masked_fill_(w.abs() <= threshold, 0)
        zeros += int((w == 0).sum())
        total += w.numel()
    return dict(sparsity=zeros / total, pruned_population=total)


def examples(tok, sft_rows, cpt_text, task, pct, seed):
    if task == "cpt":
        ids = tok.encode(cpt_text, add_special_tokens=False)
        data = [(chunk, chunk) for pos in range(0, len(ids), 256)
                if len(chunk := ids[pos:pos + 256]) >= 2]
        random.Random(seed).shuffle(data)
    else:
        data = []
        ordered = list(sft_rows)
        random.Random(seed).shuffle(ordered)
        for row in ordered[:max(1, math.ceil(len(ordered) * pct / 100))]:
            prompt = tok.apply_chat_template([dict(role="user", content=row["instruction"])],
                                             tokenize=True, add_generation_prompt=True)
            answer = tok.encode(row["response"], add_special_tokens=False)[:127] + [tok.eos_token_id]
            prompt = prompt[-(256-len(answer)):]
            data.append((prompt + answer, [-100] * len(prompt) + answer))
    if not data:
        raise ValueError("empty training split")
    return data


def train(lab, task, scope, data, budget, seed, fixed_mask=False):
    torch.manual_seed(seed)
    model = lab.model
    layers = model.model.layers
    original_count = sum(p.numel() for p in model.parameters())
    adapters = []
    model.float()
    if scope.startswith("r"):
        for block in layers:
            base = block.mlp.down_proj
            adapter = LoRALinear(base, int(scope[1:]))
            block.mlp.down_proj = adapter
            adapters.append((block, base, adapter))
    elif scope == "full":
        model.requires_grad_(True)
    else:
        for block in layers[-int(scope):]:
            block.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    masks = []
    if fixed_mask:
        if adapters:
            raise ValueError("fixed mask requires direct weight updates")
        masks = [(m.weight, m.weight.detach() == 0) for _, _, m in projections(model)
                 if m.weight.requires_grad]
    lr = 1e-4 if adapters else 1e-5
    optimizer = torch.optim.AdamW(params, lr=lr, foreach=False)
    tokens = steps = targets = 0
    model.train()
    while (steps < len(data) if budget is None else tokens < budget):
        ids, labels = data[steps % len(data)]
        # Consume complete SFT pairs; a fixed-token run may overshoot by one example.
        x = torch.tensor([ids], device=lab.device)
        y = torch.tensor([labels], device=lab.device)
        optimizer.zero_grad(set_to_none=True)
        model(x, labels=y, use_cache=False).loss.backward()
        optimizer.step()
        with torch.no_grad():
            for weight, mask in masks:
                weight.masked_fill_(mask, 0)
        tokens += len(ids)
        targets += sum(v != -100 for v in labels[1:])
        steps += 1
    for block, base, adapter in adapters:
        adapter.merge_()
        block.mlp.down_proj = base
    model.requires_grad_(False).eval()
    return dict(tokens=tokens, targets=targets, steps=steps, examples=len(data), lr=lr,
                trainable_fraction=sum(p.numel() for p in params) / original_count)

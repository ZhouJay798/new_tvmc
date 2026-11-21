"""Low-rank fingerprint carriers with an explicit model-quality constraint.

The closed-form carrier in shared_carriers.CarrierLab solves each probe's
correction independently; with ~85 probes sharing one down_proj matrix the
corrections interfere, and the shipped parameters fail acceptance
(measured error rate 0.359, ||dW||/||W|| up to 2.68).

This module replaces the WRITE step with a low-rank (LoRA) optimisation that
uses real forward passes of the edited model, so the training objective is the
same quantity the final reader measures:

    per bit b with three probes, target sign y_b in {-1,+1}:
        m_b = y_b * (d_{b,1} + d_{b,2} + d_{b,3})
        loss_bit = relu(margin_target - m_b)

and constrains utility in the SAME optimisation:
    loss = mean_b loss_bit + kl_weight * KL(next_token(owner) || next_token(edited))

Only the LoRA parameters train; the Qwen weights stay frozen. Checkpoints are
selected on a calibration split that is disjoint from both the probe pool and
the ordinary training text.
"""
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .shared_carriers import CarrierLab


class LoRALinear(nn.Module):
    """Frozen Linear plus a trainable rank-r update; B starts at zero."""

    def __init__(self, base, rank, alpha=None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha if alpha is not None else rank)
        self.scaling = self.alpha / self.rank
        self.A = nn.Parameter(torch.empty(self.rank, base.in_features,
                                          dtype=base.weight.dtype, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, self.rank,
                                          dtype=base.weight.dtype, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scaling

    def merge_(self):
        with torch.no_grad():
            self.base.weight.add_(self.scaling * (self.B @ self.A))


class LoRACarrierLab(CarrierLab):
    """CarrierLab whose fingerprint is written by a constrained low-rank update."""

    def __init__(self, source, threads=4, layers=5, device="cuda:0"):
        super().__init__(source, threads=threads, layers=layers, device=device)
        self._lora = []
        self.lora_rank = None
        self.lora_blocks = None

    # ---------------- LoRA plumbing ----------------
    def attach_lora(self, blocks=(-1,), rank=8, alpha=None):
        """Attach LoRA adapters. blocks are indices into model.model.layers;
        the default -1 selects the LAST transformer block (mlp.down_proj)."""
        layers = self.model.model.layers
        self.detach_lora()
        self.lora_blocks = tuple(blocks)
        self.lora_rank = int(rank)
        for b in blocks:
            idx = b if b >= 0 else len(layers) + b
            block = layers[idx]
            base = block.mlp.down_proj
            adapter = LoRALinear(base, rank, alpha)
            block.mlp.down_proj = adapter
            self._lora.append((block, adapter, base))
        return self.lora_parameters()

    def detach_lora(self):
        for block, adapter, base in self._lora:
            block.mlp.down_proj = base
        self._lora = []

    def lora_parameters(self):
        return [p for _, ad, _ in self._lora for p in (ad.A, ad.B)]

    def reset_lora(self, seed=0):
        for _, ad, _ in self._lora:
            with torch.no_grad():
                g = torch.Generator(device=ad.A.device).manual_seed(seed)
                nn.init.kaiming_uniform_(ad.A, a=math.sqrt(5), generator=g)
                ad.B.zero_()

    def merge_lora(self):
        """Merge the adapter into the frozen weights and unwrap the modules so
        save_pretrained writes plain Qwen weights (the reader is unchanged)."""
        for block, adapter, base in self._lora:
            adapter.merge_()
            block.mlp.down_proj = base
        self._lora = []

    def lora_norms(self):
        """||dW||/||W|| of the merged adapter for each selected matrix."""
        out = {}
        for block, adapter, base in self._lora:
            with torch.no_grad():
                d = (adapter.scaling * (adapter.B @ adapter.A)).norm().item()
                out[block] = (d, base.weight.norm().item())
        return out

    # ---------------- readout ----------------
    def readout_logits(self, texts, max_length=64):
        """Real forward passes of the EDITED model; one last-token logit row per probe."""
        x = self.tokenizer(list(texts), return_tensors="pt", truncation=True,
                           max_length=max_length, padding=True,
                           return_token_type_ids=False).to(self.device)
        hidden = self.model.model(**x, use_cache=False).last_hidden_state
        last = x["attention_mask"].sum(1) - 1
        h = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
        return self.model.lm_head(h)

    @torch.no_grad()
    def probe_values(self, probes, batch=48):
        """d_j = logits[a] - logits[b] for each probe, in probe order."""
        out = []
        for k in range(0, len(probes), batch):
            chunk = probes[k:k + batch]
            logits = self.readout_logits([p["text"] for p in chunk])
            a = torch.tensor([p["tokens"][0] for p in chunk], device=self.device)
            b = torch.tensor([p["tokens"][1] for p in chunk], device=self.device)
            idx = torch.arange(len(chunk), device=self.device)
            out.append((logits[idx, a] - logits[idx, b]).detach().cpu())
        return torch.cat(out).numpy()

    def bit_margins(self, probes, word, repeats, values=None):
        """m_b = y_b * (d_{b,1} + d_{b,2} + d_{b,3}), exactly the reader's rule."""
        v = self.probe_values(probes) if values is None else values
        n = len(probes) // repeats
        sums = np.asarray(v).reshape(n, repeats).sum(1)
        y = 2 * np.asarray(word, dtype=np.int64) - 1
        return y * sums, sums

    def readout_stats(self, probes, word, repeats, delta=1.0):
        """error rate, flips, erasures and the per-bit margin distribution."""
        margins, sums = self.bit_margins(probes, word, repeats)
        obs = np.where(sums >= delta, 1, np.where(sums <= -delta, 0, -1))
        target = np.asarray(word, dtype=np.int64)
        flips = int(np.sum((obs != -1) & (obs != target)))
        erasures = int(np.sum(obs == -1))
        return dict(bits=len(obs), error=float(np.mean(obs != target)),
                    flips=flips, erasures=erasures,
                    margin_min=float(margins.min()),
                    margin_median=float(np.median(margins)),
                    margin_below_target=int(np.sum(margins < delta)),
                    margins=[float(x) for x in margins],
                    sums=[float(x) for x in sums])

    # ---------------- ordinary-text KL reference ----------------
    def prepare_kl_batches(self, texts, batch_size=2, count=8, max_length=96, seed=7):
        """Cache the frozen owner's next-token log-probabilities for ordinary text.
        All valid next-token positions are used, so the gradient accounts for the
        shared parameters at every token position."""
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(texts))
        batches = []
        for k in range(count):
            chunk = [texts[i] for i in idx[k * batch_size:(k + 1) * batch_size]]
            if not chunk:
                break
            x = self.tokenizer(chunk, return_tensors="pt", truncation=True,
                               max_length=max_length, padding=True,
                               return_token_type_ids=False)
            mask = x["attention_mask"]
            pos = []
            for b in range(len(chunk)):
                n = int(mask[b].sum())
                for l in range(max(1, n - 1)):
                    pos.append((b, l))
            with torch.no_grad():
                hidden = self.model.model(**x.to(self.device), use_cache=False).last_hidden_state
                bi = torch.tensor([p[0] for p in pos], device=self.device)
                li = torch.tensor([p[1] for p in pos], device=self.device)
                logp = F.log_softmax(self.model.lm_head(hidden[bi, li]), dim=-1).cpu()
            batches.append(dict(texts=chunk, positions=pos, owner_logp=logp))
        return batches

    def next_token_logits(self, batch, max_length=96):
        x = self.tokenizer(batch["texts"], return_tensors="pt", truncation=True,
                           max_length=max_length, padding=True,
                           return_token_type_ids=False).to(self.device)
        hidden = self.model.model(**x, use_cache=False).last_hidden_state
        bi = torch.tensor([p[0] for p in batch["positions"]], device=self.device)
        li = torch.tensor([p[1] for p in batch["positions"]], device=self.device)
        return self.model.lm_head(hidden[bi, li])

    # ---------------- training ----------------
    def train_write(self, word, probes, repeats=3, *, margin_target=1.5,
                    kl_weight=1.0, steps=60, bits_per_batch=8, lr=1e-4,
                    kl_batches=None, kl_max_length=96, eval_every=5,
                    calib_texts=None, owner_ppl=None, owner_preds=None,
                    ppl_budget=1.10, top1_budget=0.85, seed=0,
                    grad_clip=1.0, log=print):
        """Train the low-rank write with a joint fingerprint + quality objective.

        Returns a history and the selected checkpoint. A checkpoint is retained
        only if it satisfies the quality budget on the calibration split; among
        retained checkpoints the one with the lowest bit error rate wins.
        """
        if not self._lora:
            raise ValueError("attach_lora must be called before training")
        word = np.asarray(word, dtype=np.int64)
        groups = {}
        for i, p in enumerate(probes):
            groups.setdefault(int(p["bit"]), []).append(i)
        bits = sorted(groups)
        if len(groups[bits[0]]) != repeats:
            raise ValueError("probe groups must have `repeats` members")
        self.probes = probes
        self.repeats = repeats
        self.bits = len(bits)

        params = self.lora_parameters()
        opt = torch.optim.AdamW(params, lr=lr)
        rng = np.random.default_rng(seed)
        history = []
        best = None
        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            chosen = rng.choice(bits, size=min(bits_per_batch, len(bits)), replace=False)
            idxs = [i for b in chosen for i in groups[b]]
            logits = self.readout_logits([probes[i]["text"] for i in idxs])
            a = torch.tensor([probes[i]["tokens"][0] for i in idxs], device=self.device)
            b = torch.tensor([probes[i]["tokens"][1] for i in idxs], device=self.device)
            rows = torch.arange(len(idxs), device=self.device)
            d = logits[rows, a] - logits[rows, b]
            sums = d.view(len(chosen), repeats).sum(1)
            y = torch.tensor(2 * word[chosen] - 1, dtype=torch.float32, device=self.device)
            margins = y * sums
            loss_bit = F.relu(margin_target - margins).mean()
            loss = loss_bit
            kl_val = 0.0
            if kl_weight > 0 and kl_batches:
                batch = kl_batches[step % len(kl_batches)]
                edited = self.next_token_logits(batch, kl_max_length)
                owner_logp = batch["owner_logp"].to(self.device)
                edited_logp = F.log_softmax(edited, dim=-1)
                kl = (owner_logp.exp() * (owner_logp - edited_logp)).sum(-1).mean()
                kl = torch.nan_to_num(kl, nan=0.0, posinf=0.0, neginf=0.0)
                loss = loss + kl_weight * kl
                kl_val = float(kl.detach())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()

            if (step + 1) % eval_every == 0 and calib_texts is not None:
                q, _ = self.utility(calib_texts, owner_preds)
                ratio = q["perplexity"] / owner_ppl
                stats = self.readout_stats(probes, word, repeats)
                ok = ratio <= ppl_budget and q["top1_agreement"] >= top1_budget
                record = dict(step=step + 1, loss_bit=float(loss_bit.detach()),
                              kl=kl_val, ppl=q["perplexity"], ppl_ratio=ratio,
                              top1=q["top1_agreement"], error=stats["error"],
                              flips=stats["flips"], erasures=stats["erasures"],
                              margin_median=stats["margin_median"], accepted=bool(ok))
                history.append(record)
                if ok and (best is None or stats["error"] < best["error"]):
                    best = dict(record)
                    best["state"] = [(ad.A.detach().clone(), ad.B.detach().clone())
                                     for _, ad, _ in self._lora]
                log(f"step {step+1:>4} | loss_bit={loss_bit.item():.4f} kl={kl_val:.5f} "
                    f"| ppl={q['perplexity']:.2f} ({ratio:.3f}x) top1={q['top1_agreement']:.4f} "
                    f"| err={stats['error']:.4f} flips={stats['flips']} erase={stats['erasures']} "
                    f"| margin med={stats['margin_median']:.3f} min={stats['margin_min']:.3f} "
                    f"| {'ACCEPT' if ok else 'reject'}")
        if best is not None:
            for (_, ad, _), (A, B) in zip(self._lora, best["state"]):
                ad.A.data.copy_(A)
                ad.B.data.copy_(B)
        return dict(history=history, selected=best,
                    accepted_steps=[h["step"] for h in history if h["accepted"]],
                    best_error=best["error"] if best else None)


def ordinary_splits(path):
    """Three disjoint ordinary-text splits (train / calibration / final eval).

    They are disjoint from the probe pool (0.40-0.75) and from the owner
    training text (0.00-0.25) used by split_corpus.
    """
    text = Path(path).read_text()
    n = len(text)

    def chunks(lo, hi, width=200):
        part = text[int(n * lo):int(n * hi)]
        return [part[i:i + width] for i in range(0, max(1, len(part) - width), width)]

    return dict(train=chunks(.25, .40), calib=chunks(.80, .875), eval=chunks(.875, .95))

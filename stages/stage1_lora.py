"""Stage 1: 16-bit fingerprint capacity diagnostic under a quality budget.

Question: on one Qwen2.5-1.5B owner, can 16 bits (3 probes each) all read back
correctly after a merged low-rank write, while the same-corpus perplexity stays
within 10% of the owner and top-1 agreement stays at or above 85%?

Controls:
  * unedited owner        -- must NOT carry the fingerprint;
  * closed-form carrier   -- the original algorithm, on the same probes;
  * LoRA write            -- the new method, with a joint fingerprint + KL term.

All groups embed the SAME probes and are evaluated on the SAME calibration
split, which is disjoint from both the probe pool and the ordinary training text.

Usage (remote, from the project root):
    .venv/bin/python -u stages/stage1_lora.py --bits 16 --margin 1.5
"""
import argparse
import json
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tvmc.shared_carriers import CarrierLab, split_corpus
from tvmc.lora_carrier import LoRACarrierLab, ordinary_splits
from files import pack
from tvmc.stream_delivery import file_hash

MODEL = Path("models/Qwen2.5-1.5B-Instruct/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
CORPUS = Path(".cache/public_data/tinyshakespeare-6f9487a.txt")


def metrics(lab, calib_texts, owner_ppl, owner_preds):
    q, _ = lab.utility(calib_texts, owner_preds)
    return dict(perplexity=q["perplexity"], ppl_ratio=q["perplexity"] / owner_ppl,
                top1_agreement=q["top1_agreement"], tokens=q["tokens"])


def roundtrip_verify(probes, word, repeats, calib_texts, owner_ppl, owner_preds,
                     source_dir, out, tag):
    """Merge, save, zip, reload from the ZIP, and re-measure everything.

    This checks that the fingerprint survives in the file the recipient gets,
    not only in the in-memory model used for training.
    """
    with TemporaryDirectory(dir=out) as scratch:
        source_dir.mkdir(parents=True, exist_ok=True)
        archive = Path(scratch) / "model.zip"
        pack(source_dir, archive)
        sha = file_hash(archive)
        with ZipFile(archive) as z:
            z.extractall(Path(scratch) / "unzipped")
        lab2 = CarrierLab(Path(scratch) / "unzipped", device="cuda:0", read_only=True)
        lab2.repeats = repeats
        lab2.probes = probes
        obs, values = lab2.read()
        target = np.asarray(word, dtype=np.int64)
        flips = int(np.sum((obs != -1) & (obs != target)))
        erasures = int(np.sum(obs == -1))
        q, _ = lab2.utility(calib_texts, owner_preds)
        report = dict(tag=tag, zip_sha256=sha, error=float(np.mean(obs != target)),
                      flips=flips, erasures=erasures, perplexity=q["perplexity"],
                      ppl_ratio=q["perplexity"] / owner_ppl,
                      top1_agreement=q["top1_agreement"])
        del lab2
        torch.cuda.empty_cache()
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bits", type=int, default=16)
    ap.add_argument("--screen-bits", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--blocks", type=str, default="-1")
    ap.add_argument("--margin", type=float, default=1.5)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--kl-weights", type=str, default="0,1,10")
    ap.add_argument("--owner-steps", type=int, default=12)
    ap.add_argument("--ppl-budget", type=float, default=1.10)
    ap.add_argument("--top1-budget", type=float, default=0.85)
    ap.add_argument("--out", type=Path, default=Path("runs/stage1"))
    ap.add_argument("--skip-closed-form", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    bits, screen, repeats = args.bits, args.screen_bits, args.repeats
    total = bits + screen
    blocks = [int(x) for x in args.blocks.split(",")]
    kl_weights = [float(x) for x in args.kl_weights.split(",")]
    started = time.monotonic()
    summary = dict(bits=bits, screen_bits=screen, repeats=repeats, rank=args.rank,
                   blocks=blocks, margin_target=args.margin, steps=args.steps,
                   lr=args.lr, ppl_budget=args.ppl_budget, top1_budget=args.top1_budget)

    print(f"=== STAGE 1: {total} bits x {repeats} probes = {total*repeats} probes ===")
    print(f"LoRA rank={args.rank} on blocks={blocks} (mlp.down_proj)")

    # ---------- one owner, shared probe selection ----------
    lab = LoRACarrierLab(MODEL, device="cuda:0")
    train, generic, pool, _ = split_corpus(CORPUS, total * repeats)
    owner_train = lab.owner_train(train, steps=args.owner_steps)
    print("owner_train:", owner_train)
    probes = lab.select_probes(pool, generic, bits=bits, repeats=repeats,
                               screen_bits=screen)[0]
    lab.probes = probes
    lab.repeats = repeats
    summary["probes"] = len(probes)

    texts = ordinary_splits(CORPUS)
    print(f"ordinary splits: train={len(texts['train'])} "
          f"calib={len(texts['calib'])} eval={len(texts['eval'])}")
    calib = texts["calib"]
    # utility() returns (metrics_dict, predictions)
    owner_q, owner_preds = lab.utility(calib)
    owner_ppl = owner_q["perplexity"]
    owner_eval_q, owner_eval_preds = lab.utility(texts["eval"])
    owner_eval_ppl = owner_eval_q["perplexity"]
    print(f"owner baseline on calibration: PPL={owner_ppl:.3f}")

    rng = np.random.default_rng(991)
    word = rng.integers(0, 2, total)

    results = []

    # ---------- control A: unedited owner ----------
    stats = lab.readout_stats(probes, word, repeats)
    print(f"\n[CONTROL owner] error={stats['error']:.4f} flips={stats['flips']} "
          f"erasures={stats['erasures']} margin_median={stats['margin_median']:.4f}")
    results.append(dict(group="owner_control", error=stats["error"], flips=stats["flips"],
                        erasures=stats["erasures"], perplexity=owner_ppl, ppl_ratio=1.0,
                        top1_agreement=1.0))

    # ---------- control B: closed-form carrier (unchanged algorithm) ----------
    if not args.skip_closed_form:
        lab.restore()
        lab.build(pool, generic, bits=bits, repeats=repeats, screen_bits=screen,
                  margin=3., ridge=.1)
        best_cf = None
        for gain in (.6, .8, 1.0):
            lab.apply(word, gain)
            obs, _ = lab.read()
            target = np.asarray(word, dtype=np.int64)
            err = float(np.mean(obs != target))
            q = metrics(lab, calib, owner_ppl, owner_preds)
            ratios = [(lab.modules[n].weight.detach() - lab.owner[n]).norm().item() /
                      lab.owner[n].norm().item() for n in lab.modules]
            rec = dict(group="closed_form", gain=gain, error=err,
                       flips=int(np.sum((obs != -1) & (obs != target))),
                       erasures=int(np.sum(obs == -1)), **q,
                       dw_ratio_max=max(ratios), dw_ratio_med=float(np.median(ratios)))
            print(f"[CONTROL closed-form] gain={gain}: err={err:.4f} "
                  f"ppl={q['perplexity']:.2f} ({q['ppl_ratio']:.3f}x) "
                  f"top1={q['top1_agreement']:.4f} |dW|/W max={max(ratios):.4f}")
            results.append(rec)
            key = (err, -q["top1_agreement"])
            if best_cf is None or key < best_cf[0]:
                best_cf = (key, gain)
        lab.restore()

    # ---------- LoRA write ----------
    kl_batches = lab.prepare_kl_batches(texts["train"], batch_size=2, count=8)
    lab.attach_lora(blocks=blocks, rank=args.rank)
    print(f"\nLoRA attached: {len(lab.lora_parameters())} parameter tensors")

    for kw in kl_weights:
        lab.reset_lora(seed=7)
        t0 = time.monotonic()
        print(f"\n--- LoRA write: margin_target={args.margin} kl_weight={kw} "
              f"lr={args.lr} steps={args.steps} ---")
        out = lab.train_write(word, probes, repeats=repeats,
                              margin_target=args.margin, kl_weight=kw,
                              steps=args.steps, lr=args.lr,
                              kl_batches=kl_batches, eval_every=args.eval_every,
                              calib_texts=calib, owner_ppl=owner_ppl, owner_preds=owner_preds,
                              ppl_budget=args.ppl_budget, top1_budget=args.top1_budget)
        rec = dict(group="lora", kl_weight=kw, margin_target=args.margin,
                   lr=args.lr, steps=args.steps, accepted_steps=out["accepted_steps"],
                   best_error=out["best_error"], seconds=time.monotonic() - t0,
                   selected=out["selected"])
        if out["selected"] is not None:
            print(f"  selected checkpoint at step {out['selected']['step']} "
                  f"(err={out['selected']['error']:.4f}, "
                  f"ppl_ratio={out['selected']['ppl_ratio']:.3f})")
        else:
            print("  NO checkpoint satisfied the quality budget")
        results.append(rec)
        with (args.out / f"lora_kl{kw}.json").open("w") as f:
            json.dump(rec, f, indent=2, default=str)

    # ---------- round-trip verification of the best LoRA write ----------
    best = min((r for r in results if r.get("group") == "lora" and r.get("selected")),
               key=lambda r: r["best_error"], default=None)
    if best is not None:
        print(f"\n=== round-trip verification of the best LoRA write "
              f"(kl_weight={best['kl_weight']}) ===")
        lab.merge_lora()
        stats = lab.readout_stats(probes, word, repeats)
        q = metrics(lab, calib, owner_ppl, owner_preds)
        print(f"  merged in-memory: err={stats['error']:.4f} flips={stats['flips']} "
              f"erasures={stats['erasures']} ppl={q['perplexity']:.2f} "
              f"({q['ppl_ratio']:.3f}x) top1={q['top1_agreement']:.4f}")
        merged_dir = args.out / "merged_model"
        lab.save_copy(merged_dir)
        rt = roundtrip_verify(probes, word, repeats, calib, owner_ppl, owner_preds,
                              merged_dir, args.out, tag="lora_merged")
        print(f"  after ZIP round-trip: err={rt['error']:.4f} flips={rt['flips']} "
              f"erasures={rt['erasures']} ppl={rt['perplexity']:.2f} "
              f"({rt['ppl_ratio']:.3f}x) top1={rt['top1_agreement']:.4f}")
        final_eval = metrics(lab, texts["eval"], owner_eval_ppl, owner_eval_preds)
        print(f"  held-out final split: ppl={final_eval['perplexity']:.2f} "
              f"({final_eval['ppl_ratio']:.3f}x) top1={final_eval['top1_agreement']:.4f}")
        results.append(dict(group="lora_roundtrip", **rt))
        results.append(dict(group="lora_heldout_eval", **final_eval))

    summary["seconds"] = time.monotonic() - started
    summary["results"] = results
    with (args.out / "stage1.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n=== STAGE1 DONE ===")
    print(f"summary written to {args.out/'stage1.json'}")


if __name__ == "__main__":
    main()

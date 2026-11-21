"""Stage 2: full-capacity calibration at 128 individual + 16 screen bits.

Three pre-generated calibration codewords, each written independently from the
SAME owner. Acceptance (per codeword):
    total error rate <= 2%, same-corpus PPL ratio <= 1.10, top-1 >= 0.85.
Individual-code (bits 0..127) and screen-code (bits 128..143) flips/erasures
are reported separately; the WORST of the three codewords is published.

This is the capacity regime where the closed-form carrier fails (each matrix
serves ~85 probes), so it is where the low-rank write must demonstrate its
advantage. Run after stage1 confirms the pipeline at 16 bits.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tvmc.shared_carriers import split_corpus
from tvmc.lora_carrier import LoRACarrierLab, ordinary_splits

MODEL = Path("models/Qwen2.5-1.5B-Instruct/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
CORPUS = Path(".cache/public_data/tinyshakespeare-6f9487a.txt")


def bit_stats_by_region(stats, word, n_individual):
    """Split flips/erasures into individual vs screen regions."""
    margins = np.asarray(stats["margins"])
    sums = np.asarray(stats["sums"])
    obs = np.where(sums >= 1.0, 1, np.where(sums <= -1.0, 0, -1))
    target = np.asarray(word, dtype=np.int64)
    out = {}
    for name, lo, hi in (("individual", 0, n_individual),
                         ("screen", n_individual, len(word))):
        o, t = obs[lo:hi], target[lo:hi]
        out[name] = dict(bits=len(o), error=float(np.mean(o != t)),
                         flips=int(np.sum((o != -1) & (o != t))),
                         erasures=int(np.sum(o == -1)),
                         margin_min=float(margins[lo:hi].min()),
                         margin_median=float(np.median(margins[lo:hi])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bits", type=int, default=128)
    ap.add_argument("--screen-bits", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--blocks", type=str, default="-1")
    ap.add_argument("--margin", type=float, default=1.5)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bits-per-batch", type=int, default=24)
    ap.add_argument("--kl-weight", type=float, default=1.0)
    ap.add_argument("--owner-steps", type=int, default=12)
    ap.add_argument("--ppl-budget", type=float, default=1.10)
    ap.add_argument("--top1-budget", type=float, default=0.85)
    ap.add_argument("--codewords", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("runs/stage2"))
    ap.add_argument("--skip-closed-form", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    bits, screen, repeats = args.bits, args.screen_bits, args.repeats
    total = bits + screen
    blocks = [int(x) for x in args.blocks.split(",")]
    started = time.monotonic()

    print(f"=== STAGE 2: {bits} individual + {screen} screen x {repeats} "
          f"= {total*repeats} probes ===")
    print(f"LoRA rank={args.rank} blocks={blocks} lr={args.lr} steps={args.steps} "
          f"kl_weight={args.kl_weight}")

    lab = LoRACarrierLab(MODEL, device="cuda:0")
    train, generic, pool, _ = split_corpus(CORPUS, total * repeats)
    lab.owner_train(train, steps=args.owner_steps)
    probes = lab.select_probes(pool, generic, bits=bits, repeats=repeats,
                               screen_bits=screen)[0]
    lab.probes = probes
    lab.repeats = repeats
    print(f"probes: {len(probes)}")

    texts = ordinary_splits(CORPUS)
    calib = texts["calib"]
    owner_q, owner_preds = lab.utility(calib)
    owner_ppl = owner_q["perplexity"]
    print(f"owner baseline calibration PPL={owner_ppl:.3f}")

    rng = np.random.default_rng(991)
    codewords = [rng.integers(0, 2, total) for _ in range(args.codewords)]

    results = []

    # ---- closed-form control on the full-capacity codeword ----
    if not args.skip_closed_form:
        lab.restore()
        lab.build(pool, generic, bits=bits, repeats=repeats, screen_bits=screen,
                  margin=3., ridge=.1)
        w0 = codewords[0]
        for gain in (.6, .8, 1.0):
            lab.apply(w0, gain)
            obs, _ = lab.read()
            target = np.asarray(w0, dtype=np.int64)
            q, _ = lab.utility(calib, owner_preds)
            ratios = [(lab.modules[n].weight.detach() - lab.owner[n]).norm().item() /
                      lab.owner[n].norm().item() for n in lab.modules]
            print(f"[closed-form] gain={gain}: err={float(np.mean(obs != target)):.4f} "
                  f"ppl={q['perplexity']:.1f} ({q['perplexity']/owner_ppl:.3f}x) "
                  f"top1={q['top1_agreement']:.4f} |dW|/W max={max(ratios):.4f}")
            results.append(dict(group="closed_form", gain=gain,
                                error=float(np.mean(obs != target)),
                                ppl_ratio=q["perplexity"] / owner_ppl,
                                top1=q["top1_agreement"], dw_ratio_max=max(ratios)))
        lab.restore()
        del lab.edits

    # ---- LoRA write, one fresh run per codeword ----
    kl_batches = lab.prepare_kl_batches(texts["train"], batch_size=2, count=8)
    lab.attach_lora(blocks=blocks, rank=args.rank)
    for wi, w in enumerate(codewords):
        lab.reset_lora(seed=100 + wi)
        t0 = time.monotonic()
        print(f"\n--- codeword {wi+1}/{args.codewords} ---")
        out = lab.train_write(w, probes, repeats=repeats, margin_target=args.margin,
                              kl_weight=args.kl_weight, steps=args.steps, lr=args.lr,
                              bits_per_batch=args.bits_per_batch,
                              kl_batches=kl_batches, eval_every=args.eval_every,
                              calib_texts=calib, owner_ppl=owner_ppl, owner_preds=owner_preds,
                              ppl_budget=args.ppl_budget, top1_budget=args.top1_budget)
        rec = dict(group="lora", codeword=wi, kl_weight=args.kl_weight,
                   seconds=time.monotonic() - t0, selected=out["selected"],
                   best_error=out["best_error"],
                   accepted_steps=out["accepted_steps"])
        if out["selected"] is not None:
            stats = lab.readout_stats(probes, w, repeats)
            regions = bit_stats_by_region(stats, w, bits)
            rec["regions"] = regions
            rec["error"] = stats["error"]
            ind, scr = regions["individual"], regions["screen"]
            print(f"  codeword {wi}: SELECTED step {out['selected']['step']} "
                  f"total_err={stats['error']:.4f} "
                  f"| ind err={ind['error']:.4f} (flip {ind['flips']}/erase {ind['erasures']}) "
                  f"| screen err={scr['error']:.4f} (flip {scr['flips']}/erase {scr['erasures']}) "
                  f"| ppl_ratio={out['selected']['ppl_ratio']:.3f}")
        else:
            print(f"  codeword {wi}: NO checkpoint satisfied the budget")
        results.append(rec)
        with (args.out / f"codeword{wi}.json").open("w") as f:
            json.dump(rec, f, indent=2, default=str)

    lora_errs = [r.get("error") for r in results if r.get("group") == "lora" and "error" in r]
    summary = dict(bits=bits, screen_bits=screen, repeats=repeats, rank=args.rank,
                   blocks=blocks, steps=args.steps, lr=args.lr,
                   kl_weight=args.kl_weight, seconds=time.monotonic() - started,
                   codeword_errors=lora_errs,
                   worst_error=max(lora_errs) if lora_errs else None,
                   passed=bool(lora_errs) and max(lora_errs) <= 0.02,
                   results=results)
    with (args.out / "stage2.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n=== STAGE2 DONE ===")
    print(f"codeword errors: {lora_errs}")
    print(f"worst error: {summary['worst_error']}  passed(<=2% all): {summary['passed']}")
    print(f"summary -> {args.out/'stage2.json'}")


if __name__ == "__main__":
    main()

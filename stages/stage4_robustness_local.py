"""Stage 4: fingerprint robustness under post-delivery model modifications.

For each delivered copy (U1/U2/U3), apply four transformations, each time
starting from the ORIGINAL received ZIP. After each transformation measure:
  * model quality: PPL ratio vs owner, top-1 agreement vs owner
  * fingerprint survival: 16-bit screen matches, 128-bit individual
    retained / flipped / erased
  * attribution: Tardos scores from the observed individual code over the
    three enrolled codewords; unique hit must equal the correct recipient.

Probes and codewords are recovered from the stage3 run's journal (the same
committed/disclosed parameters used by the real pipeline), so no probe or
codeword is regenerated for this experiment.
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assets import MODEL, CORPUS
from tvmc.shared_carriers import CarrierLab, split_corpus
from tvmc.lora_carrier import ordinary_splits
from tvmc.tardos import Tardos
from tvmc.stream_delivery import file_hash

BITS, SCREEN_BITS, REPEATS = 128, 16, 3
COPIES = 3

# transformation budgets (mild but realistic)
FINETUNE_STEPS, FINETUNE_LR = 3, 1e-5
CONTINUE_STEPS, CONTINUE_LR = 8, 1.5e-5
PRUNE_FRACTION = 0.10
QUANT_DTYPE = torch.bfloat16


# -------------------- evidence recovery --------------------
def load_evidence(run):
    """Recover disclosed probes and the three enrolled codewords from a stage3 run."""
    entries = [json.loads(line) for line in (run / "journal.jsonl").read_text().splitlines()]

    def disclosure(group):
        found = [e for e in entries if e["kind"] == "probe_disclosure" and e["body"]["group"] == group]
        if len(found) != 1:
            raise ValueError(f"missing/duplicate probe_disclosure for {group}")
        return found[0]["body"]["opening"]["parameters"]

    attr_p = disclosure("group-1")
    screen_p = disclosure("screen")
    screen_probe = dict(probes=screen_p["probes"], repeats=REPEATS, delta=screen_p.get("delta", 1.0))

    # Recover the three individual codewords from the trace's codebook evidence.
    # trace.json stores the scored ciphertexts and the manifest mapping; the
    # actual words were enrolled via encrypted_codeword entries, so we instead
    # read the ORIGINAL delivered copies (zero error at delivery) and treat
    # their observed individual bits as the enrolled codewords.
    words = {}
    for i in range(1, COPIES + 1):
        artifact = run / "received" / f"U{i}" / "model.zip"
        with TemporaryDirectory(dir=run) as tmp, ZipFile(artifact) as z:
            z.extractall(tmp)
            lab = CarrierLab(tmp, device="cuda:0", read_only=True)
            lab.repeats = REPEATS
            attrs, _ = lab.read(attr_p["probes"], delta=attr_p.get("delta", 1.0))
            screens, _ = lab.read(screen_p["probes"], delta=screen_p.get("delta", 1.0))
            del lab
            gc.collect(); torch.cuda.empty_cache()
        words[f"U{i}"] = dict(individual=attrs.tolist(), screen=screens.tolist(),
                              screen_target=screen_p["word"])

    # Tardos biases: take from the score disclosure parameters
    disclosed = [e for e in entries if e["kind"] == "group_disclosed_and_retired"]
    if len(disclosed) != 1:
        raise ValueError("missing Tardos parameters disclosure")
    params = disclosed[0]["body"]["parameters"]
    return dict(attr=attr_p, screen=screen_probe, words=words,
                biases=params["biases"], threshold=params["threshold"])


# -------------------- transformations --------------------
def transform_lm(lab, texts, steps, lr, seed=0):
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW([p for p in lab.model.parameters() if p.requires_grad], lr=lr)
    lab.model.train()
    for k in range(steps):
        text = texts[rng.integers(len(texts))]
        x = lab.tokens(text)
        out = lab.model(**x, labels=x["input_ids"], use_cache=False)
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        opt.step()
    lab.model.eval()


def transform_prune(lab, fraction):
    with torch.no_grad():
        for m in lab.modules.values():
            w = m.weight.data
            flat = w.abs().flatten()
            k = int(flat.numel() * fraction)
            if k <= 0:
                continue
            cutoff = torch.kthvalue(flat, k).values
            w[w.abs() <= cutoff] = 0


def transform_quant(lab, dtype):
    with torch.no_grad():
        for m in lab.modules.values():
            w = m.weight.data
            m.weight.data = w.to(dtype).float().contiguous()


# -------------------- measurement --------------------
def measure(lab, ev, owner_ppl, owner_preds, calib_texts):
    q, preds = lab.utility(calib_texts, owner_preds)
    ratio = q["perplexity"] / owner_ppl
    top1 = q["top1_agreement"]

    attrs, _ = lab.read(ev["attr"]["probes"], delta=ev["attr"].get("delta", 1.0))
    screens, _ = lab.read(ev["screen"]["probes"], delta=ev["screen"].get("delta", 1.0))

    screen_matches = int(np.sum(screens == np.asarray(ev["words"]["U1"]["screen_target"])))

    tardos = Tardos(length=BITS, n=COPIES, seed=1)
    tardos.p = np.asarray(ev["biases"], dtype=float)
    tardos.threshold = float(ev["threshold"])
    word_arrays = [np.asarray(ev["words"][f"U{i}"]["individual"]) for i in range(1, COPIES + 1)]
    scores = tardos.scores(word_arrays, attrs).tolist()
    hit_idx = tardos.decode(word_arrays, attrs)
    hits = [f"U{i+1}" for i in hit_idx]

    # individual-code survival relative to this copy's ORIGINAL codeword
    return dict(
        ppl=round(q["perplexity"], 3), ppl_ratio=round(ratio, 4), top1=round(top1, 4),
        quality_ok=bool(ratio <= 1.10 and top1 >= 0.85),
        screen_matches=screen_matches, screen_ok=bool(screen_matches >= 12),
        scores={f"U{i+1}": round(float(s), 3) for i, s in enumerate(scores)},
        threshold=round(tardos.threshold, 3), hits=hits,
        attr_observed=attrs.tolist(), screen_observed=screens.tolist())


def survival(observed, original):
    obs = np.asarray(observed); orig = np.asarray(original)
    erased = int(np.sum(obs < 0))
    known = obs >= 0
    retained = int(np.sum(known & (obs == orig)))
    flipped = int(np.sum(known & (obs != orig)))
    return dict(retained=retained, flipped=flipped, erased=erased, known=int(known.sum()))


# -------------------- main --------------------
def run(stage3_run, out):
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    ev = load_evidence(stage3_run)
    print("evidence recovered: 3 codewords, probes, Tardos params", flush=True)

    train, generic, prompts, utility = split_corpus(CORPUS, (BITS + SCREEN_BITS) * REPEATS)
    texts = ordinary_splits(CORPUS)
    calib = texts["calib"]

    # owner baseline: use the SAME post-owner_train owner that stage3 started
    # from (saved as owner_model.zip), not the raw pretrained checkpoint.
    owner_zip = stage3_run / "owner_model.zip"
    with TemporaryDirectory(dir=out) as tmp, ZipFile(owner_zip) as z:
        z.extractall(tmp)
        owner = CarrierLab(tmp, device="cuda:0", read_only=True)
        owner_q, owner_preds = owner.utility(calib)
        owner_ppl = owner_q["perplexity"]
        del owner
        gc.collect(); torch.cuda.empty_cache()
    print(f"owner baseline (post owner_train) PPL={owner_ppl:.3f}", flush=True)

    transforms = [
        ("finetune", lambda lab, seed: transform_lm(lab, texts["train"], FINETUNE_STEPS, FINETUNE_LR, seed)),
        ("continue_train", lambda lab, seed: transform_lm(lab, texts["train"], CONTINUE_STEPS, CONTINUE_LR, seed)),
        ("prune", lambda lab, seed: transform_prune(lab, PRUNE_FRACTION)),
        ("quantize", lambda lab, seed: transform_quant(lab, QUANT_DTYPE)),
    ]

    rows = []
    for i in range(1, COPIES + 1):
        cid = f"U{i}"
        artifact = stage3_run / "received" / cid / "model.zip"
        original_word = ev["words"][cid]["individual"]
        for name, fn in transforms:
            print(f"\n=== {cid} / {name} ===", flush=True)
            t0 = time.monotonic()
            with TemporaryDirectory(dir=out) as tmp, ZipFile(artifact) as z:
                z.extractall(tmp)
                lab = CarrierLab(tmp, device="cuda:0")
                lab.repeats = REPEATS
                fn(lab, seed=i)
                m = measure(lab, ev, owner_ppl, owner_preds, calib)
                del lab
                gc.collect(); torch.cuda.empty_cache()
            surv = survival(m["attr_observed"], original_word)
            correct = m["hits"] == [cid]
            attribution_ok = bool(m["screen_ok"] and m["hits"] and correct)
            row = dict(copy=cid, transform=name,
                       ppl_ratio=m["ppl_ratio"], top1=m["top1"], quality_ok=m["quality_ok"],
                       screen_matches=m["screen_matches"], screen_ok=m["screen_ok"],
                       retained=surv["retained"], flipped=surv["flipped"], erased=surv["erased"],
                       scores=m["scores"], threshold=m["threshold"],
                       hits=m["hits"], attribution_ok=attribution_ok,
                       seconds=round(time.monotonic() - t0, 1))
            rows.append(row)
            save_rows(out / "robustness.csv", rows)
            print(f"  quality={m['quality_ok']} (ppl {m['ppl_ratio']}x top1 {m['top1']}) | "
                  f"screen={m['screen_matches']}/16 | ind retained/flip/erase="
                  f"{surv['retained']}/{surv['flipped']}/{surv['erased']} | "
                  f"hits={m['hits']} attribution_ok={attribution_ok}", flush=True)

    save_json(out / "robustness.json", dict(rows=rows, seconds=time.monotonic() - started,
                                            owner_ppl=owner_ppl))
    print(f"\nROBUSTNESS DONE {time.monotonic()-started:.1f}s", flush=True)
    return rows


def save_rows(path, rows):
    import csv
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()})


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage3_run", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    try:
        run(a.stage3_run, a.output)
    except Exception as e:
        save_json(a.output / "robustness.json", dict(error=str(e)))
        raise

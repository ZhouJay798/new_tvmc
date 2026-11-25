"""Offline paired robustness study on saved stage-3 deliveries."""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import gc
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from assets import CORPUS, MODEL
from stages.stage4_robustness import load_evidence, save_rows, survival
from tvmc.lora_carrier import ordinary_splits
from tvmc.robust_transforms import EvaluationModel, examples, prune, save_variant, train
from tvmc.stream_delivery import file_hash


def scenarios():
    single = ["original", "bf16", "int8", "int4", "int4-g32", "int4-g16", "int4-cal16",
              *[f"prune{p}" for p in (5, 10, 20, 30, 40)],
              *[f"sft-{s}" for s in ("r8", "r16", "1", "3", "7", "full")],
              *[f"data{p}" for p in (1, 5, 10, 25, 100)],
              *[f"cpt{k}" for k in (10000, 50000, 200000, 1000000)]]
    return {**{s: [s] for s in single},
            "C1": ["sft-r16", "int8"], "C2": ["sft-r16", "int4"],
            "C3": ["prune20", "sft-1"], "C4": ["sft-1", "prune20"],
            "C5": ["cpt200000", "sft-r16"], "C6": ["cpt200000", "sft-r16", "int4"]}


@contextmanager
def original(run, cid, scratch):
    if cid == "base":
        yield MODEL
        return
    archive = run / ("owner_model.zip" if cid == "owner" else f"received/{cid}/model.zip")
    with TemporaryDirectory(dir=scratch) as tmp, ZipFile(archive) as z:
        if any(Path(name).name != name or name.startswith(".") for name in z.namelist()):
            raise ValueError("unexpected archive path")
        z.extractall(tmp)
        yield Path(tmp)


def fingerprint(lab, ev, scale):
    attr = ev["attr"]
    obs, values = lab.read(attr["probes"], delta=attr["delta"])
    screen, _ = lab.read(ev["screen"]["probes"], delta=ev["screen"]["delta"])
    matches = int(sum(screen == ev["words"]["U1"]["screen_target"]))
    a, offset = [], []
    for p, y in zip(ev["biases"], obs, strict=True):
        u, v = math.sqrt((1-p)/p), math.sqrt(p/(1-p))
        a.append(round(scale * (0 if y < 0 else (u+v)*(1 if y else -1))))
        offset.append(0 if y < 0 else (-v if y else v))
    scores = {cid: int(np.dot(a, w["individual"])) + round(scale * sum(offset))
              for cid, w in ev["words"].items()}
    cutoff = math.ceil(scale * ev["threshold"] + (len(obs)+1)/2)
    hits = [cid for cid, score in scores.items() if score >= cutoff] if matches >= 12 else []
    return obs, np.reshape(values, (-1, 3)).sum(1), dict(
        screen_matches=matches, scores_integer=scores, cutoff_integer=cutoff, scale=scale, hits=hits)


def transform(lab, op, previous, args, sft, cpt, seed):
    if op.startswith("prune"):
        return prune(lab.model, int(op[5:]))
    if op.startswith(("sft-", "data", "cpt")):
        pct = int(op[4:]) if op.startswith("data") else 100
        scope = op.split("-")[1] if op.startswith("sft-") else ("1" if op.startswith("cpt") else "r16")
        task = "cpt" if op.startswith("cpt") else "sft"
        data = examples(lab.tokenizer, sft, cpt, task, pct, seed)
        budget = int(op[3:]) if task == "cpt" else args.sft_tokens
        return train(lab, task, scope, data, None if op.startswith("data") else budget,
                     seed, fixed_mask=previous == "prune20")
    return {}


def make_challenge(out, recipient, args):
    """Produce an opaque file before the fresh stage-3 investigation opens its probes."""
    from files import pack
    ops = scenarios()[args.challenge]
    sft = [json.loads(line) for line in args.sft.read_text().splitlines() if line.strip()] if args.sft else []
    splits = ordinary_splits(CORPUS)
    cpt = args.cpt.read_text() if args.cpt else "\n".join(splits["train"])
    with original(out, "owner", out) as path:
        lab = EvaluationModel(path)
        owner, preds = lab.utility(splits["eval"])
        del lab
    history = []
    artifact = out / "challenge.zip"
    with original(out, f"U{recipient}", out) as source, TemporaryDirectory(dir=out) as tmp:
        current = source
        for n, op in enumerate(ops):
            lab = EvaluationModel(current)
            stats = transform(lab, op, ops[n-1] if n else None, args, sft, cpt, args.seed)
            history.append(dict(op=op, **stats))
            dest = Path(tmp)/str(n)
            save_variant(lab, dest, op)
            del lab
            if current != source:
                shutil.rmtree(current)
            current = dest
        pack(current, artifact)
        lab = EvaluationModel(current)
        q, _ = lab.utility(splits["eval"], preds)
    ratio = q["perplexity"]/owner["perplexity"]
    return artifact, dict(case=args.challenge, history=history, seed=args.seed, ppl_ratio=ratio,
                          top1=q["top1_agreement"], sft_sha256=file_hash(args.sft) if args.sft else None,
                          cpt_sha256=sha256(cpt.encode()).hexdigest(),
                          quality_ok=ratio <= 1.10 and q["top1_agreement"] >= .85)


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))
    tmp.replace(path)


def summarize(out):
    rows = [json.loads(p.read_text()) for p in sorted((out / "rows").glob("*.json"))]
    if not rows:
        return
    save_rows(out / "measurements.csv", rows)
    groups = defaultdict(list)
    matrix = Counter()
    for row in rows:
        if row["terminal"]:
            groups[(row["case"], row["negative"])].append(row)
            matrix[row["case"], row["source"], row["decision"]] += 1
    summaries = []
    for (case, negative), group in groups.items():
        useful = [r for r in group if r["quality_ok"]]
        summaries.append(dict(case=case, negative=negative, files=len(group),
            quality_pass=len(useful), correct_decisions=sum(r["correct"] for r in group),
            joint_pass=sum(r["joint_pass"] for r in group),
            joint_rate=sum(r["joint_pass"] for r in group)/len(group),
            correct_given_quality=sum(r["correct"] for r in useful)/len(useful) if useful else None))
    if summaries:
        save_rows(out / "summary.csv", summaries)
        save_rows(out / "confusion.csv", [dict(case=c, source=s, decision=d, count=n)
                                          for (c,s,d),n in sorted(matrix.items())])


def batch_evidence(run_dir):
    evidence = json.loads((run_dir / "report.json").read_text())
    deliveries = {r["cid"]: r for r in evidence["deliveries"]}
    if not evidence["passed"] or not all(deliveries[c]["error"] == 0 for c in ("U1","U2","U3")):
        raise ValueError("reference-word reconstruction requires three zero-error deliveries")
    journal = [json.loads(line) for line in (run_dir / "journal.jsonl").read_text().splitlines()]
    for i in range(1,4):
        cid = f"U{i}"
        hashes = [e["body"]["sha256"] for e in journal
                  if e["kind"] == "received_file" and e["body"]["cid"] == cid]
        if hashes != [file_hash(run_dir / "received" / cid / "model.zip")]:
            raise ValueError(f"received archive mismatch: {cid}")
    scale, = [e["body"]["parameters"]["scale"] for e in journal if e["kind"] == "group_disclosed_and_retired"]
    ev = load_evidence(run_dir)
    if any(not set(w["individual"]) <= {0,1} for w in ev["words"].values()):
        raise ValueError("reference archive has erasures")
    return ev, scale


def run(args):
    cases = scenarios()
    chosen = args.cases.split(",")
    if chosen == ["all"]:
        chosen = list(cases)
    if not set(chosen) <= cases.keys():
        raise ValueError(f"choose from {list(cases)}")
    sources = args.sources.split(",")
    if len(set(chosen)) != len(chosen) or len(set(sources)) != len(sources):
        raise ValueError("duplicate case or source")
    if not set(sources) <= {"U1", "U2", "U3", "owner", "base"}:
        raise ValueError("unknown source")
    if any(op.startswith(("sft-", "data")) for c in chosen for op in cases[c]) and not args.sft:
        raise ValueError("SFT cases require --sft with local instruction/response JSONL")
    sft = [json.loads(line) for line in args.sft.read_text().splitlines() if line.strip()] if args.sft else []
    if args.sft and (not sft or not all(isinstance(r.get(k), str) and r[k].strip()
                                       for r in sft for k in ("instruction","response"))):
        raise ValueError("invalid SFT dataset")
    splits = ordinary_splits(CORPUS)
    cpt = args.cpt.read_text() if args.cpt else "\n".join(splits["train"])
    out = args.output
    run_dir = args.stage3_run
    ev, scale = batch_evidence(run_dir)
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.sft_tokens < 1 or len(set(seeds)) != len(seeds):
        raise ValueError("invalid budget or duplicate seed")
    out.mkdir(parents=True, exist_ok=True)
    plan = dict(stage3_sha256=file_hash(run_dir/"journal.jsonl"), owner_sha256=file_hash(run_dir/"owner_model.zip"),
                cases={c:cases[c] for c in chosen}, sources=sources, seeds=seeds, sft_tokens=args.sft_tokens,
                sft_sha256=file_hash(args.sft) if args.sft else None,
                corpus_sha256=file_hash(CORPUS), cpt_sha256=sha256(cpt.encode()).hexdigest(),
                torch=str(torch.__version__), gpu=torch.cuda.get_device_name(0),
                revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
    manifest = out / "plan.json"
    if manifest.exists() and json.loads(manifest.read_text()) != plan:
        raise ValueError("output already belongs to another plan")
    atomic_json(manifest, plan)
    results = out / "rows"
    results.mkdir(exist_ok=True)
    with original(run_dir, "owner", out) as path:
        lab = EvaluationModel(path)
        owner, owner_preds = lab.utility(splits["eval"])
        del lab
    for cid in sources:
        with original(run_dir, cid, out) as path:
            lab = EvaluationModel(path)
            clean, clean_preds = lab.utility(splits["eval"])
            del lab
            for seed in seeds:
                for case in chosen:
                    ops = cases[case]
                    stem = f"{cid}-{seed}-{case}"
                    if (results / f"{stem}-{len(ops)}.json").exists():
                        continue
                    print(stem, flush=True)
                    with TemporaryDirectory(dir=out) as tmp:
                        current = path
                        for index, op in enumerate(ops, 1):
                            lab = EvaluationModel(current)
                            details = transform(lab, op, ops[index-2] if index>1 else None, args, sft,cpt,seed)
                            dest = Path(tmp)/str(index)
                            save_variant(lab, dest, op)
                            del lab
                            if current != path:
                                shutil.rmtree(current)
                            current = dest
                            gc.collect()
                            torch.cuda.empty_cache()
                            artifact = {p.name:file_hash(p) for p in sorted(dest.iterdir()) if p.is_file()}
                            lab = EvaluationModel(dest)
                            quality, predictions = lab.utility(splits["eval"], owner_preds)
                            clean_top1 = sum(a==b for x,y in zip(clean_preds,predictions,strict=True)
                                             for a,b in zip(x,y,strict=True)) / quality["tokens"]
                            obs, sums, metrics = fingerprint(lab, ev, scale)
                            del lab
                            negative = cid not in ev["words"]
                            hits = metrics["hits"]
                            if metrics["screen_matches"] < 12:
                                decision = "blocked"
                            elif len(hits) == 1:
                                decision = hits[0]
                            else:
                                decision = "multiple" if hits else "no_hit"
                            correct = decision == ("blocked" if negative else cid)
                            ratio = quality["perplexity"] / owner["perplexity"]
                            useful = ratio <= 1.10 and quality["top1_agreement"] >= .85
                            word = None if negative else np.array(ev["words"][cid]["individual"])
                            counts = survival(obs, word) if word is not None else {}
                            row = dict(source=cid, negative=negative, seed=seed, case=case, step=index,
                                terminal=index==len(ops), transform=op, training=details, artifacts=artifact,
                                artifact_bytes=sum(p.stat().st_size for p in dest.iterdir() if p.is_file()),
                                ppl_ratio=ratio, top1=quality["top1_agreement"], quality_ok=useful,
                                clean_ppl_ratio=quality["perplexity"]/clean["perplexity"], clean_top1=clean_top1,
                                counts=counts, survival_rate=counts.get("retained",0)/len(obs) if counts else None,
                                attr_observed=obs.tolist(), bit_sums=sums.tolist(),
                                margin_min=float(np.min(sums*(2*word-1))) if word is not None else None,
                                decision=decision, correct=correct, joint_pass=useful and correct, **metrics)
                            atomic_json(results/f"{stem}-{index}.json", row)
                    summarize(out)
    summarize(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage3_run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cases", default="original,bf16,prune10,int8,int4")
    parser.add_argument("--sources", default="U1,U2,U3,owner,base")
    parser.add_argument("--seeds", default="17")
    parser.add_argument("--sft", type=Path)
    parser.add_argument("--cpt", type=Path)
    parser.add_argument("--sft-tokens", type=int, default=50000)
    import fcntl
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)

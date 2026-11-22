"""Offline robustness measurements on the three delivered Stage 3 Qwen copies."""
import argparse
import csv
import gc
import json
import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assets import CORPUS, REPO, REVISION
from files import save_json
from tvmc.lora_carrier import LoRALinear, ordinary_splits
from tvmc.shared_carriers import CarrierLab
from tvmc.stream_delivery import file_hash
from tvmc.transparency import Journal
from verify import unique

COPIES = ("U1", "U2", "U3")
CASES = ("clean", "instruction_lora", "prune20", "int8_roundtrip", "continue_last_block")
STEPS = 48


def unpack(archive, target):
    with ZipFile(archive) as src:
        for name in src.namelist():
            if Path(name).name != name or name.startswith("."):
                raise ValueError("Invalid Stage 3 archive member")
        src.extractall(target)


def train_variant(lab, case, train_texts, seed):
    rng = np.random.default_rng(seed)
    block = lab.model.model.layers[-1]
    if case == "instruction_lora":
        adapter = LoRALinear(block.mlp.down_proj, rank=16)
        block.mlp.down_proj = adapter
        params = [adapter.A, adapter.B]
        lr = 3e-4
    else:
        block.requires_grad_(True)
        params = list(block.parameters())
        lr = 2e-5
    optimizer = torch.optim.AdamW(params, lr=lr)
    lab.model.train()
    for _ in range(STEPS):
        if case == "instruction_lora":
            a, b = rng.integers(1, 100, size=2)
            messages = [dict(role="user", content=f"What is {a} + {b}? Reply with the number.")]
            prefix = lab.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            answer = lab.tokenizer.encode(str(a + b), add_special_tokens=False)
            ids = torch.tensor([prefix + answer + [lab.tokenizer.eos_token_id]], device=lab.device)
            labels = ids.clone()
            labels[:, :len(prefix)] = -100
        else:
            ids = lab.tokens(train_texts[rng.integers(len(train_texts))])["input_ids"]
            labels = ids
        optimizer.zero_grad(set_to_none=True)
        loss = lab.model(input_ids=ids, labels=labels, use_cache=False).loss
        loss.backward()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if case == "instruction_lora":
        adapter.merge_()
        block.mlp.down_proj = adapter.base
    lab.model.requires_grad_(False)
    lab.model.eval()


@torch.no_grad()
def transform_weights(lab, case):
    # Only transformer linear layers; preserve shared embedding / output weights.
    for module in lab.model.model.layers.modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        for rows in module.weight.split(128):
            if case == "prune20":
                k = rows.shape[1] // 5
                indices = rows.abs().topk(k, dim=1, largest=False).indices
                rows.scatter_(1, indices, 0.0)
            else:
                scale = (rows.abs().amax(dim=1, keepdim=True) / 127).clamp_min(1e-30)
                integers = (rows / scale).round().clamp(-127, 127).to(torch.int8)
                rows.copy_(integers.float() * scale)


def observe(lab, opening):
    lab.repeats = opening["repeats"]
    return lab.read(opening["probes"], delta=opening["delta"])[0]


def scores(params, observed, words):
    # Same signed integer coefficients and shared rounded offset as FileTracer.
    scale = params["scale"]
    coefficients = []
    offset = 0.0
    for p, y in zip(params["biases"], observed, strict=True):
        u, v = math.sqrt((1-p)/p), math.sqrt(p/(1-p))
        coefficients.append(round(scale * (0 if y < 0 else (u+v) * (1 if y else -1))))
        offset += 0 if y < 0 else (-v if y else v)
    return {cid: int(np.dot(word, coefficients)) + round(scale * offset)
            for cid, word in words.items()}


def run(evidence, out, seed):
    report = json.loads((evidence / "report.json").read_text())
    if not (report["passed"] and report["model"] == REPO and report["revision"] == REVISION):
        raise ValueError("Expected successful pinned Qwen Stage 3 run")
    entries = Journal(evidence / "journal.jsonl").entries
    trace = json.loads((evidence / "trace.json").read_text())
    def opening(group):
        record = unique(entries, "probe_disclosure", lambda body: body["group"] == group)
        return record["body"]["opening"]["parameters"]
    screen, attr = opening("screen"), opening(trace["group"])
    params = unique(entries, "group_disclosed_and_retired",
                    lambda body: body["parameters"]["group"] == trace["group"])["body"]["parameters"]
    cutoff = trace["cutoff_integer"]
    texts = ordinary_splits(CORPUS)
    evaluation = texts["eval"][:64]
    owner = CarrierLab(evidence / "owner_model", device="cuda:0", read_only=True)
    owner_quality, reference = owner.utility(evaluation)
    owner_ind, owner_screen = observe(owner, attr), observe(owner, screen)
    del owner
    words = {}
    archives = {}
    with (evidence / "deliveries.csv").open(encoding="utf-8-sig", newline="") as stream:
        deliveries = {row["cid"]: row for row in csv.DictReader(stream)}
    for cid in COPIES:
        archive = evidence / "received" / cid / "model.zip"
        row = deliveries[cid]
        if float(row["error"]) != 0 or file_hash(archive) != row["sha256"]:
            raise ValueError(f"Expected exact zero-error Stage 3 delivery for {cid}")
        archives[cid] = archive
        with TemporaryDirectory(dir=out) as temp:
            unpack(archive, temp)
            lab = CarrierLab(temp, device="cuda:0", read_only=True)
            words[cid] = observe(lab, attr)
            if (len(words[cid]) != 128 or np.any(words[cid] < 0)
                    or not np.array_equal(observe(lab, screen), screen["word"])):
                raise ValueError(f"Invalid clean readout for {cid}")
            del lab
    # Cross-check reconstructed codebook against all three published U1 scores.
    if scores(params, words["U1"], words) != {
            cid: round(value * params["scale"]) for cid, value in trace["scores"].items()}:
        raise ValueError("Clean scoring disagrees with Stage 3")
    rows = []
    def record(cid, case, ind, shared, quality):
        matches = int(np.sum(shared == screen["word"]))
        passed = matches >= screen["minimum_matches"]
        integers = scores(params, ind, words)
        hits = [key for key, value in integers.items() if passed and value >= cutoff]
        row = dict(copy=cid, case=case, screen_matches=matches, screen_passed=passed,
                   screen_survival=matches / len(shared), hits=hits,
                   scores={k: v / params["scale"] for k, v in integers.items()},
                   ppl_ratio=quality["perplexity"] / owner_quality["perplexity"],
                   top1=quality["top1_agreement"])
        row["utility_ok"] = row["ppl_ratio"] <= 1.10 and row["top1"] >= .85
        if cid in words:
            row.update(individual_survival=float(np.mean(ind == words[cid])),
                       flips=int(np.sum((ind >= 0) & (ind != words[cid]))),
                       erasures=int(np.sum(ind < 0)), unique_correct=hits == [cid],
                       wrong_recipient=any(key != cid for key in hits))
        rows.append(row)
        path = out / "measurements.tmp"
        save_json(path, rows)
        path.replace(out / "measurements.json")

    record("owner", "negative_control", owner_ind, owner_screen,
           dict(perplexity=owner_quality["perplexity"], top1_agreement=1.0))
    for cid in COPIES:
        with TemporaryDirectory(dir=out) as temp:
            unpack(archives[cid], temp)
            for case in CASES:
                torch.manual_seed(seed)
                lab = CarrierLab(temp, device="cuda:0", read_only=True)
                if case in ("instruction_lora", "continue_last_block"):
                    train_variant(lab, case, texts["train"], seed)
                elif case != "clean":
                    transform_weights(lab, case)
                quality, _ = lab.utility(evaluation, reference)
                record(cid, case, observe(lab, attr), observe(lab, screen), quality)
                del lab
                gc.collect()
    summary = {}
    for case in CASES:
        subset = [r for r in rows if r["case"] == case]
        summary[case] = dict(copies=len(subset),
                             unique_correct=sum(r["unique_correct"] for r in subset),
                             wrong_recipient=sum(r["wrong_recipient"] for r in subset),
                             utility_ok=sum(r["utility_ok"] for r in subset),
                             usable_and_traced=sum(r["utility_ok"] and r["unique_correct"] for r in subset))
    save_json(out / "summary.json", dict(seed=seed, steps=STEPS, model=REPO, revision=REVISION,
              cutoff_integer=cutoff, eval_examples=len(evaluation), cases=summary,
              owner_false_positive=bool(rows[0]["hits"]), peak_reserved_bytes=torch.cuda.max_memory_reserved()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    run(args.evidence, args.output, args.seed)

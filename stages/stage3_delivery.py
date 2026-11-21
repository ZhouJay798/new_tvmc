"""Stage 3: full file delivery + attribution with LoRA fingerprint embedding.

Replaces the closed-form carrier write (lab.apply) with constrained low-rank
training, then drives the ENTIRE downstream pipeline unchanged:
  DKG threshold custody -> encrypted delivery -> receipt -> hash check
  -> shared-screen identification -> Tardos attribution -> public judge_file.

Also runs a negative control: the un-embedded owner model must FAIL screening.
"""
import argparse
import gc
import secrets
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assets import MODEL, CORPUS, REPO, REVISION
from files import pack, reader, save_json, save_rows
from tvmc.crypto import digest
from tvmc.paper_trace import FileTracer, escrow_parameters, disclose_parameters, judge_file
from tvmc.paper_workflow import Custody
from tvmc.shared_carriers import split_corpus
from tvmc.lora_carrier import LoRACarrierLab, ordinary_splits
from tvmc.stream_delivery import issue_file, receive_file, file_hash
from tvmc.tardos import Tardos
from tvmc.transparency import Journal

BITS, SCREEN_BITS, REPEATS = 128, 16, 3
COPIES = 3
QUORUM = [2, 3, 4]

# LoRA training hyper-parameters (validated in stage 2)
LORA_RANK = 16
LORA_BLOCKS = (-1,)
TRAIN_STEPS = 150
LR = 3e-4
MARGIN_TARGET = 1.5
KL_WEIGHT = 1.0
BITS_PER_BATCH = 24
PPL_BUDGET = 1.10
TOP1_BUDGET = 0.85


def run(out):
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    ledger = Journal(out / "journal.jsonl")
    custody = Custody(n=5, t=3, ledger=ledger)

    # ---- Model & data ----
    lab = LoRACarrierLab(MODEL, device="cuda:0")
    train, generic, prompts, utility = split_corpus(CORPUS, (BITS + SCREEN_BITS) * REPEATS)
    texts = ordinary_splits(CORPUS)

    # ---- Owner training ----
    training = lab.owner_train(train, steps=12)
    owner_q, owner_preds = lab.utility(texts["calib"])
    owner_ppl = owner_q["perplexity"]
    print(f"owner baseline on calibration: PPL={owner_ppl:.3f}", flush=True)

    # Save owner model for negative control (before any fingerprint)
    owner_dir = out / "owner_model"
    lab.save_copy(owner_dir)
    owner_zip = out / "owner_model.zip"
    pack(owner_dir, owner_zip)
    print(f"owner model saved: {owner_zip} ({file_hash(owner_zip)[:16]}...)", flush=True)

    # ---- Probe selection (shared across all copies) ----
    probes, _sel, _asg, _rows, _gk, _cnt, _names = lab.select_probes(
        prompts, generic, bits=BITS, screen_bits=SCREEN_BITS, repeats=REPEATS)
    print(f"selected {len(probes)} probes ({BITS} ind + {SCREEN_BITS} screen, {REPEATS} repeats)", flush=True)

    # ---- Codebook ----
    rng = np.random.default_rng(secrets.randbits(128))
    code = Tardos(length=BITS, n=COPIES, seed=secrets.randbits(64))
    attr_probes = probes[:BITS * REPEATS]
    screen_probes = probes[BITS * REPEATS:]
    screen_word = rng.integers(0, 2, SCREEN_BITS)

    # ---- Tracing setup ----
    tracer = FileTracer(custody, code.p, code.threshold, reader_spec=dict(
        kind="causal-lm-logit-difference", bits=BITS, repeats=REPEATS, delta=1.,
        probe_hash=digest(attr_probes)))
    attr_cap = escrow_parameters(custody, tracer.group,
                                 dict(probes=attr_probes, repeats=REPEATS, delta=1.))
    screen_cap = escrow_parameters(custody, "screen", dict(
        probes=screen_probes, word=screen_word.tolist(), repeats=REPEATS,
        delta=1., minimum_matches=12))

    # ---- KL reference batches (cached once from owner) ----
    kl_batches = lab.prepare_kl_batches(texts["train"], batch_size=2, count=8)

    # ---- Deliver 3 copies ----
    rows = []
    for recipient in range(1, COPIES + 1):
        word = (rng.random(BITS) < code.p).astype(np.int8)
        full_word = np.concatenate([word, screen_word])
        cid = f"U{recipient}"
        print(f"\n=== {cid} ===", flush=True)

        # Restore owner weights, attach fresh LoRA
        lab.restore()
        lab.detach_lora()
        lab.attach_lora(blocks=LORA_BLOCKS, rank=LORA_RANK)
        lab.reset_lora(seed=recipient)

        result = lab.train_write(
            full_word, probes, repeats=REPEATS,
            margin_target=MARGIN_TARGET, kl_weight=KL_WEIGHT,
            steps=TRAIN_STEPS, bits_per_batch=BITS_PER_BATCH, lr=LR,
            kl_batches=kl_batches,
            calib_texts=texts["calib"], owner_ppl=owner_ppl, owner_preds=owner_preds,
            ppl_budget=PPL_BUDGET, top1_budget=TOP1_BUDGET, seed=recipient)

        if result["selected"] is None:
            raise ValueError(f"No accepted checkpoint for {cid}")

        lab.merge_lora()

        # Readout verification (post-merge, pre-save)
        stats = lab.readout_stats(probes, full_word, REPEATS)
        quality, _ = lab.utility(texts["calib"], owner_preds)
        print(f"  readout: err={stats['error']:.4f} flips={stats['flips']} "
              f"erase={stats['erasures']} margin_min={stats['margin_min']:.3f}", flush=True)
        print(f"  quality: PPL={quality['perplexity']:.2f} "
              f"({quality['perplexity']/owner_ppl:.3f}x) top1={quality['top1_agreement']:.4f}", flush=True)

        # Enroll attribution codeword in tracer
        tracer.enroll(cid, word)

        # Save model, pack, deliver via DKG pipeline
        with TemporaryDirectory(dir=out) as scratch:
            model = Path(scratch) / "model"
            lab.save_copy(model)
            archive = Path(scratch) / "model.zip"
            pack(model, archive)
            sent_hash = file_hash(archive)

            package = issue_file(custody, recipient, archive, out / "transit" / cid, cid)
            member = custody.members[recipient - 1]
            request = custody.request(member, package)
            voters = [v for v in custody.members if v.i != recipient][:3]
            for voter in voters:
                custody.intent(voter, request, package)
            approvals = [custody.approval(voter, request, package) for voter in voters]
            inner = custody.release(request, package, approvals)
            received = receive_file(custody, member, package, inner,
                                     out / "received" / cid / "model.zip")
            recv_hash = file_hash(received)
            exact = recv_hash == sent_hash

        row = dict(
            cid=cid, voters=",".join(str(v.i) for v in voters),
            bytes=received.stat().st_size, sha256=recv_hash, exact=exact,
            error=stats["error"], flips=stats["flips"], erasures=stats["erasures"],
            margin_min=stats["margin_min"], margin_median=stats["margin_median"],
            perplexity=quality["perplexity"],
            ppl_ratio=quality["perplexity"] / owner_ppl,
            top1_agreement=quality["top1_agreement"],
            passed=exact and stats["error"] == 0 and quality["top1_agreement"] >= TOP1_BUDGET)
        rows.append(row)
        save_rows(out / "deliveries.csv", rows)
        ledger.append("received_file", dict(cid=cid, sha256=recv_hash, size=received.stat().st_size))
        print(f"  delivered: exact={exact} bytes={received.stat().st_size} "
              f"sent={sent_hash[:16]}... recv={recv_hash[:16]}...", flush=True)

        if not row["passed"]:
            raise ValueError(f"Delivery or readout failed for {cid}")

    del lab
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Shared-screen identification of U1 ----
    print("\n=== Screening ===", flush=True)
    suspect = out / "received/U1/model.zip"
    checksum = file_hash(suspect)
    ledger.append("evidence_commitment", dict(group="screen", file_sha256=checksum))
    screening = disclose_parameters(custody, "screen", screen_cap, QUORUM, checksum)
    screen_read = reader(screening, out)
    observed = screen_read(suspect)
    matches = sum(a == b for a, b in zip(observed, screening["word"], strict=True))
    screen_passes = matches >= screening["minimum_matches"]
    print(f"  U1 screen: {matches}/16 matches (min={screening['minimum_matches']}) "
          f"-> {'PASS' if screen_passes else 'FAIL'}", flush=True)
    ledger.append("screen_result", dict(matches=matches, minimum=screening["minimum_matches"]))

    # ---- Negative control: un-embedded owner must FAIL screening ----
    owner_observed = screen_read(owner_zip)
    owner_matches = sum(a == b for a, b in zip(owner_observed, screening["word"], strict=True))
    owner_passes = owner_matches >= screening["minimum_matches"]
    print(f"  owner screen (negative control): {owner_matches}/16 matches "
          f"-> {'PASSES (BAD!)' if owner_passes else 'blocked (correct)'}", flush=True)

    # ---- Tardos attribution on U1 ----
    print("\n=== Attribution ===", flush=True)
    tracer.commit_file(suspect)
    opening = disclose_parameters(custody, tracer.group, attr_cap, QUORUM, checksum)
    trace_reader = reader(opening, out)
    trace = tracer.trace_file(suspect, trace_reader, QUORUM)
    save_json(out / "trace.json", trace)
    print(f"  scores: { {k: round(v, 3) for k, v in trace['scores'].items()} }", flush=True)
    print(f"  hits: {trace['hits']}  recipients: {trace['recipients']}", flush=True)
    attribution_ok = trace["recipients"] == [1]

    # ---- Public independent verification ----
    print("\n=== Verification ===", flush=True)
    entries = custody.ledger.entries
    head = entries[-1]["hash"]
    verified = judge_file(trace, suspect, trace_reader, entries, head,
                          custody.issuer_pk, custody.registry, custody.code_key)
    print(f"  judge_file: {'PASS' if verified else 'FAIL'}", flush=True)

    all_exact = all(r["exact"] for r in rows)
    all_passed = all(r["passed"] for r in rows)

    report = dict(
        recipients=trace["recipients"], expected=[1],
        attribution_ok=attribution_ok,
        screen_matches=matches, screen_minimum=screening["minimum_matches"],
        screen_passes=screen_passes,
        owner_screen_matches=owner_matches, owner_screen_passes=owner_passes,
        judge_verified=verified,
        all_receipts_exact=all_exact,
        deliveries=[dict(cid=r["cid"], exact=r["exact"], error=r["error"],
                         ppl_ratio=round(r["ppl_ratio"], 4),
                         top1=round(r["top1_agreement"], 4), passed=r["passed"])
                    for r in rows],
        passed=bool(attribution_ok and screen_passes and not owner_passes
                    and verified and all_exact and all_passed),
        seconds=time.monotonic() - started,
        gpu=torch.cuda.get_device_name(0),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        torch_version=torch.__version__, cuda=torch.version.cuda,
        model=REPO, revision=REVISION)
    save_json(out / "report.json", report)
    print(f"\n{'='*50}", flush=True)
    print(f"STAGE3 {'PASS' if report['passed'] else 'FAIL'}", flush=True)
    print(f"  receipts exact:  {all_exact} (3/3)", flush=True)
    print(f"  U1 screen:       {matches}/16 (min 12) {'OK' if screen_passes else 'FAIL'}", flush=True)
    print(f"  owner control:   {owner_matches}/16 {'BLOCKED (good)' if not owner_passes else 'PASSED (BAD)'}", flush=True)
    print(f"  attribution:     {trace['recipients']} (expect [1]) {'OK' if attribution_ok else 'FAIL'}", flush=True)
    print(f"  judge_file:      {'PASS' if verified else 'FAIL'}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    out = parser.parse_args().output
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    try:
        report = run(out)
    except Exception as error:
        save_json(out / "report.json", dict(passed=False, error=str(error)))
        raise
    raise SystemExit(0 if report["passed"] else 2)

"""Issue three Qwen copies and trace the U1 file using the committed parameters."""
import argparse
import gc
import secrets
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from assets import MODEL, CORPUS, REPO, REVISION
from files import pack, reader, save_json, save_rows
from tvmc.crypto import digest
from tvmc.paper_trace import FileTracer, escrow_parameters, disclose_parameters
from tvmc.paper_workflow import Custody
from tvmc.shared_carriers import CarrierLab, split_corpus
from tvmc.stream_delivery import issue_file, receive_file, file_hash
from tvmc.tardos import Tardos
from tvmc.transparency import Journal

BITS, SCREEN_BITS, REPEATS = 128, 16, 3
COPIES = 3
QUORUM = [2, 3, 4]


def prepare_carriers(lab, out):
    train, generic, prompts, utility = split_corpus(CORPUS, (BITS + SCREEN_BITS) * REPEATS)
    base, _ = lab.utility(utility)
    training = lab.owner_train(train, steps=12)
    owner, reference = lab.utility(utility)
    save_json(out / "training.json", dict(base=base, training=training, owner=owner))
    probes = lab.build(prompts, generic, bits=BITS, screen_bits=SCREEN_BITS, repeats=REPEATS)
    text = CORPUS.read_text()
    calibration = text[int(len(text) * .75):int(len(text) * .80)]
    calibration = [calibration[i:i + 200] for i in range(0, 3200, 200)]
    _, calibration_reference = lab.utility(calibration)
    words = np.random.default_rng(991).integers(0, 2, (3, BITS + SCREEN_BITS))
    rows = []
    for gain in (.6, .8, 1.0):
        for i, word in enumerate(words):
            lab.apply(word, gain)
            observed, _ = lab.read()
            quality, _ = lab.utility(calibration, calibration_reference)
            error = float(np.mean(observed != word))
            rows.append(dict(gain=gain, trial=i + 1, error=error, **quality,
                             passed=error <= .02 and quality["top1_agreement"] >= .85))
        save_rows(out / "calibration.csv", rows)
        if all(row["passed"] for row in rows[-len(words):]):
            return probes, gain, utility, reference
    raise ValueError("Carrier calibration failed before issuance")


def deliver(custody, tracer, lab, word, attribution, gain, recipient, quality_inputs, out):
    cid = f"U{recipient}"
    tracer.enroll(cid, attribution)
    lab.apply(word, gain)
    observed, _ = lab.read()
    quality, _ = lab.utility(*quality_inputs)
    with TemporaryDirectory(dir=out) as scratch:
        model = Path(scratch) / "model"
        lab.save_copy(model)
        archive = Path(scratch) / "model.zip"
        pack(model, archive)
        package = issue_file(custody, recipient, archive, out / "transit" / cid, cid)
        member = custody.members[recipient - 1]
        request = custody.request(member, package)
        voters = [v for v in custody.members if v.i != recipient][:3]
        for voter in voters:
            custody.intent(voter, request, package)
        approvals = [custody.approval(voter, request, package) for voter in voters]
        inner = custody.release(request, package, approvals)
        received = receive_file(custody, member, package, inner, out / "received" / cid / "model.zip")
        checksum = file_hash(received)
        exact = checksum == file_hash(archive)
    error = float(np.mean(observed[:BITS] != attribution))
    return dict(cid=cid, voters=",".join(str(v.i) for v in voters), bytes=received.stat().st_size,
                sha256=checksum, exact=exact, error=error, **quality,
                passed=exact and error <= .10 and quality["top1_agreement"] >= .85)


def run(out):
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    ledger = Journal(out / "journal.jsonl")
    custody = Custody(n=5, t=3, ledger=ledger)
    lab = CarrierLab(MODEL, device="cuda:0")
    probes, gain, utility, reference = prepare_carriers(lab, out)
    ledger.append("version_acceptance", dict(model=REPO, revision=REVISION, gain=gain,
                                              calibration_sha256=file_hash(out / "calibration.csv")))
    rng = np.random.default_rng(secrets.randbits(128))
    code = Tardos(length=BITS, n=COPIES, seed=secrets.randbits(64))
    attr = probes[:BITS * REPEATS]
    screen_word = rng.integers(0, 2, SCREEN_BITS)
    tracer = FileTracer(custody, code.p, code.threshold, reader_spec=dict(
        kind="causal-lm-logit-difference", bits=BITS, repeats=REPEATS, delta=1., probe_hash=digest(attr)))
    attr_cap = escrow_parameters(custody, tracer.group, dict(probes=attr, repeats=REPEATS, delta=1.))
    screen_cap = escrow_parameters(custody, "screen", dict(probes=probes[BITS * REPEATS:],
        word=screen_word.tolist(), repeats=REPEATS, delta=1., minimum_matches=12))

    rows = []
    for recipient in range(1, COPIES + 1):
        word = (rng.random(BITS) < code.p).astype(np.int8)
        row = deliver(custody, tracer, lab, np.concatenate([word, screen_word]), word,
                      gain, recipient, (utility, reference), out)
        rows.append(row)
        save_rows(out / "deliveries.csv", rows)
        if not row["passed"]:
            raise ValueError(f"Delivery or model readout failed for {row['cid']}")
        ledger.append("received_file", dict(cid=row["cid"], sha256=row["sha256"], size=row["bytes"]))
        print(f"Delivered {row['cid']}", flush=True)
    del lab, word, rng, attr, probes, screen_word
    gc.collect()
    torch.cuda.empty_cache()

    suspect = out / "received/U1/model.zip"
    checksum = file_hash(suspect)
    ledger.append("evidence_commitment", dict(group="screen", file_sha256=checksum))
    screening = disclose_parameters(custody, "screen", screen_cap, QUORUM, checksum)
    observed = reader(screening, out)(suspect)
    matches = sum(a == b for a, b in zip(observed, screening["word"], strict=True))
    ledger.append("screen_result", dict(matches=matches, minimum=screening["minimum_matches"]))
    if matches < screening["minimum_matches"]:
        raise ValueError("Insufficient common fingerprint matches")
    tracer.commit_file(suspect)
    opening = disclose_parameters(custody, tracer.group, attr_cap, QUORUM, checksum)
    trace = tracer.trace_file(suspect, reader(opening, out), QUORUM)
    save_json(out / "trace.json", trace)
    checkpoint = ledger.checkpoint(custody.issuer, custody.members[1:4])
    save_json(out / "trust.json", dict(pki=custody.registry.trust.export(), registry=custody.registry,
              issuer_pk=custody.issuer_pk.hex(), code_key=custody.code_key.__dict__,
              release_key=custody.release_key.__dict__, checkpoint=checkpoint))
    save_rows(out / "attribution.csv", [dict(cid=cid, score=score, nominal_threshold=code.threshold,
              integer_cutoff=trace["cutoff_integer"], hit=cid in trace["hits"]) for cid, score in trace["scores"].items()])
    return dict(recipients=trace["recipients"], expected=[1], screen_matches=matches,
                passed=trace["recipients"] == [1], seconds=time.monotonic() - started,
                gpu=torch.cuda.get_device_name(0), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                torch_version=torch.__version__, cuda=torch.version.cuda, model=REPO, revision=REVISION)


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
    save_json(out / "report.json", report)
    raise SystemExit(0 if report["passed"] else 2)

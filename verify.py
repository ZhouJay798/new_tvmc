"""Public, file-only verifier. Run in a fresh process with a retained trust digest."""
import argparse
import json
from pathlib import Path

from files import reader
from tvmc.crypto import canonical, digest
from tvmc.paper_threshold import ThresholdPublic, open_capsule
from tvmc.paper_trace import judge_file
from tvmc.paper_workflow import verify
from tvmc.pki import Trust, Registry, issuer_key, member_key
from tvmc.stream_delivery import file_hash
from tvmc.transparency import Journal, verify_checkpoint, root, leaf


def unique(entries, kind, predicate):
    matches = [e for e in entries if e["kind"] == kind and predicate(e["body"])]
    if len(matches) != 1:
        raise ValueError(f"Missing or duplicate {kind}")
    return matches[0]


def disclosed(entries, group, checksum, release, registry):
    def same(body):
        return body["group"] == group
    escrow = unique(entries, "probe_escrow", same)
    evidence = unique(entries, "evidence_commitment", same)
    disclosure = unique(entries, "probe_disclosure", same)
    if not escrow["seq"] < evidence["seq"] < disclosure["seq"]:
        raise ValueError("Evidence order")
    body = disclosure["body"]
    if body["evidence_hash"] != checksum or evidence["body"]["file_sha256"] != checksum:
        raise ValueError("Different suspect artifact")
    capsule = escrow["body"]["capsule"]
    context = ["investigation", group, checksum, digest(capsule)]
    for proof, signature in zip(body["proofs"], body["signatures"], strict=True):
        if not verify(member_key(registry, proof["i"]), "investigation", dict(context=context, proof=proof), signature):
            raise ValueError("Investigation signature")
    opening = json.loads(open_capsule(release, capsule, body["proofs"], ["probes", group], context))
    if (opening != body["opening"] or
            digest([opening["parameters"], opening["salt"]]) != escrow["body"]["commitment"]):
        raise ValueError("Different probe opening")
    return opening["parameters"], disclosure["seq"]


def verify_run(out, anchor):
    if file_hash(out / "trust.json") != anchor:
        raise ValueError("Trust digest mismatch")
    trust = json.loads((out / "trust.json").read_text())
    trace = json.loads((out / "trace.json").read_text())
    ca = trust["pki"]
    registry = Registry(Trust(ca["root_certificate"], ca["crl"], ca["issuer_certificate"]))
    registry.update({int(i): record for i, record in trust["registry"].items()})
    issuer = issuer_key(registry, bytes.fromhex(trust["issuer_pk"]))
    release = ThresholdPublic(**trust["release_key"])
    code = ThresholdPublic(**trust["code_key"])
    checkpoint = trust["checkpoint"]
    journal = Journal(out / "journal.jsonl")
    head = checkpoint["head"]
    if (not verify_checkpoint(checkpoint, issuer, registry, release.t) or not journal.check(head["chain"])
            or head["size"] != len(journal.entries) or head["root"] != root([leaf(canonical(e)) for e in journal.entries]).hex()):
        raise ValueError("Witness checkpoint mismatch")
    artifact = out / "received/U1/model.zip"
    checksum = file_hash(artifact)
    screen, seq = disclosed(journal.entries, "screen", checksum, release, registry)
    observed = reader(screen, out)(artifact)
    count = sum(a == b for a, b in zip(observed, screen["word"], strict=True))
    decision = unique(journal.entries, "screen_result", lambda _: True)
    params, attr_seq = disclosed(journal.entries, trace["group"], checksum, release, registry)
    if (not seq < decision["seq"] < attr_seq or count < screen["minimum_matches"]
            or decision["body"] != dict(matches=count, minimum=screen["minimum_matches"])):
        raise ValueError("Common fingerprint screening failed")
    if digest(params["probes"]) != trace["scope"]["probe_hash"]:
        raise ValueError("Attribution probes mismatch")
    if not judge_file(trace, artifact, reader(params, out), journal.entries, head["chain"], issuer, registry, code):
        raise ValueError("Public attribution verification failed")
    for i in range(1, 4):
        cid = f"U{i}"
        manifest = unique(journal.entries, "manifest", lambda b: b["message"]["cid"] == cid)["body"]["message"]
        signed = unique(journal.entries, "receipt", lambda b: b["receipt"]["cid"] == cid)["body"]
        recorded = unique(journal.entries, "received_file", lambda b: b["cid"] == cid)["body"]
        received = out / "received" / cid / "model.zip"
        if recorded["size"] != received.stat().st_size or recorded["sha256"] != file_hash(received):
            raise ValueError(f"Received file differs from the witnessed record for {cid}")
        expected = dict(cid=cid, ciphertext_hash=manifest["ciphertext_hash"], received=True)
        if "sha256" in manifest:
            expected.update(sha256=recorded["sha256"], size=recorded["size"])
            if any(expected[k] != manifest[k] for k in ("sha256", "size")):
                raise ValueError("Sender and recipient file mismatch")
        if (manifest["recipient"] != i or signed["receipt"] != expected
                or not verify(member_key(registry, i), "receipt", expected, signed["signature"])):
            raise ValueError(f"Invalid signed receipt for {cid}")
    return dict(verified=trace["recipients"] == [1], recipients=trace["recipients"], screen_matches=count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--trust-sha256", required=True)
    args = parser.parse_args()
    result = verify_run(args.output, args.trust_sha256)
    print(json.dumps(result))
    raise SystemExit(0 if result["verified"] else 2)

"""Public replay of the single-host 3-of-5 file delivery experiment."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from tvmc.crypto import canonical, digest
from tvmc.paper_threshold import ThresholdPublic, open_capsule
from tvmc.paper_workflow import verify
from tvmc.pki import Registry, Trust, issuer_key, member_key
from tvmc.stream_delivery import file_hash
from tvmc.transparency import Journal, leaf, root, verify_checkpoint


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def export_evidence(out, custody, recipients):
    checkpoint = custody.ledger.checkpoint(custody.issuer, custody.members[1:4])
    data = dict(pki=custody.registry.trust.export(), registry=custody.registry,
                issuer_pk=custody.issuer_pk.hex(), release_key=vars(custody.release_key),
                code_key=vars(custody.code_key), recipients=recipients, checkpoint=checkpoint)
    path = out / "trust.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    anchor = file_hash(path)
    (out / "trust.sha256").write_text(anchor + "\n")
    return anchor


def audit(out, anchor):
    path = out / "trust.json"
    require(file_hash(path) == anchor, "trust digest mismatch")
    data = json.loads(path.read_text())
    ca = data["pki"]
    registry = Registry(Trust(ca["root_certificate"], ca["crl"], ca["issuer_certificate"]))
    registry.update({int(i): record for i, record in data["registry"].items()})
    issuer = issuer_key(registry, bytes.fromhex(data["issuer_pk"]))
    key = ThresholdPublic(**data["release_key"])
    require((key.n, key.t) == (5, 3) and set(registry) == set(range(1, 6)), "membership")
    journal = Journal(out / "journal.jsonl")
    entries = journal.entries
    checkpoint = data["checkpoint"]
    head = checkpoint["head"]
    require(verify_checkpoint(checkpoint, issuer, registry, key.t) and
            journal.check(head["chain"]) and len(entries) == head["size"] and
            root([leaf(canonical(e)) for e in entries]).hex() == head["root"], "checkpoint")

    def one(kind, matches):
        found = [e for e in entries if e["kind"] == kind and matches(e["body"])]
        require(len(found) == 1, "missing or duplicate " + kind)
        return found[0]["seq"], found[0]["body"]

    joins = {}
    for i, record in registry.items():
        seq, signed = one("registration", lambda b: b["registration"]["i"] == i)
        member_key(registry, i, "receiving")
        require(signed["registration"] == record and
                verify(member_key(registry, i), "join", record, signed["signature"]), "registration signature")
        joins[i] = seq

    rows = []
    recipients = data["recipients"]
    require(len(recipients) == len(set(recipients)) > 0 and
            len(recipients) == sum(e["kind"] == "manifest" for e in entries), "manifest inventory")
    for i in recipients:
        cid = f"U{i}"
        mid, signed = one("manifest", lambda b: b["message"]["cid"] == cid)
        msg = signed["message"]
        require(msg["registration"] == registry[i] and msg["recipient"] == i and
                msg["release_kid"] == key.kid and verify(issuer, "manifest", msg, signed["signature"]), "manifest")
        capseq, wrapped = one("delivery_capsule", lambda b: b["cid"] == cid)
        cap = wrapped["capsule"]
        chunks = sorted((out / "transit" / cid).glob("*.bin"))
        require(chunks and digest([file_hash(p) for p in chunks]) == msg["ciphertext_hash"]
                and digest(cap) == msg["capsule_hash"], "ciphertext or capsule")
        reqseq, signed = one("request", lambda b: b["request"]["cid"] == cid)
        req = signed["request"]
        require(req["recipient"] == i and req["kid"] == key.kid and req["epoch"] == key.epoch
                and req["capsule_hash"] == digest(cap) and req["policy"] == "any-t-registered"
                and req["nonce"] and verify(member_key(registry, i), "request", req, signed["signature"]), "request")
        rid = digest(req)
        relseq, release = one("release", lambda b: b["rid"] == rid)
        proofs = []
        for approval in release["approvals"]:
            proof = approval["proof"]
            voter = proof["i"]
            seq, intent = one("intent", lambda b: b["rid"] == rid and b["i"] == voter)
            pk = member_key(registry, voter)
            require(reqseq < seq < relseq and verify(pk, "intent", dict(i=voter, rid=rid), intent["signature"])
                    and approval["rid"] == rid and verify(pk, "approval", dict(rid=rid, proof=proof),
                                                          approval["signature"]), "approval or intent")
            proofs.append(proof)
        open_capsule(key, cap, proofs, [cid, msg["ciphertext_hash"]], ["release", rid, digest(cap)])
        receiptseq, signed = one("receipt", lambda b: b["receipt"]["cid"] == cid)
        require(max(joins.values()) < mid < capseq < reqseq < relseq < receiptseq, "event order")
        artifact = out / "received" / cid / "model.zip"
        receipt = dict(cid=cid, received=True, ciphertext_hash=msg["ciphertext_hash"],
                       size=artifact.stat().st_size, sha256=file_hash(artifact))
        require(receipt == signed["receipt"] and receipt["size"] == msg["size"]
                and receipt["sha256"] == msg["sha256"] and
                verify(member_key(registry, i), "receipt", receipt, signed["signature"]), "signed file receipt")
        rows.append(dict(cid=cid, voters=[p["i"] for p in proofs], sha256=receipt["sha256"]))
    return dict(verified=True, deliveries=rows)


def independent(out, anchor):
    result = subprocess.check_output([sys.executable, __file__, str(out), "--trust-sha256", anchor], text=True)
    (out / "delivery_audit.json").write_text(result)
    return json.loads(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--trust-sha256", required=True)
    args = parser.parse_args()
    try:
        result = audit(args.output, args.trust_sha256)
    except ValueError as error:
        print(json.dumps(dict(verified=False, reason=str(error))))
        raise SystemExit(2)
    print(json.dumps(result))

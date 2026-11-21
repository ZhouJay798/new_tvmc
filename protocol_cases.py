"""CPU-only acceptance cases for threshold delivery and offline audit."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from shutil import copytree

from delivery_audit import export_evidence, independent
from tvmc.paper_workflow import Custody
from tvmc.stream_delivery import CHUNK, issue_file, receive_file
from tvmc.transparency import Journal
from subprocess import CalledProcessError


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    cases = []

    def record(name, result):
        cases.append(dict(case=name, outcome=result))
        (out / "cases.json").write_text(json.dumps(cases, indent=2))

    def reject(name, action, expected=ValueError):
        try:
            action()
        except expected:
            record(name, "rejected")
        else:
            record(name, "unexpected acceptance")
            raise ValueError(name)

    custody = Custody(5, 3, Journal(out / "journal.jsonl"))
    source = out / "source.bin"
    source.write_bytes(b"a" * CHUNK + b"second block")
    package = issue_file(custody, 1, source, out / "transit/U1", "U1")
    member = custody.members[0]
    target = out / "received/U1/model.zip"
    req = custody.request(member, package)
    voters = custody.members[1:4]
    for voter in voters[:2]:
        custody.intent(voter, req, package)
    reject("two intents", lambda: custody.approval(voters[0], req, package))
    custody.intent(voters[2], req, package)
    approvals = [custody.approval(v, req, package) for v in voters]
    release = lambda votes: custody.release(req, package, votes)
    reject("two approvals", lambda: release(approvals[:2]))
    reject("duplicate approver", lambda: release([approvals[0], approvals[0], approvals[1]]))
    inner = release(approvals)
    record("three distinct approvals", "released")
    reject("release replay", lambda: release(approvals))
    chunks = package["chunks"]
    for name, altered in [("missing chunk", chunks[:-1]), ("reordered chunks", chunks[::-1])]:
        reject(name, lambda: receive_file(custody, member, dict(package, chunks=altered), inner, target))
        if target.exists() or any(e["kind"] == "receipt" for e in custody.ledger.entries):
            raise ValueError("invalid package produced a receipt or file")
    receive_file(custody, member, package, inner, target)
    if target.read_bytes() != source.read_bytes():
        raise ValueError("received bytes differ")
    record("intact file", "received")
    anchor = export_evidence(out, custody, [1])
    independent(out, anchor)
    record("fresh process audit", "verified")
    def rejected_audit(name, directory, reason):
        try:
            independent(directory, anchor)
        except CalledProcessError as error:
            if error.returncode != 2 or json.loads(error.output).get("reason") != reason:
                raise
            record(name, "rejected")
        else:
            raise ValueError(name)

    with TemporaryDirectory() as scratch:
        shadow = Path(scratch) / "evidence"
        copytree(out, shadow)
        file = shadow / "received/U1/model.zip"
        with file.open("r+b") as stream:
            stream.write(b"x")
        rejected_audit("changed received bytes", shadow, "signed file receipt")
        file.write_bytes(source.read_bytes())
        journal = shadow / "journal.jsonl"
        entries = [json.loads(line) for line in journal.read_text().splitlines()]
        next(e for e in entries if e["kind"] == "release")["body"]["approvals"].pop()
        journal.write_text("".join(json.dumps(e) + "\n" for e in entries))
        rejected_audit("removed approval in archived evidence", shadow, "journal corrupt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)

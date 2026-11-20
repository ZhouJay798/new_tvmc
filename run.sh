#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
umask 077
export HF_HOME="$PWD/.cache/huggingface"
export PIP_CACHE_DIR="$PWD/.cache/pip"
export TMPDIR="$PWD/.cache/tmp"
mkdir -p "$TMPDIR" runs
session=$(mktemp -d -p runs file.XXXXXX)
printf 'Run directory: %s/%s\n' "$PWD" "$session"
exec > >(tee "$session/session.log") 2>&1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python3 - <<'PY'
import shutil
import sys
import torch
if sys.version_info[:2] != (3, 12):
    raise SystemExit("Use the Python 3.12 GPU image")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the image Python")
print("Torch", torch.__version__, "CUDA", torch.version.cuda, "GPU", torch.cuda.get_device_name(0))
print("Project filesystem free GiB:", round(shutil.disk_usage(".").free / 2**30, 1))
# Exercise the same FP64 solver used by the fingerprint computation.
a = torch.eye(2, dtype=torch.float64, device="cuda")
torch.testing.assert_close(torch.linalg.solve(a, a), a)
PY
python3 -m venv --system-site-packages .venv
py=.venv/bin/python
"$py" -m pip install -r requirements.txt
"$py" assets.py
"$py" -u experiment.py "$session/evidence"
sha256sum "$session/evidence/trust.json" | cut -d ' ' -f 1 > "$session/trust.sha256"
"$py" verify.py "$session/evidence" --trust-sha256 "$(cat "$session/trust.sha256")" | tee "$session/verification.json"

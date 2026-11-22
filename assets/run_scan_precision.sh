#!/usr/bin/env bash
# Run one slice of the paired robustness sweep. Invoke from the repo root.
set -uo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." || exit 1
STAGE3=${STAGE3:-runs/lora-delivery-001}
SEEDS=${SEEDS:-17}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-"$PWD/.cache/huggingface"}
export TMPDIR=${TMPDIR:-"$PWD/.cache/tmp"}
mkdir -p "$TMPDIR" runs
echo "=== scan-precision start $(date) ==="
.venv/bin/python -u stages/stage4_matrix.py "$STAGE3" "runs/scan-precision" \
  --cases "original,bf16,int8,int4,prune5,prune10,prune20,prune30,prune40" --sources "U1,U2,U3,owner,base" --seeds "$SEEDS"
rc=$?
echo "=== scan-precision done rc=$rc $(date) ==="
exit "$rc"

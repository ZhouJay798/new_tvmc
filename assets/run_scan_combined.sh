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
echo "=== scan-combined start $(date) ==="
.venv/bin/python -u stages/stage4_matrix.py "$STAGE3" "runs/scan-combined" \
  --cases "C1,C2,C3,C4,C5,C6" --sources "U1,U2,U3" --seeds "$SEEDS"  --sft assets/instructions_512.jsonl
rc=$?
echo "=== scan-combined done rc=$rc $(date) ==="
exit "$rc"

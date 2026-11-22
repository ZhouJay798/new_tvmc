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
echo "=== scan-sft start $(date) ==="
.venv/bin/python -u stages/stage4_matrix.py "$STAGE3" "runs/scan-sft" \
  --cases "sft-r8,sft-r16,sft-1,sft-3,sft-7,sft-full,data1,data5,data10,data25,data100" --sources "U1,U2,U3" --seeds "$SEEDS"  --sft assets/instructions_512.jsonl
rc=$?
echo "=== scan-sft done rc=$rc $(date) ==="
exit "$rc"

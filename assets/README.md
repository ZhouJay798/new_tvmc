# Robustness sweep reproduction assets

Companion assets for `stages/stage4_matrix.py` (introduced in commit `2f0593b`).
The upstream `ROBUSTNESS.md` documents the experiment; this directory supplies
the instruction data and one-command launchers used for the scan reported on
2026-11-22 (RTX 5090, 129 measurements).

## Contents

| File | Purpose |
|---|---|
| `instructions_512.jsonl` | 512 arithmetic instruction/response pairs for SFT cases (`sft-*`, `data*`, and joint cases C1/C2/C5/C6). UTF-8, one JSON object per line with `instruction` and `response`. |
| `run_scan_precision.sh` | original, bf16, int8, int4, prune5/10/20/30/40 (incl. owner/base negative controls) |
| `run_scan_sft.sh` | sft-r8/r16, sft-1/3/7/full, data1/5/10/25/100 |
| `run_scan_cpt.sh` | cpt10000/50000/200000/1000000 |
| `run_scan_combined.sh` | joint cases C1–C6 |

`instructions_512.jsonl` SHA-256:
`bb0ce2f2e766dcf77b05bc54df641c869df3d80b1ad2b7d8ad87eb0c6d9f89f8`

## Usage

From the repository root, after stage 3 has produced a complete run directory:

```bash
bash assets/run_scan_precision.sh   # or sft / cpt / combined
# override defaults when needed:
STAGE3=runs/my-delivery SEEDS=17,29 bash assets/run_scan_sft.sh
```

Each launcher delegates to `stages/stage4_matrix.py`, which resumes by skipping
already-completed `source/seed/case` triples. Keep the same arguments when
resuming. Do not count repeated runs of the same deterministic transform as
independent samples.

CPT uses the pinned Tiny Shakespeare corpus prepared by `assets.py` unless
`--cpt` is supplied separately (see `ROBUSTNESS.md`).

#!/usr/bin/env bash
# Run as the systemd user unit `diag-readout` (unsandboxed GPU). Each stage resumes from durable shards/items.
set -euo pipefail
cd /home/station/Documents/Bonsai-demo/analysis/bonsai2/replication
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/station/Documents/Bonsai-demo/.venv/bin/python
for stage in score generate kl analyze; do
  "$PY" -u diag_run.py "$stage"
done
[ -e results/diag_readout/decision.json ] || "$PY" -u diag_run.py decide
"$PY" -u diag_run.py report

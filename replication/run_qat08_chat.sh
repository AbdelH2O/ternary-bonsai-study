#!/usr/bin/env bash
# Invoke as a systemd user unit; retrying this unit resumes the last atomic checkpoint.
set -euo pipefail
cd /home/station/Documents/Bonsai-demo/analysis/bonsai2/replication
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /home/station/Documents/Bonsai-demo/.venv/bin/python -u qat_08b_chat.py train

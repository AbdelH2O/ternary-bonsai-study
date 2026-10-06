#!/usr/bin/env bash
# Run as a systemd user unit after approval. Stops after sealing, before training.
set -euo pipefail
cd /home/station/Documents/Bonsai-demo/analysis/bonsai2/replication
export HF_HUB_DISABLE_XET=1
qat_chat_python=/home/station/Documents/Bonsai-demo/.venv/bin/python
"$qat_chat_python" freeze_qat08_chat.py verify-design
"$qat_chat_python" -u prepare_qat08_chat.py generate
if [[ ! -f work/qat08_chat/data/data_record.json ]]; then
  "$qat_chat_python" -u prepare_qat08_chat.py pack
fi
if [[ ! -f results/qat08_chat/protocol.json ]]; then
  "$qat_chat_python" -u freeze_qat08_chat.py protocol
fi
"$qat_chat_python" -u qat_08b_chat.py verify

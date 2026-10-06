#!/usr/bin/env bash
# Run this on the host with NVIDIA devices available, from any working directory.
set -euo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$root"
base=analysis/bonsai2/scorer
model=models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf
bash "$base/build.sh"
python3 "$base/freeze.py"
awk '$4==1024 && ++n<=2' "$base/baseline_cases.tsv" > "$base/fixture_cases.tsv"
python3 - <<'PY'
from pathlib import Path
p = Path('analysis/bonsai2/scorer')
ids = 'analysis/bonsai2/scorer/pride_and_prejudice.ids'
(p / 'stock_cases.tsv').write_text(f'stock-a\t{ids}\t33\t33\t31\nstock-b\t{ids}\t97\t33\t31\n')
lines = (p / 'fixture_cases.tsv').read_text().splitlines()
(p / 'fixture_reset.tsv').write_text('\n'.join((lines[0], lines[1], lines[0])) + '\n')
PY
bin/cuda/llama-perplexity -m "$model" -ngl 99 -fa on -c 64 -b 64 -ub 64 \
  -f "$base/pride_and_prejudice.txt" --chunks 2 --save-all-logits "$base/stock_fixture_logits.bin" \
  > "$base/stock_fixture.log" 2>&1
score() {
  local cases=$1 output=$2 batch=$3 ubatch=$4
  "$base/position_scorer" score "$model" "$base/$cases" "$base/$output.jsonl" "$batch" "$ubatch" \
    > "$base/$output.log" 2>&1
}
score stock_cases.tsv stock_compare 64 64
score fixture_cases.tsv fixture_run1 512 512
score fixture_cases.tsv fixture_run2 512 512
score fixture_cases.tsv fixture_batch256 256 256
score fixture_cases.tsv fixture_batch1119 1119 512
score fixture_cases.tsv fixture_ubatch256 512 256
score fixture_reset.tsv fixture_reset 512 512
score baseline_cases.tsv baseline_512 512 512
python3 "$base/validate.py"

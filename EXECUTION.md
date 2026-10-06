# Bonsai 2 recurrent-state experiment: execution notes

As of 2026-09-29, the [GPU pilot and two short calibrations](CALIBRATION_REPORT.md)
and [frozen exploratory matrix](EXPERIMENT_REPORT.md) have run. The host GPU
works; the default agent sandbox hides its device nodes. No two matched,
measurably damaging strengths were found, so the tested 2% and 3% pairs are
exploratory arms rather than calibrated causal comparisons.

## Prepared artifacts

- Original model SHA-256: `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1`.
- Nine gitignored candidate GGUFs and JSON patch manifests in `variants/`:
  state and MLP at 1%, 2%, 3%, and 5%, plus MLP at 10%. Each affects exactly
  184,320 groups (23,592,960 effective weights) across 48 linear blocks. The
  original is untouched. `make_variants.py --verify-only` checks metadata,
  dimensions, expected values and every byte outside intended patch ranges.
  All candidates loaded and answered the one-question smoke test sensibly.
- [prompts/primary.jsonl](prompts/primary.jsonl) freezes 24 paired short/long
  questions, half with the target near the beginning and half near the middle.
  The short prompts contain 516–526 and long prompts 11,984–12,140 actual chat
  tokens. [prompts/calibration_v2.jsonl](prompts/calibration_v2.jsonl) contains
  32 independent harder 513-token questions; manifests record input hashes.
- [benchmark/selected.jsonl](benchmark/selected.jsonl) freezes 20 `qasper_e`
  and 20 `2wikimqa_e` examples. The [selection manifest](benchmark/selection_manifest.json)
  lists IDs, source hashes, length-bin counts, and exclusions. All retained
  prompts fit 16,384 context with at least 2,000 tokens reserved; their range
  is 2,159–14,062 tokens. The official source files are gitignored under
  `benchmark/source/`.
- [PRIMARY_FREEZE.json](PRIMARY_FREEZE.json) records the exploratory designation,
  input and patch-manifest hashes, server settings, five arms, and a 900-second
  per-arm inference limit. It was written before held-out scoring.
- `runs/primary-logprobs-{arm}.jsonl` and `runs/longbench-{arm}.jsonl` contain
  all 48 and 40 completed requests for each arm. The [paired primary table](runs/paired_primary.csv)
  and [paired LongBench table](runs/paired_longbench.csv) expose item-level
  contrasts; paired summary JSON files are in the same directory.

## Host GPU execution and reproduction

The default sandbox mounts a separate `/dev` without NVIDIA device nodes, so
`nvidia-smi` and CUDA inference fail there. Host-device execution reaches the
RTX 5070 and Prism CUDA binaries. The four-item baseline pilot passed at 16K
context with 8,232 MiB sampled peak VRAM; two short requests took about
0.7 seconds and two long requests about 9.6 seconds each.

From the repository root, launch a single baseline server with the same
settings used by the matrix:

```bash
export BONSAI_GGUF=models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf
BONSAI_FAMILY=bonsai2 BONSAI_NGL=99 BONSAI_CTX=16384 \
  BONSAI_SPECULATIVE=0 BONSAI_KV4=0 \
  ./scripts/start_llama_server.sh --parallel 1 -b 512 -ub 512 --reasoning off
```

For the full exploratory run, `run_frozen_matrix.py` instead starts and stops
one model/server at a time on port 18082, verifies the frozen hashes, and
resumes saved item IDs without repeating them:

```bash
python3 analysis/bonsai2/run_frozen_matrix.py
```

It completed 440/440 requests with no errors, OOMs, or time-limit truncation.
Inference time summed to about 470–496 seconds per arm, excluding model load.
The runner used temperature 0, fixed seed 20260929, no thinking, FP16 KV,
512 batch and microbatch, one slot, and no vision projector. Correct-answer
token log probabilities were returned and reconstructed for every primary
response. Raw JSONL records model path, prompt tokens, response, score, and
timings for each item.

Recreate the paired analyses and tables from saved results:

```bash
python3 analysis/bonsai2/analyze_experiment.py --state-arm state-d02 --mlp-arm mlp-d02
python3 analysis/bonsai2/analyze_experiment.py --state-arm state-d03 --mlp-arm mlp-d03
python3 analysis/bonsai2/analyze_longbench.py --state-arm state-d02 --mlp-arm mlp-d02
python3 analysis/bonsai2/analyze_longbench.py --state-arm state-d03 --mlp-arm mlp-d03
python3 analysis/bonsai2/export_paired_tables.py
```

The 24-question pilot did not meet the calibrated-damage or accuracy-sensitivity
conditions for a confirmation run. A later audit also found a unique target
code prefix in all primary prompts; a question-blind regex solves all 48.
Interpret the run using the limits in the
[experiment report](EXPERIMENT_REPORT.md). The benchmark source is the
[official LongBench repository](https://github.com/THUDM/LongBench) and
[dataset](https://huggingface.co/datasets/zai-org/LongBench).

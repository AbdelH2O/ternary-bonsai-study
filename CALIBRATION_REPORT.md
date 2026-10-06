# Bonsai 2 perturbation calibration, 2026-09-29

The host GPU is a GeForce RTX 5070 with 12,227 MiB of VRAM and NVIDIA driver
610.57.04. The normal agent command sandbox mounts a separate `/dev` with
`nodev` and no NVIDIA device nodes, so `nvidia-smi` fails there. Host-device
execution works: the Prism CUDA server loaded the original PQ2_0 checkpoint at
16,384 context with one slot, no vision projector, FP16 KV, and reasoning off.

## Baseline cost gate

The four planned baseline pilot requests all answered correctly. The two short
prompts contained 516 and 526 actual chat tokens and took 0.65 and 0.83 seconds.
The two long prompts contained 11,984 and 12,140 tokens and took 9.53 and
9.63 seconds. Sampled peak VRAM was 8,232 MiB. At this rate, 48 primary
requests cost roughly four minutes per arm, excluding model loads and any
different behavior on remaining cases. Natural-task time has not been measured.
Raw responses and timings are in [runs/pilot-baseline.jsonl](runs/pilot-baseline.jsonl);
VRAM samples are in [runs/pilot_vram.csv](runs/pilot_vram.csv).

## Short-context calibration

All eight separate ~512-token retrieval questions were answered exactly by the
baseline and every tested candidate. A later audit found that these items give
the gold code a unique `C...` prefix while every distractor starts with `D...`;
a question-blind prefix search answers all eight. Their accuracy ceiling is
therefore not meaningful evidence of intact retrieval. For a more sensitive check, the pinned
Prism `llama-perplexity` binary scored eight 512-token chunks from the frozen
calibration corpus, without a saved logit file. Each row below is compared to
the same baseline PPL of 3.4009. Positive excess negative log-likelihood
(NLL) means worse calibration loss.

| Arm | PPL | Excess NLL, nats/token |
| --- | ---: | ---: |
| State 1% | 3.4041 | +0.00094 |
| MLP 1% | 3.4023 | +0.00041 |
| State 2% | 3.3991 | -0.00053 |
| MLP 2% | 3.3913 | -0.00283 |
| State 3% | 3.4523 | +0.01500 |
| MLP 3% | 3.3993 | -0.00047 |
| State 5% | 3.3844 | -0.00486 |
| MLP 5% | 3.3696 | -0.00925 |
| MLP 10% | 3.3989 | -0.00059 |

The 3% state arm worsened six of eight chunk losses; a paired
bootstrap over the eight chunks gives a 95% interval of about +0.0024 to
+0.0262 nats/token for its mean excess NLL. This is exploratory: the same
corpus was used to choose strengths, chunks are few and share repetitive
registry text, and the response-accuracy measure is at a ceiling. The other
strengths are non-monotonic and do not provide a second small damaging state
arm matched by MLP. The 10% MLP adjustment also stays near baseline.

## Independent short-context calibration

We then froze 32 different, harder 513-token questions in
[calibration_v2.jsonl](prompts/calibration_v2.jsonl). Baseline gave 30/32 exact-format
answers; two responses contained the correct code plus extra explanation. The
2% and 3% state and MLP arms also each gave 30/32, with no new failures. On the
30 baseline-correct items, mean excess generated-answer NLL was −0.0031 and
−0.0042 for state 2% and 3%, versus −0.0113 and −0.0082 for MLP 2% and 3%.
Every corresponding paired 95% bootstrap interval included zero. The other
candidates were similarly close to baseline: state 1% and 5% each had one new
exact-format failure, while the MLP arms had none. Full values and intervals
are in [calibration_v2_summary.json](runs/calibration_v2_summary.json).

**Decision:** no two small, measurably damaging and matched strengths were
found in either calibration. Before opening the held-out primary or
LongBench-E scores, we froze equal-delta 2% and 3% state/MLP pairs as an
**exploratory** pilot in [PRIMARY_FREEZE.json](PRIMARY_FREEZE.json). This is an
explicit deviation from the planned calibrated causal comparison. The
[completed run](EXPERIMENT_REPORT.md) can describe sensitivity at those exact
strengths, but cannot determine whether matched recurrent-state errors
accumulate more than MLP errors in general.

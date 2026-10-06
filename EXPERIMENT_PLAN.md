# Experiment plan: do recurrent-state errors accumulate?

## Question and scope

**Hypothesis:** small weight errors in Bonsai 2's recurrent-state control path cause more *additional* damage as context grows than comparable errors in a non-recurrent projection. The first test targets `ssm_alpha.weight` and `ssm_beta.weight` in the 48 linear-attention blocks. In the [Qwen implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modular_qwen3_5.py), these help set state retention and write strength.

We will compare three variants of the **same downloaded `PQ2_0` model**. Qwen FP16 is neither required nor loaded. This tests sensitivity to a controlled perturbation, not whether Prism's particular training or precision choices were optimal.

## Variants and intervention

| Arm | Changed weights | Purpose |
| --- | --- | --- |
| Baseline | None; original GGUF read-only | Reference for all paired comparisons |
| State control | `ssm_alpha.weight` and `ssm_beta.weight` in all 48 linear blocks (23,592,960 BF16 values) | Test recurrent gate sensitivity |
| MLP control | The same number of effective weights, sampled evenly from `ffn_up.weight` in the same 48 blocks | Test whether the effect is specific to the recurrent path |

For both changed arms, apply a deterministic, zero-mean **groupwise multiplicative perturbation** to 128 consecutive weights: multiply each group by `1 + δ` or `1 - δ`, using a fixed seed and equal numbers of signs. The state arm rewrites BF16 values in their existing GGUF slots. The MLP arm changes the FP16 scale of selected `PQ2_0` groups, leaving their ternary codes untouched. This gives the same fractional perturbation at the same 128-weight granularity and the same number of affected weights. It is deliberately an *error injection*, not a faithful reenactment of ternary quantization-aware training.

Use a separate 512-token calibration set to choose perturbation strengths. Try δ = 1%, 3%, and 5%; keep two strengths that produce small, measurable damage without making either variant unusable. If equal δ creates different short-context damage, adjust the MLP δ on the calibration set so both arms have similar short-context loss. Freeze those choices before scoring held-out cases. If the short-context effects cannot be matched within a reasonable strength range, report that limitation rather than forcing a comparison.

Create each variant with `cp --reflink=always` on this Btrfs filesystem and patch only the specified tensor payload bytes. Keep variants in a gitignored `analysis/bonsai2/variants/` directory. Record the original SHA-256, seed, selected tensor names, group indices, δ, file sizes, and patch manifest. Reparse each GGUF; verify unchanged metadata and dimensions, no changed bytes outside the intended tensor ranges, and that a one-question smoke test still produces a sensible answer. Never patch the original.

## Measurement design

### Primary: paired context-length test

Prepare 24 independent, answerable retrieval questions with a known short answer. For each question, build the **same factual target** into a short (~512-token) and long (~12,000-token) context. Balance the target's placement near the beginning and middle; keep the question and answer unchanged. Distractors should be plausible and contain no duplicate answer. Freeze all texts and answers before running a variant.

Run all 48 prompts on each arm, one arm at a time. The primary measure is exact answer accuracy. If the pinned server exposes reliable answer-token log probabilities, also record the correct answer's log probability as a more sensitive secondary measure; otherwise use exact accuracy alone. Use one short output format and identical sampling for every arm. For this mechanistic test, disable or tightly cap reasoning so generation cost and reasoning variation do not hide the context effect. Record actual prompt tokens, response, timings, and any OOM/error for every item.

**Primary contrast:** for each perturbed arm, measure its loss relative to baseline at each length. Then compare the change from short to long between the state and MLP arms. A larger *additional* long-context loss in the state arm supports accumulation. Bootstrap paired questions (not individual length variants) for a 95% interval. With 24 questions this is a pilot; a wide interval means inconclusive. If the pilot is viable, freeze and add 24 more questions for confirmation.

### Secondary: real-task check

Use a preselected subset of [LongBench-E](https://github.com/THUDM/LongBench/blob/main/LongBench/README.md): `qasper_e` for single-document QA and `2wikimqa_e` for multi-document QA. Its published evaluator uses QA F1 for both. Select up to 20 examples from each task *before* looking at treatment scores, stratified across its length bins. Count tokens with the actual Bonsai chat template and include only untruncated examples that fit the available context and output reserve. Publish the selected IDs and exclusion counts; do not call this a full LongBench score. Use the benchmark's fixed prompt and evaluator across all arms, with the same bounded reasoning setting.

This stage asks whether the sensitivity seen in the controlled test also appears on natural questions. It will not identify a specific causal block by itself. Do not run Terminal-Bench or SWE-bench for this first study: their agent scaffolds and long output budgets would dominate the compute budget and add extra sources of variance.

## 12 GB execution profile and cost gate

Use the installed Prism CUDA binaries, text only, with **one model and one server slot at a time**. Start at `BONSAI_CTX=16384`, `--parallel 1`, FP16 KV cache, no vision projector, no speculative decoding, and the same batch settings for every arm. Reserve at least 2,000 tokens for prompt template and output; prefilter examples before scoring. If the baseline cannot load comfortably, try 8K context and rerun the entire design at that limit; do not mix context limits between arms. The demo launcher accepts an explicit variant through `BONSAI_GGUF`.

Before committing to the full matrix, run 2 short and 2 long baseline prompts and record the response `timings` and peak VRAM. Project total time from those measurements, then freeze an item count and run limit. The default target is 48 primary prompts × 3 arms plus at most 40 real examples × 3 arms, sequentially. Stop only for a declared run limit, load failure, or OOM; retain all completed results and report missing items. Avoid full-vocabulary logit dumps: they can consume gigabytes of RAM and disk on this model. The repo's `llama-perplexity` binary can score a separate small calibration corpus without a saved logit file if answer log probabilities are unavailable. [llama.cpp's perplexity guide](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md) explains that metric and its limits.

## Interpretation and follow-up

Evidence **for** the hypothesis: after matching short-context damage, state perturbation has consistently larger excess loss at long context than the MLP control, in both perturbation strengths, and the paired interval is mostly or entirely above zero. A similar direction on the LongBench-E subset strengthens external relevance.

Evidence **against or inconclusive**: effects are equal, reverse, unstable across strengths, or too small for the sample to resolve. One failed prompt, larger group scales, or a higher raw sparsity fraction is not evidence of regional importance. If the broad state arm shows a signal, split it into `ssm_alpha` versus `ssm_beta`, then test the much smaller `ssm_a`/`ssm_dt.bias`, convolution, and norm tensors separately. Those are follow-up experiments; the first stage does not claim to map every layer.

## Deliverables and current status

The experiment should produce: a patch manifest; three validated model paths; a frozen prompt/answer and benchmark-ID manifest; per-request JSONL with variant, seed, prompt tokens, score and timings; paired difference plots/tables; and a short report including uncertainty and failures. The existing [region map](README.md) provides tensor names and offsets.

**Status (2026-09-29):** GPU access works through host-device execution; the default agent sandbox hides `/dev/nvidia*`. The baseline timing/VRAM gate passed, and nine state/MLP candidate checkpoints were created and validated. Two short-context calibrations failed to find the two matched damaging strengths required for the planned causal comparison. Before held-out scoring, the 2% and 3% equal-delta pairs were [frozen](PRIMARY_FREEZE.json) as an explicitly exploratory pilot. All five arms completed the 24 paired primary questions and 40 selected LongBench-E items without request errors. Primary exact accuracy was 24/24 at both lengths for every arm, but a later audit found a unique answer-code prefix that solves all 48 primary prompts without reading the question. The secondary log-probability contrast was tiny and inconsistent across strengths. The natural QA changes were isolated to two Qasper items at 2% state and one 2Wiki item in both 3% arms. See the [experiment report](EXPERIMENT_REPORT.md), [calibration report](CALIBRATION_REPORT.md), and [execution notes](EXECUTION.md). The pre-existing `uv.lock` modification is unrelated and untouched.

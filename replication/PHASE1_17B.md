# Phase 1: matched 1.7B comparison with Prism's Ternary-Bonsai-1.7B — 2026-10-02

**Update:** the bounded 0.8B training test is reported in [QAT08.md](QAT08.md).

**Decision (frozen rules):**
- **Not non-inferior; large gap.** The best one-shot arm (GPTQ-style ternary in an H1024 folded basis, selected on validation) is **+1.31 nat/token** worse than Prism's model on twelve held-out books [+1.04, +1.58].
- **It is functionally broken on tasks.** It answers MMLU-Redux at chance level (23.5% vs 47.7%), gets 0.0% on GSM8K (vs 74.7%), and fails retrieval (54% vs 100%).
- **Prism's model stays close to full precision on the same ancestor and in the same file format.**
  - +0.22 nat/token on books;
  - −1.4 MMLU points (interval includes zero);
  - −7.3 GSM8K points.

Prism's weights moved far from the ancestor: 89% sign agreement, and a relative weight change of 0.90 versus 0.54 for our arm. Their per-layer output errors against the FP layers are also much *larger* than ours. Together this points to end-to-end training as the main ingredient, not a better rounding rule (🟡 inferred). The token embedding is the exception: it matches a data-free BitNet-style absmean projection of the ancestor in 99.83% of entries.

This is the Phase 1 step of the [replication plan](REPLICATION_PLAN.md), following the 0.8B [data-aware arm](GPTQ_PILOT.md) and its [refinements](GPTQ_V2.md). The [protocol](results/q17b/protocol.json) (sha `b70dec33…`) fixed the arms, evaluation sets, selection rules, margins and readings before any arm was built or scored. Evidence tags: 🟢 measured; 🟡 inferred; 🟣 proposed.

## What was compared

All arms derive from the pinned **Qwen/Qwen3-1.7B** (`70d244cc`). They share Prism's exact tensor policy: 197 PQ2_0 tensors (every projection plus the tied embedding) and 113 F32 norms, about 463 MB.

| Arm | What it is |
|---|---|
| `fp` | Qwen3-1.7B, F32 GGUF from the pinned converter (the duplicate `lm_head`, bit-identical to the embedding, is dropped so the output is tied as in Prism's file) |
| `prism` | Prism's published `Ternary-Bonsai-1.7B-PQ2_0.gguf`, unchanged (ships YaRN rope scaling, factor 4) |
| `prism_noyarn` | The same file with rope scaling forced to none, matching the ancestor's rope configuration |
| `absmax` | Pinned `llama-quantize` PQ2_0 (data-free) |
| `ls` | Least-squares ternary per group (data-free; the best 0.8B data-free rule) |
| `gptq` | The 0.8B-selected GPTQ-style rule, unrotated: 64×512 calibration tokens, layer-sequential |
| `gptqh` | Same rule in a folded H1024 Hadamard basis (Bonsai 2's block size) |

🟢 **Format check.** Running the pinned converter plus `llama-quantize --pure --token-embedding-type PQ2_0` on Prism's unpacked FP16 checkpoint reproduces **all 310 tensor payloads of Prism's published PQ2_0 file byte for byte**. Our arms are therefore in exactly Prism's format and policy.

🟢 **Fold check.** Base and H1024-folded F32 differ by at most 0.034 nat on any token on the GPU runtime (window means within 0.0003). The pinned runtime's Hadamard path works for Qwen3.

**Evaluation**, all through the pinned runtime on CUDA with identical token IDs for every arm. The softmax is normalized over the first 151,669 tokens because Prism trims Qwen's padded vocabulary.
1. **Book NLL:** 1,024-token slices at 25% and 65% of 15 freshly downloaded Gutenberg books, none used before: three for validation, twelve held out. Targets are positions 64–1023. The book is the unit.
2. **MMLU-Redux 2.0:** 5,330 items with `error_type == ok`; zero-shot, chat template with thinking off; answer = the next-token letter. Reports accuracy and the gold-minus-best-wrong log-prob margin.
3. **GSM8K:** the full 1,319-item test set; greedy, thinking off, up to 1,024 new tokens; boxed or last-number parsing.
4. **Retrieval:** 200 synthetic registries of 30 "Name: code" lines. Margin = NLL(another entry's code) − NLL(gold code).

**Selection** was written to [selection.json](results/q17b/selection.json) before any held-out metric was computed:
- method `gptqh` (validation 5.003 vs 11.876 for `gptq`);
- Prism reference `prism_noyarn` (3.581 vs 3.597).

## Results

🟢 **Held-out** ([results.json](results/q17b/results.json)):

| Arm | Book NLL (nat/token) | MMLU-Redux acc. | MMLU margin (nat) | GSM8K | Retrieval margin (nat) / acc. |
|---|---:|---:|---:|---:|---:|
| FP | **3.067** | 49.0% | +1.38 | **82.0%** | **15.31** / 99.5% |
| Prism (no YaRN, reference) | 3.285 | 47.7% | −0.09 | 74.7% | 8.45 / 100% |
| Prism (as published) | 3.298 | **49.2%** | −0.09 | — | 8.46 / 100% |
| **GPTQ-H1024 (selected)** | 4.591 | 23.5% | −1.39 | 0.0% | 0.18 / 54% |
| GPTQ, unrotated | 11.647 | 22.8% | −1.51 | 0.0% | 0.04 / 48% |
| Least squares | 13.212 | 22.9% | −1.46 | — | 0.13 / 55% |
| Absmax | 19.446 | 24.9% | −2.73 | — | −0.39 / 46% |

Chance level is 25% on MMLU and 50% on retrieval.

🟢 **Primary comparison, selected minus reference** (95% intervals; books t over 12, items normal, registries t over 200):

| Metric | Difference | Margin | Pass |
|---|---|---|---|
| Book NLL | +1.307 [+1.038, +1.575] | ≤ +0.10 | no |
| MMLU-Redux accuracy | −24.2 points [−26.1, −22.3] | ≥ −3.0 | no |
| GSM8K accuracy | −74.7 points [−77.0, −72.3] | ≥ −3.0 | no |
| Retrieval margin | −8.28 nat [−8.76, −7.79] | ≥ −0.25 | no |

- Reading: **large_gap** (book NLL difference > 0.50).
- Secondary gap closure on book NLL: (ls − selected)/(ls − prism) = **0.87**. Read this with the task results: the arm closes most of the *loss* gap but none of the *capability* gap.

🟢 **Prism versus FP:**
- Books: +0.218 [+0.166, +0.269];
- MMLU: −1.4 points [−2.9, +0.2];
- GSM8K: −7.3 points [−9.6, −4.9];
- Retrieval margin: −6.85 nat, though still at 100% accuracy.

As a harness check, this local GSM8K setup gives FP 82.0 and Prism 74.7, against published 83.1 and 74.2. Local MMLU letter scoring (49.0) is far below the publisher's generative MMLU-Redux (66.8), so the MMLU numbers compare arms only.

🟢 **YaRN effect on Prism's file:**
- turning it off improves book NLL slightly (−0.014 [−0.021, −0.006]);
- it costs 1.5 MMLU points [−2.2, −0.8];
- retrieval is unchanged.

The validation rule picked no-YaRN; neither choice changes any decision.

🟢 **Generation failure mode:** 1,317 of 1,319 GPTQ-H1024 GSM8K outputs hit the token limit in a loop of whitespace and markdown fragments. The unrotated arm repeats "the". A 4.6 nat/token model can still assign plausible next-token probabilities to book text and yet be unable to follow a chat prompt.

## Static weight comparison (descriptive)

[weights.json](results/q17b/weights.json), [analyze_weights_17b.py](analyze_weights_17b.py). Layer output error = tr(ΔHΔᵀ)/tr(WHWᵀ) with Hessians from the **FP** ancestor on the calibration windows, so every arm is judged on the same inputs. Our GPTQ arms were built with sequential packed-input Hessians.

| | Prism | LS | GPTQ | GPTQ-H1024 |
|---|---:|---:|---:|---:|
| Zero fraction (parameter-weighted) | 38.3% | 47.1% | 46.7% | — |
| Nonzero sign agreement with ancestor | 89.2% | 100% | — | — |
| Support overlap (Jaccard) with Prism | 1 | 0.56 | 0.54 | — |
| Relative weight change ‖Q−W‖/‖W‖ | 0.90 | 0.44 | 0.56 | 0.54 |
| Median layer output error | **0.70** | 0.17 | 0.13 | **0.044** |

- 🟢 **Projections moved far.**
  - Prism's sign agreement with Qwen3 is 84–91% by tensor kind and falls with depth (0.88 in early blocks, 0.825 in block 27).
  - Its zero set overlaps a top-magnitude rule at Prism's own per-group count only 0.53–0.59.
  - Zeros per group vary widely (5th–95th percentiles about 38–59 of 128 in early blocks), unlike Bonsai 2's fixed 42.
- 🟢 **Layer outputs far from FP, yet the best model.** Prism's per-layer errors are 5–16× those of our arm, worst in `ffn_up` (1.70) and `attn_v` (1.61). Prism's F32 norms are essentially unchanged (relative change ≤ 0.002, except 0.033 on block 0's input norm).
- 🟡 **Interpretation.** A model whose layers don't reproduce the FP layer outputs, but which reproduces the FP model's behavior, was optimized end to end: training or distillation, not layer-wise reconstruction. Part of the `ffn_up`/`attn_v` error is plausibly harmless rescaling against their partners (`ffn_down`, `attn_output`), which this metric can't separate out.
- 🟢 **The embedding is a data-free projection.**
  - Signs match 100%, and the support equals the top-|w| set (Jaccard 0.9993).
  - BitNet b1.58's absmean rule fits closely: scale = mean |w| per 128-group rounded to BF16, then stored as FP16 (bit-exact in 95.7% of groups); code = clip(round(w/scale), −1, 1), matching 99.83% of entries.
  - All mismatches lie at |w|/scale ≈ 0.5, on both sides of the threshold, and no tie rule removes them. So it is a near-exact description (slight latent drift or arithmetic differences), not a recovered exact mapping.
  - This is a candidate mapping for Question A, but only for the embedding.

## What this means for the replication

- 🟢 **On the same ancestor and in the identical file format, one-shot rounding loses most of the model's capability, and Prism's model does not.** The 0.8B ceiling (about +1.5 nat/token above FP) repeats at 1.7B (+1.52) and turns out to mean chance-level tasks.
- 🟡 **Prism's quality almost certainly comes from end-to-end training.** The static evidence and the quality evidence agree, and cheaper ingredients don't close the gap: better rounding, calibration, rotation.
- 🟢 **Rotation matters much more at 1.7B than at 0.8B** (validation 5.00 vs 11.88 unrotated). That fits Qwen3 having outlier channels that a Hadamard fold spreads out. Rotation is still not enough by itself.
- 🟣 **Recipe clues worth carrying into a training experiment:**
  - an absmean-style ternary quantizer (it describes the untouched embedding almost exactly);
  - a free per-group zero count (gen-1) versus Bonsai 2's fixed 42;
  - training only the projections while the embedding stays a fixed projection;
  - initialization from GPTQ-H1024 or the absmean projection.
- Per the plan's §5, the Phase 2 entry criterion is met: a predeclared quality shortfall that cheap reconstruction cannot close. Training at 1.7B needs more than the local 12 GB GPU (full Adam is about 27 GB before activations). The 0.8B bounded training test remains the locally feasible next step. A rented GPU would allow the 1.7B version with Prism as a direct yardstick.

**Limits:**
- one gen-1 pair (Qwen3-1.7B, full attention, unrotated reference); nothing here tests Bonsai 2's GDN, rotation-plus-training, or 42-zero recipe;
- MMLU uses letter logits and GSM8K uses one prompt, both different from the publisher's EvalScope harness;
- deterministic methods, no seeds;
- public-domain books may be in pretraining data (affects all arms alike);
- MMLU and GSM8K intervals treat items as independent.

## Deviations and incidents (all before or independent of the decisions)

- **Folded build memory fix.** The `gptqh` build first ran out of GPU memory in a whole-embedding fold assertion. I chunked that check and rebuilt only `gptqh`; the method is unchanged. So `build_gptq.json` records the earlier builder hash; both build records match the protocol and file hashes checked at selection.
- **GSM8K interruption.** The session ended during GSM8K after `fp` and `prism_noyarn`; the run resumed for the remaining arms.
- **Server parse failure.** llama-server parses the finished text into a chat message even for raw `/completion` requests, and it threw on some `gptqh` outputs. [gsm8k_17b.py](gsm8k_17b.py) now regenerates such items in streaming mode and keeps the raw token text, flagging them `parse_fallback`. This affected **6 `gptqh` items**; `fp`, `prism_noyarn` and `gptq` had none. With all `gptqh` items scored 0.0, it doesn't affect the result.
- **Static analysis fixes.** The weight analysis needed row-chunking for the embedding and a row cap for our padded-vocabulary arms; metric definitions are unchanged. It also adds the folded arm through its unfolded effective weights; its codes are not code-comparable with Prism's.

## Reproduction

From `analysis/bonsai2/replication` with the project venv; CUDA for builds and scoring. Model files live in the git-ignored `work/qwen3_17b/`.

```bash
python3 make_folded_17b.py                       # tied + H1024-folded HF checkpoints
# pinned converter (work/llama.cpp-842b188…/convert_hf_to_gguf.py --outtype f32) on base_tied, folded, bonsai_unpacked
# bin/cuda/llama-quantize --pure --token-embedding-type PQ2_0 X-f32.gguf X-pq2.gguf PQ2_0 8
python3 freeze_17b.py                            # already run; refuses to overwrite
python3 make_gptq_17b.py ls gptq gptqh && python3 validate_17b.py
python3 score_17b.py validation && python3 score_17b.py select    # select refuses to overwrite
python3 score_17b.py heldout && python3 score_17b.py gsm8k && python3 score_17b.py analyze
python3 analyze_weights_17b.py
```

The scorer is [score_17b.cpp](score_17b.cpp), built against the pinned headers and `bin/cuda/libllama.so`. GSM8K generations for each arm are in `results/q17b/gsm8k/`.

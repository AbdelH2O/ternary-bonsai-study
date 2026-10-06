# 0.8B data-aware (GPTQ-style) arm — 2026-10-01

**Update 2026-10-01:** refinements (42-zero budget, 4x calibration, within-layer sequencing) are reported in [GPTQ_V2.md](GPTQ_V2.md).

**Decision (frozen rule 1):** data-aware ternary reconstruction closes **78%** of the gap between the best data-free packed arm and full precision on five held-out books. Under the predeclared rule, it goes into the matched 1.7B comparison. The packed model is still **+1.49 nat/token** worse than full precision, so this is a large improvement, not usable quality. It says nothing about Bonsai 2's quality or PrismML's training recipe.

This amends the [0.8B quality pilot](QUALITY_PILOT.md). The [protocol](results/gptq_pilot/protocol.json) fixed everything below before any arm was built:

- the calibration windows and the method;
- three new sealed held-out books;
- the selection rule;
- the decision thresholds.

Evidence tags: 🟢 measured in this run; 🟡 inferred; 🟣 proposed.

## Method

- **Source:** the same pinned `Qwen/Qwen3.5-0.8B` ancestor, packed into the same 151 PQ2_0 tensors, with the 36 BF16 recurrent gates and 133 F32 tensors byte-identical to the original pilot's templates. File sizes are unchanged (213.7 MB), under the 220 MB ceiling.
- **Calibration:** 64 windows of 512 tokens (32,768 tokens), evenly spaced through *Emma*, *Little Women* and *Middlemarch*. None of these is a validation or held-out book ([window IDs](results/gptq_pilot/calibration_ids.npy), hashed in the protocol).
- **Objective:** for each linear layer, minimize the output error `tr((W − Q) H (W − Q)ᵀ)`, where `H = Σ x xᵀ` over that layer's calibration inputs. The standard GPTQ procedure does this column by column: damping is 1% of the mean diagonal; there's no activation reordering; blocks are 128 columns, aligned with PQ2_0 groups.
- **Ternary rule:** at the start of each 128-weight group, the scale is the independent least-squares ternary scale of the error-updated weights, rounded to FP16. Each code is `clamp(round(w/s), −1, 1)`.
- **Order:** layers are processed sequentially, so each decoder layer is calibrated on inputs produced by the already-packed earlier layers. The tied embedding and output head come last, using the final-norm Hessian.
- **Rotated arm:** the same objective in the H512 basis (`H_rot = R H Rᵀ`) applied to the folded weights. A self-test confirmed that this transform preserves the objective to within 7×10⁻⁸ relative.
- **Implementation:** [make_gptq_pilot.py](make_gptq_pilot.py) on the local RTX 5070 took about 80 s per arm. The [validation](results/gptq_pilot/validation.json) confirms tensor types and offsets, that the 169 non-PQ2 payloads are byte-identical to the template, and that 1,208 sampled groups per file decode identically in the Python and pinned C codecs. Scoring used the original pilot's unchanged CPU scorer and slices ([score_gptq_pilot.py](score_gptq_pilot.py)).

## Results

🟢 **Validation** (*Dracula*, *Pride and Prejudice*; mean NLL in nat/token; the six earlier arms are copied from the original [selection](results/quality_pilot/selection.json)):

| Arm | Unrotated | H512 folded |
|---|---:|---:|
| Full precision | 3.0414 | 3.0414 |
| Absmax (pinned quantizer) | 16.2932 | 16.0260 |
| Fixed 86 nonzeros | 12.3406 | 11.3003 |
| Least squares | 10.6614 | 12.3210 |
| **GPTQ (data-aware)** | **4.6717** | **4.4269** |

The frozen rule selected **folded GPTQ** ([selection](results/gptq_pilot/selection.json), written before any new held-out book was scored).

🟢 **Held-out, five books** (mean of two slices per book, nat/token; [full results](results/gptq_pilot/results.json)):

| Book | FP | Least squares (prior selection) | Folded GPTQ | Gap closed |
|---|---:|---:|---:|---:|
| *Frankenstein* † | 3.328 | 10.292 | 4.687 | 0.805 |
| *Sherlock Holmes* † | 3.439 | 10.272 | 4.934 | 0.781 |
| *Monte Cristo* | 3.519 | 10.953 | 5.102 | 0.787 |
| *War of the Worlds* | 3.471 | 9.950 | 4.786 | 0.797 |
| *Wuthering Heights* | 3.270 | 10.052 | 4.979 | 0.748 |
| **Mean** | **3.406** | **10.304** | **4.898** | **C = 0.784** |

† These two books were opened once before, for the FP and least-squares arms, so this is a second look at them. The three new books were sealed until selection, and their closure (0.75–0.80) matches the reused pair.

- GPTQ minus least squares: **−5.41 nat/token**, descriptive 95% t interval over books [−5.80, −5.01].
- GPTQ minus FP: **+1.49 nat/token**.
- Re-scoring the original held-out slices reproduced the earlier FP and least-squares scores exactly (maximum difference 0).

🟢 **In-sample diagnostics** (calibration windows, PyTorch; not evaluation):
- Median relative layer output error: 0.028 (unrotated) and 0.023 (rotated) for GPTQ, versus 0.158 and 0.148 for the least-squares rule on the same Hessians.
- Calibration NLL: 3.51 at FP; 4.96 unrotated packed; 4.71 rotated packed.
- Zero fraction: 47% (unrotated), 45% (rotated), versus 32.8% in Bonsai 2's sampled groups.

## Interpretation

- 🟢 On this model, *how* ternary codes are chosen matters more than any data-free rule tested so far. Data-aware error compensation removed most of the loss that simple rounding caused.
- 🟢 The rotation helped under GPTQ (validation 4.43 vs 4.67) but hurt under least squares (12.32 vs 10.66). Rotation interacts with the assignment rule, so it can't be judged on its own. This comes from two validation books; the unrotated GPTQ arm was not scored on the held-out books.
- 🟡 This fits Phase 0's finding that Bonsai 2's zeros are not simply the smallest-magnitude weights. A data-aware choice of which weights to zero is a plausible part of the recipe, though not a demonstrated one.
- 🟡 Our arm zeros 45–47% of weights, where Bonsai keeps about 33% (42 of 128). A GPTQ variant with a fixed 42-zero budget per group is a natural untested arm, and it would also move this toward Bonsai's observed format.
- 🟡 A +1.49 nat/token residual is still large (roughly 4.4× the full-precision perplexity). Closing it will likely need more than one-shot reconstruction: more calibration, within-layer sequencing, a zero budget, or training.

**Limits:**
- one 0.8B model;
- 128-token context;
- natural-text NLL only, no retrieval or tasks;
- five books as the units, a descriptive interval;
- 32K calibration tokens;
- one deterministic method with no seed variation;
- the embedding was calibrated last while earlier layers used full-precision embeddings.

## Next options

1. **Planned route (rule 1):** the matched 1.7B comparison against Prism's Bonsai-1.7B, with GPTQ as an arm, a larger frozen evaluation, and predeclared non-inferiority margins.
2. **Cheap 0.8B refinements first:**
   - a 42-zero budget;
   - more calibration tokens;
   - within-layer sequencing;
   - activation-order columns.

   Each needs a new frozen amendment, and selection must move to fresh validation books, because the current two have now been used for eight arms.
3. **Bounded training:** if refinements plateau well above full precision, test a short quantization-aware training run on 0.8B, starting from the GPTQ weights, with measured memory and throughput.

## Reproduction

With the repository root as the working directory and CUDA available for the build:

```bash
.venv/bin/python analysis/bonsai2/replication/freeze_gptq_pilot.py        # already run; refuses to overwrite
.venv/bin/python analysis/bonsai2/replication/make_gptq_pilot.py base folded
python3 analysis/bonsai2/replication/validate_gptq_pilot.py
python3 analysis/bonsai2/replication/score_gptq_pilot.py validation
python3 analysis/bonsai2/replication/score_gptq_pilot.py select            # refuses to overwrite
python3 analysis/bonsai2/replication/score_gptq_pilot.py heldout
python3 analysis/bonsai2/replication/score_gptq_pilot.py analyze
```

Model files are in the git-ignored `work/qwen35_08b/` (`base-pq2-gptq.gguf`, `folded-pq2-gptq.gguf`). Their SHA-256 hashes, build timings and per-tensor diagnostics are in [build_base.json](results/gptq_pilot/build_base.json), [build_folded.json](results/gptq_pilot/build_folded.json) and [results.json](results/gptq_pilot/results.json).

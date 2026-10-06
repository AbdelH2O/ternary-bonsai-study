# 0.8B GPTQ refinements — 2026-10-01

**Update 2026-10-02:** the matched 1.7B comparison against Prism's model is reported in [PHASE1_17B.md](PHASE1_17B.md).

**Decision (frozen rules):**
- **No material gain.** The frozen selection kept the v1 data-aware arm, so the primary comparison is zero by construction. Its validation lead over the within-layer v2 arm was only 0.0006 nat/token.
- **Bonsai's 42-zero format costs quality here:** +0.45 nat/token.
- **The training entry criterion is met.** The best held-out arm is still +1.60 nat/token above full precision.

The factorial secondary analysis shows the v2 reconstruction (4× calibration text plus within-layer sequencing) does help, by −0.15 nat/token averaged over budgets, but the gain is too small to change the conclusion. Nothing here speaks to Bonsai 2's quality or PrismML's recipe.

This amends the [data-aware arm](GPTQ_PILOT.md). The [protocol](results/gptq_v2/protocol.json) fixed the arms, rules and thresholds before any build. It uses **nine freshly downloaded Gutenberg books** that no earlier arm had seen, replacing the reused ones: three for validation and six held out. Evidence tags: 🟢 measured; 🟡 inferred; 🟣 proposed.

## Design

A 2×2 in the folded H512 basis (the basis selected last time). Every arm uses the same source, templates and tensor policy (151 PQ2_0, 36 BF16, 133 F32; 213.7 MB).

| | v1 reconstruction (64 windows, per-layer Hessians) | v2 reconstruction (256 windows, four stages per layer) |
|---|---|---|
| **Free zero count** (least-squares scale, rounding) | `folded-pq2-gptq` (existing) | `folded-pq2-gptqv2` |
| **Exactly 42 zeros per group** (keep the 86 largest updated magnitudes at group start, scale = their mean) | `folded-pq2-gptq42` | `folded-pq2-gptq42v2` |

- **v2 staging:** input projections, then the attention output projection, then MLP gate/up, then MLP down. Hessians are recollected after each stage, with the earlier stages already packed.
- **Calibration text:** all calibration windows come from the same three calibration books as before, and don't overlap.
- **Implementation checks:** [make_gptq_v2.py](make_gptq_v2.py) reuses the unchanged v1 helpers. With no budget it reproduces the v1 routine bit for bit, and a self-test confirmed exactly 42 zeros per group with the budget. v1-reconstruction arms took about 83 s on the RTX 5070; v2 arms about 13–14 min.
- **File checks** ([validation](results/gptq_v2/validation.json)):
  - template-identical layout;
  - 169 byte-identical non-PQ2 payloads;
  - 99.998% of budget-arm groups with exactly 42 zeros (the rest are all-zero groups from input columns with no activation);
  - 1,208 sampled groups per file identical in the Python and pinned C decoders.

## Results

🟢 **Validation** (*Great Expectations*, *Jane Eyre*, *Tom Sawyer*; mean NLL, nat/token; [selection](results/gptq_v2/selection.json), written before any held-out score):

| Arm | Validation NLL |
|---|---:|
| Full precision | 3.4176 |
| **v1, free (selected)** | **4.7242** |
| v2, free | 4.7248 |
| v2, 42 zeros | 5.1519 |
| v1, 42 zeros | 5.3755 |

🟢 **Held-out, six books** (mean of two slices per book; [full results](results/gptq_v2/results.json)):

| Book | FP | v1 free | v2 free | v1 42-zero | v2 42-zero |
|---|---:|---:|---:|---:|---:|
| *Moby Dick* | 3.818 | 5.986 | 5.925 | 6.376 | 6.176 |
| *A Tale of Two Cities* | 3.436 | 4.780 | 4.639 | 5.421 | 5.117 |
| *Treasure Island* | 3.533 | 5.324 | 5.358 | 5.928 | 5.627 |
| *The Picture of Dorian Gray* | 3.213 | 4.956 | 4.975 | 5.818 | 5.624 |
| *The Time Machine* | 3.625 | 5.167 | 5.060 | 5.479 | 5.397 |
| *Heart of Darkness* | 4.024 | 5.361 | 5.266 | 5.833 | 5.458 |
| **Mean** | **3.608** | **5.262** | **5.204** | **5.809** | **5.567** |

**Effects** (book-level 95% t intervals, six books, descriptive):
- **42-zero budget minus free** (averaged over both reconstructions): **+0.455 nat/token [+0.272, +0.638]**. Reading: *costs*. The budget made every book worse at both reconstruction levels.
- **v2 minus v1 reconstruction** (averaged over budgets): **−0.150 [−0.217, −0.084]**. Within the free-budget pair alone it is −0.058: v2 better on four books, worse on two. Within the budget pair it is −0.243.
- **Best held-out arm minus FP:** +1.596 nat/token (v2 free), above the 1.0 threshold.

The held-out NLL levels are higher than in the previous amendment because the books differ (FP 3.61 here vs 3.41 there). Compare arms only within one book set.

## Interpretation

- 🟢 Forcing Bonsai 2's exact format (42 zeros per group) onto one-shot data-aware reconstruction is clearly worse than letting GPTQ choose the zero count. Our free arms zero about 45% of weights.
- 🟡 Bonsai 2 keeps about 33% zeros yet retains most of its base model's quality (published retention 98.2% at 27B). If its 42-zero budget were imposed on a one-shot method like ours, it would be a handicap. So Bonsai's quality most likely comes from something that works *within* that constraint, plausibly training, rather than from better one-shot rounding. That is an inference: it comes from one small model with a different architecture scale, and it is not a measurement of Prism's method.
- 🟢 More calibration and within-layer sequencing help modestly, and mostly when the support is constrained. In the free-budget setting, one-shot reconstruction appears close to its plateau: v2 moved the mean by only −0.06 while quadrupling calibration and using four times as many Hessian passes.
- 🟡 Taken with the previous amendment, one-shot reconstruction on this 0.8B model plateaus around +1.5 to +1.6 nat/token above full precision on natural text. Closing that gap needs a different kind of method.
- 🟢 The frozen selection picked v1 over v2 by 0.0006 nat/token on validation, but v2 was 0.058 better on held-out. Neither passes the frozen −0.10 threshold for "refinements help". This is a reminder that tiny validation differences can't order arms.

**Limits:**
- one 0.8B model;
- 128-token natural-text NLL, no retrieval or tasks;
- six held-out books, descriptive intervals;
- deterministic methods, no seeds;
- famous public-domain books may appear in the model's pretraining data, affecting all arms alike;
- the 42-zero arm fixes its support at each group start, and other budget-constrained assignments (for example, re-choosing the support after error feedback) were not tested.

## Next step

Under the frozen rule, the training entry criterion is met. The candidate experiment 🟣 would be:
- a short, bounded quantization-aware training (or distillation) run on this 0.8B model;
- initialized from the best GPTQ weights;
- one arm with a free zero count and one with the 42-zero budget, so it also tests whether training removes the budget's cost;
- measured memory and throughput, a GPU-hour ceiling, and the same frozen evaluation discipline on fresh books.

Whether to run it is your decision.

## Reproduction

From the repository root (CUDA for the build):

```bash
.venv/bin/python analysis/bonsai2/replication/freeze_gptq_v2.py      # already run; refuses to overwrite
cd analysis/bonsai2/replication && ../../../.venv/bin/python make_gptq_v2.py && python3 validate_gptq_v2.py && cd -
python3 analysis/bonsai2/replication/score_gptq_v2.py validation
python3 analysis/bonsai2/replication/score_gptq_v2.py select           # refuses to overwrite
python3 analysis/bonsai2/replication/score_gptq_v2.py heldout
python3 analysis/bonsai2/replication/score_gptq_v2.py analyze
```

Raw Gutenberg downloads and their stripped bodies are in `results/gptq_v2/books/` and hashed in the protocol. Model files are in the git-ignored `work/qwen35_08b/`. Build records with per-tensor diagnostics are in `results/gptq_v2/build_*.json`.

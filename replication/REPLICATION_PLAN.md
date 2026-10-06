# Bonsai ternary replication plan

**Goal update (2026-10-06, user decision):** the headline goal is now Question B: match Prism's results on the same base model, in the same format, under the same evaluation, by any method. Question A (recovering Prism's process) continues only where it suggests ways to improve results. See [QAT17B_DESIGN.md](QAT17B_DESIGN.md) v0.2 and [RESEARCH_LOG.md](RESEARCH_LOG.md).

**Status:** revised after independent [Astra](reviews/REVIEW_astra.md) and [Fable](reviews/REVIEW_fable.md) reviews, 2026-09-30. Execution is in progress: the [Phase 0 record](PHASE0_RESULTS.md) contains the complete 449-tensor protected scan, six streamed matrices including a frozen held-out set, codec/layout fixtures, and native F32 fold tests on small hybrid models. The [0.8B hybrid quality pilot](QUALITY_PILOT.md) has now screened absmax, fixed-86, and least-squares PQ2_0 assignments in unrotated and H512 bases; all tested packed arms remain far from FP quality. A frozen [data-aware GPTQ-style arm](GPTQ_PILOT.md) then closed 78% of the least-squares-to-FP gap on five held-out books (4.90 vs 10.30 vs 3.41 nat/token), meeting its predeclared rule for inclusion in the 1.7B comparison; a +1.49 nat/token residual remains. A frozen [refinement amendment](GPTQ_V2.md) on nine fresh books then found no material gain from 4x calibration and within-layer sequencing (-0.06 nat/token in the free-budget pair), a clear cost from imposing Bonsai's 42-zero budget (+0.45 [+0.27, +0.64]), and a best arm still +1.60 nat/token above FP, meeting the predeclared entry criterion for proposing bounded training. The frozen [1.7B Phase 1 comparison](PHASE1_17B.md) then found a **large gap** to Prism's Ternary-Bonsai-1.7B on the same ancestor and in byte-identical format: the best one-shot arm (GPTQ-style, H1024 folded) is +1.31 nat/token worse on twelve held-out books and at chance on MMLU-Redux, GSM8K and retrieval, while Prism stays near FP; Prism's weights moved far from the ancestor (89% sign agreement) with larger layer output errors than ours, pointing to end-to-end training, and its embedding matches a data-free absmean projection in 99.83% of entries. The Phase 2 entry criterion is met. A frozen [bounded 0.8B quantization-aware distillation test](QAT08.md) then showed training is the missing ingredient for loss: 65.5M tokens took the ternary student from GPTQ's 4.81 to 3.80 nat/token on twelve held-out books (FP 3.28; 66% of the gap closed, still improving), Bonsai 2's 42-zero budget cost nothing after training (−0.025), and raw-text retrieval recovered (94.5%), but chat-format capability did not (MMLU at chance, GSM8K degenerate), pointing to chat-formatted distillation data as the next ingredient. The exact 27B FP-function gate and the rented 1.7B training run remain open. This document authorizes no model-weight download or experiment by itself; the user separately requested execution. Chat amendment status (2026-10-03): Chat capability did not meet the predeclared recovery rule. Book cost versus the plain-text 42-zero arm is +0.005 nat/token [-0.012, +0.021] (see [QAT08_CHAT.md](QAT08_CHAT.md)). Read-out status (2026-10-06): the diagnostics ([DIAG_READOUT.md](DIAG_READOUT.md), amended to `readout_broken`) found the chat student could not read out multiple-choice answers. A frozen 0.8B follow-up ([QAT08_MCU.md](QAT08_MCU.md)) found that multiple-choice teacher data (arm M) meets the frozen recovery gate: MMLU-Redux 35.9% [34.6, 37.2], MMLU-fresh 35.2%, GSM8K 16.3%. That is partial relative to FP (48.2%, 44.3%, 55.8%), and H_M itself failed on binding (82% < 90%). Trained norms/gates and continued training showed no MMLU effect; continued training gained GSM8K +3.9 [+1.8, +5.9]. The frozen recommendation is `rent_1p7b_with_recipe` (arm M). Renting and the 1.7B run are not yet authorized.

## 1. Questions and standard of evidence

**Question A — effective mapping.** Which specified rules can reproduce, or closely approximate, Bonsai's stored ternary codes, scales, and behavior from a public Qwen checkpoint? Static tests can reject a particular mapping. They cannot uniquely recover PrismML's training history, data, objective, schedule, compute, or an unreleased ancestor.

**Question B — functional replication.** Can we produce a model in another supported target at a comparable packed bit budget with predeclared, acceptably small quality loss, verified runtime behavior, memory, and latency? Success on Question B does not prove we recovered PrismML's original process.

The earlier [Bonsai 2 investigation](../LEARNING_REFERENCE.md) studied recurrent-state fragility in the compact checkpoint. Its position-matched scoring, retrieval controls, and freeze discipline carry over; its causal findings do not identify a quantization recipe.

Evidence tags: **[V]** checked in a local file or pinned source; **[P]** reported by a paper or model card; **[R]** preliminary reviewer probe, pending full reproduction; **[D]** experiment design; **[U]** unknown or not verified. A conclusion must keep its evidence tag and the exact checkpoint, tensor, and sample scope.

## 2. Known contract, artifacts, and preliminary clues

### 2.1 Format and runtime [V/P]

- Bonsai 2 27B has 402 PQ2_0 tensors, including 401 folded projections and the stored token embedding, plus 96 BF16 and 353 F32 tensors. The 26,238,464 higher-precision parameters are about 0.0976% of the model. The protected set includes recurrent gates and several normalization/state-path tensors (Bonsai 2 whitepaper Table 2; local GGUF header).
- Each PQ2_0 group has 128 signed ternary codes and an FP16 scale. A prior tensor-balanced sample found exactly 42 zeros in 97.2% of 12,864 groups. This is an observed **zero budget**, not a requirement for balanced positive and negative counts, and not yet a model-wide rate ([structural analysis](../README.md)).
- The local GGUF records a 1024-wide normalized Sylvester Walsh–Hadamard transform, explicit signs for widths 5120, 6144, and 17408, and grouped GDN V handling. With column activations, define R = H S / √1024, where S is the diagonal sign matrix. The runtime computes Wt(Rx), so the matching stored projection is Wt = W Rᵀ. A row-major embedding table is folded in the same right-multiplication direction, Et = E Rᵀ; its runtime **lookup restoration** applies H then S. Flip signs before applying H to each 1024-column block when constructing W Rᵀ. Derive and test this convention from the pinned runtime, not by maximizing correlation with Bonsai.
- The pinned converter supports folded weights for Llama, Qwen3, Qwen3-MoE, Qwen3.5, Qwen3.5-MoE, and Qwen3-Next, subject to tensor roles, widths divisible by the block size, tie rules, and runtime version. Architecture membership alone does not guarantee a valid export.
- The pinned PQ2_0 quantizer can ternarize arbitrary input; packing is exactly lossless only for correctly grouped, already-ternary values whose scale survives FP16 export. The public folded F16 GGUF provides stronger evidence that at least its sampled groups were ternary **before PQ2_0 packing** [R]. It is a folded re-quantization source, not an unrotated ancestor or latent training checkpoint.
- Gen-1 Bonsai retained a small high-precision normalization/scale tail; the earlier claim that it retained none was wrong. Its benchmark suite and recipe may differ from Bonsai 2. Do not compare its headline average with Bonsai 2's as if they were the same evaluation.

### 2.2 Public artifacts

| Artifact | Role and handling |
|---|---|
| Local Bonsai 2 PQ2_0 GGUF (7.2 GB) | Primary codes, scales, higher-precision tensors, and runtime metadata [V] |
| Qwen/Qwen3.8-27B (55.56 GB indexed artifact, 18 shards) | Public candidate ancestor; includes vision and MTP tensors absent from the 851-tensor language GGUF [R]; start with metadata and verified HTTP ranges |
| Ternary-Bonsai-2-27B-F16.gguf (53.8 GB, remote) | Folded F16 pre-packing reference; use bounded ranges for decoder and scale checks [R], not an initial full download |
| Bonsai 2 MLX hadamard.json | Small consistency check for signs and metadata; derived from the GGUF pipeline, so not independent provenance [R] |
| Ternary-Bonsai-1.7B-unpacked and Qwen3-1.7B | Cheap script and harness smoke test; tied embedding/output and variable zero counts mean it cannot validate the Bonsai 2 recipe [R] |
| Ternary-Bonsai-27B-unpacked and Qwen3.6-27B | Closer gen-1 hybrid-family comparison, without the disclosed Bonsai 2 runtime rotation; sample by ranges before downloading whole checkpoints [R] |

Pin repository revisions and validate range responses, lengths, tensor offsets, dtype, and hashes where available. A server that ignores Range must not cause an accidental whole-shard download. The public Qwen checkpoints are candidate ancestors, not proven exact starting points.

### 2.3 What the reviews already observed [R]

Fable compared **sampled** Bonsai 2 F32/BF16 tensors in blocks 0, 3, 31, and 63 with converter-transformed Qwen3.8 values: every sampled tensor class differed, often slightly and with high correlation ([Fable's review](reviews/REVIEW_fable.md)). Its few-row ternary probes and gen-1 probes are useful for choosing tests, not population estimates. Changed high-precision values reject a simple unchanged-parameter projection of **that public revision**, assuming correct mapping. They do **not** prove end-to-end QAT: an intermediate fine-tune, selective update, local reconstruction, functional reparameterization, or another ancestor can leave overlapping fingerprints. The all-tensor T1 scan is still required.

## 3. Phase 0 — establish the mapping and measure the fingerprint

### 3.1 Gate 0: independently prove layout and arithmetic [D]

Before comparing model similarity:

1. Freeze the pinned Prism release, Qwen revision, GGUF hash, tensor inventory, conversion manifest, and test fixtures. Build an explicit Hugging Face source name → canonical role/layout → GGUF name map. The converter's folded set takes **Hugging Face source names**, not GGUF weight names. Assert that the folded GDN out-projection follows the grouped-V branch.
2. Reproduce the converter's dtype and mapping sequence: BF16→F32 before modifications, −exp(A_log), dt-bias rename, conv squeeze, norm +1 except GDN norm, and the Qwen3.8 V-head permutations. Qwen3.8 already stores in_proj_qkv and in_proj_z separately; do not apply the Qwen3-Next qkvz split to it. Preserve q-projection output-gate row interleaving. Exclude MTP and vision tensors absent from the language GGUF.
3. Check independent Python versus pinned-runtime identities on labelled basis vectors and random FP tensors: (W Rᵀ)(Rx) = Wx; embedding restoration; GDN tiled→grouped activation handling before its folded out-projection. Cover rectangular tensors, all three sign widths, 1024/128 boundaries, and non-symmetric sign patterns. Compare full-precision logits on short prefill and decode paths before any ternary comparison.
4. Verify the PQ2_0 decoder with adversarial synthetic groups and bounded ranges of the published folded F16 GGUF: codes, signs, group scale, 42-zero examples, and effective decoded values. Test all-zero groups and scales near FP16 rounding boundaries. Verify final tensor types and counts; a pack/decode round trip using only our own implementations is insufficient to establish runtime rotation correctness.
5. Keep wrong direction, shuffled signs, wrong V ordering, and other-block comparisons as **negative controls**. Report their empirical distributions. Do not assume 50% sign agreement or treat correlation with the public ancestor as an alignment proof. If independent identities pass but a real tensor differs strongly, retain it as a real difference or unresolved ancestor issue.

The manifest used for our own export has the converter's required signed-transform name, status, axis, roles, and schema; its string is not copied verbatim from the GGUF metadata. Freeze the exact quantizer command and override default embedding/output promotion and 2-D gate quantization. Assert PQ2_0 for the intended ternary tensors and BF16/F32 for every protected tensor.

### 3.2 Gate 1: cheap streamed Bonsai 2 comparison [D]

No full Qwen or F16 download is required for this gate.

1. Read index/config and scan **all 449 non-ternary GGUF tensors** against the mapped public Qwen candidate by bounded tensor ranges (reviewer estimate: about 57 MB of useful data). T1 reports per tensor: exact element count, changed fraction, maximum and mean absolute difference, norm-normalized difference, quantiles, and correlation. Compare exact bits after specified transforms where arithmetic permits; bound exp/libm error with fixtures. Include a correctly versus incorrectly reordered V control. Do not use unstable per-element relative error near zero.
2. Analyze two or three complete representative matrices by row/1024-block/128-group tiles, starting with a residual-writing FFN matrix, an attention output matrix, and a GDN out-projection. T2–T4 report both raw and, where algebraically justified, norm-γ-corrected public candidates. Also check possible joint channel sign symmetries. Use the residual-writing matrices as primary sign evidence.
3. From remote F16 GGUF ranges, check selected published ternary groups against the local PQ2_0 decode and read the upstream F16 scale. This is a packing/codec check, not evidence of the unpublished optimizer.
4. Decide whether more blocks or a full 55.56 GB artifact would answer a **specified unresolved question**. Record that question, required tensors, bytes, peak-RSS estimate, and stop condition before any expansion. If ranges fail, a selected shard is an explicit alternative; block 0 and embeddings are in different shards.

### 3.3 Tests and permitted conclusions [D]

| Test | Measure | Permitted conclusion |
|---|---|---|
| T1 — protected tensors | Exact and quantitative comparison after validated transforms, by block and kind | Which tensors differ from the **tested public revision**; no unique training-history label |
| T2 — signs | Nonzero-code sign agreement by tensor, base-magnitude decile, and distance from zero; joint signs and γ-adjusted candidates separately | Reject or retain a specified sign-assignment rule; describe depth/type pattern |
| T3 — support | Actual positive/zero/negative counts; exact-support rate and Jaccard against fixed-86 top magnitude, unconstrained least-squares support, and prespecified importance rankings; ties at support boundary | Reject or retain each specified zero-selection rule; fixed 42 alone explains no selection process |
| T4 — scales | Compare stored FP16 bits with prespecified exported candidates; report signed LS optimum for Bonsai's fixed codes: max(0, Σᵢ tᵢwᵢ / Σᵢ tᵢ²), with a separate all-zero case | Reject or retain a scale/export formula conditional on the chosen codes; exact match does not show how codes were chosen |
| T5 — held-out reconstruction | For ordinary projections, compare ‖(W Rᵀ − Ŵ) R X‖², equivalently ‖(W − ŴR)X‖²; use the actual post-norm/post-gate input and grouped order for GDN, plus whole-block tests | Functional output-error evidence for a **specific** candidate rule, never a unique PTQ/QAT discriminator |
| T6 — map | Per-block, tensor-kind, and parameter-weighted summaries of T1–T4 | Where a reproducible method must spend effort; descriptive |

For a deterministic candidate, require exact signs/support and exported scale bits outside a separately justified arithmetic or tie ambiguity set. Report the excluded fraction and every mismatch. Do not turn a universal 99% or 99.9% agreement threshold into a historical diagnosis. Freeze candidate formulas and tie rules on discovery tensors, then assess on held-out tensors. Keep per-tensor and parameter-weighted summaries distinct; billions of correlated coefficients are not independent trials.

T5 is conditional on a candidate worth testing, not on proving a historical hypothesis. A BF16 27B base forward does not fit the local 12 GB GPU or 30 GB RAM. Bonsai-captured activations are a cheaper **student-distribution proxy** and must be labelled as such; base activations need a separately profiled streaming/offload path or larger hardware. Evaluate covariance-aware error, not channel energy alone, across held-out text domains and with document-level uncertainty.

### 3.4 Gen-1 comparisons and Phase 0 decision [D]

Use 1.7B to exercise readers, code matching in a common basis, tied-output export, and a feasible reference harness. Its closeness to public Qwen3-1.7B is a *closest tested candidate* result, never ancestor proof. Its variable zero counts do not set Bonsai 2 thresholds. Stream selected gen-1 27B versus Qwen3.6-27B tensors to compare a closer hybrid architecture, but report differences between releases rather than pooling their coefficients.

Phase 0 ends with a table of **candidate mappings retained or rejected**, the observed fingerprint and uncertainty, and the next cheapest quality experiment. A changed T1 never automatically triggers QAT; cheap projection and local reconstruction remain testable. An exact static fit establishes an effective mapping for those weights, not Prism's history. An unreleased ancestor can remain unresolved.

## 4. Phase 1 — inexpensive functional reconstruction

**Target.** Qwen3-1.7B is an engineering and quality benchmark because a public Bonsai-1.7B reference exists. It is not a proxy for Bonsai 2's hybrid GDN, rotation-plus-training interaction, or 98%-retention claim. Add an appropriately sized hybrid bridge before making any gen-2-specific claim; target availability and local feasibility must be checked first.

**Experimental arms.** Every arm starts from the same hashed FP ancestor and uses a declared tensor-precision policy, group axis, clipping, rounding, ties, scale dtype, and sign seed. Cross at least:

- Assignment: pinned quantizer absmax rule; independent-group least-squares ternarization; fixed 86-nonzero magnitude rule; any Phase 0 rule retained on held-out tensors.
- Rotation: none versus the validated folded transform, with several frozen sign seeds. Rotate the FP ancestor **before** ternarizing; do not rotate an already ternary checkpoint and treat that as the same arm.
- Protected tensors: apply a common baseline policy, then change one defined exception set at a time where that set exists in the target. Qwen3 lacks GDN gates, so its exception experiment cannot test Bonsai 2's recurrent policy.
- Optional data-aware reconstruction: compare a fully specified calibration rule with the above cheap arms. Calibration data remains separate from evaluation.

Compare codes only when basis, grouping, row order, and protected-tensor policy match. Across different rotations, compare effective unrotated matrices, output error, and behavior.

**Evaluation path.** Use the pinned Prism runtime for packed arms through a common request harness. Use a matching FP reference path and establish its numerical/per-task difference before comparing arms. An unfolded FP16 checkpoint (effective WtR, with embedding restored consistently) can be a stock HF/vLLM diagnostic; first measure the F16 unfold rounding floor. It is not a packed-size or latency result. The full publisher EvalScope/vLLM protocol may be unaffordable or incomplete locally; exact reproduction is a diagnostic, not a prerequisite that can block all matched local comparisons.

Primary metrics, reported separately:

1. Position-matched natural-text NLL at predeclared nested history lengths, nat/token, paired by book.
2. Retrieval gold-versus-wrong answer margin, nat/answer, on new exchangeable registries with shortcut and value-swap controls.
3. A feasible, version-pinned task subset, with per-task scores, prompts, parsers, output limits, thinking mode, and decoding fixed for all arms. The detailed six-task table in the gen-1 paper gives **58.47** for Bonsai-1.7B and **66.57** for Qwen3-1.7B; the prose headline says 57.5. Record that discrepancy. The gen-1 protocol uses thinking disabled and greedy decoding; Bonsai 2 uses a different thinking-mode evaluation. Do not validate an aggregate alone or mix the two suites.

Before arm scoring, predeclare a **paired non-inferiority margin** for each primary metric and a packed-byte ceiling. For lower-is-better loss, require Larm − Lreference ≤ δL; for higher-is-better utility, require Uarm − Ureference ≥ −δU, with an interval over independent units. Report absolute scores and both differences to FP16 and to Bonsai-1.7B; FP16 need not win every task. A secondary recovery ratio versus a naive arm may be shown only with an explicit orientation and a denominator distinguishable from zero; never average NLL, margins, and task points into one number. Choose the method on validation data and assess that one choice once on held-out test data. A gate cannot pass merely because a gap was measured.

## 5. Phase 2 — training only if the quality experiment warrants it

Enter after Phase 1 shows a predeclared quality shortfall that cheap scale-only, selective-weight, or local block reconstruction cannot close. A static T1 difference alone is not an entry criterion.

Freeze one complete training recipe before its comparison: ancestor and teacher, trainable tensors and latent dtype, fixed-sparsity/rotation policy, scale parameterization, STE behavior, optimizer and schedule, sequence lengths, KL temperature and direction, hard-label loss, corpus mixture, evaluation checkpoints, seeds, and stop budget. Keep training/calibration text deduplicated from evaluation. Profile student and teacher peak memory, tokens/s, activation and optimizer working sets, checkpoint size, and GPU-hours in a short run before choosing a token ladder. The local 12 GB GPU does not by itself establish which parameter count can train; 1.7B full Adam is roughly 27 GB for parameters/gradients/master/optimizer state **before** teacher, activations, and fake-quant buffers.

Measure quality and a drift fingerprint (sign flips, support Jaccard, scale changes) against tokens and seeds. Such a curve can compare **our own** methods and bound a cost on the tested scale; matching Bonsai's drift cannot recover Prism's token count. Three budgets on 0.6B cannot justify extrapolating a 27B hybrid recipe. Set a GPU-hour ceiling, confidence criteria, and a held-out larger-scale validation point before any extrapolation.

## 6. Phase 3 — unseen-target transfer and packaging

1. Freeze an unseen target and its FP checkpoint before method selection. Preflight every tensor role, width divisibility, tied embedding/output rule, MoE/router/shared-expert case, tokenizer IDs, context, protected tensors, runtime graph path, and expected packed bytes. A small full-attention target alone does not establish gen-2 hybrid transfer.
2. Independently prove FP-function preservation for the fold on labelled tensors and full-model logits, then apply the selected assignment/training rule to the **folded FP ancestor**. Make a folded HF checkpoint and valid hadamard_packing.json. Use explicit quantizer flags, including pure/type overrides for embeddings, output and BF16/F32 gates. Test PQ2_0 and PTQ1_0 codec behavior on compact fixtures first.
3. Gate conversion on final tensor types, dimensions, byte budget, decoded codes/scales/effective values, and held-out full-versus-folded-versus-packed layer/logit comparisons through prefill, decode, and chunk boundaries. Diagnose manifest, codec, dtype, and reference errors before attributing a mismatch to the runtime graph.
4. Test the frozen target on the same metric families plus long-context generation and multi-turn/tool-use tasks when those capabilities are claimed. Record actual disk bytes, peak RAM/VRAM, prefill and decode timings, and quality relative to its FP baseline under a predeclared tolerance. Treat vision separately if the target includes it.

Functional replication requires a useful result at the stated bit and resource budget on an unseen target. It does not require, or imply, identification of Prism's original optimizer.

## 7. Freezes, resources, and decisions

| Gate | Required record | Next action |
|---|---|---|
| G0 — contract | Source-name map, independent FP identities, GDN/embedding fixtures, decoder and type checks | Fix any contract failure before interpreting real weight differences |
| G1 — streamed fingerprint | Full protected-tensor scan; selected complete matrices; candidate-rule fit and held-out mismatch counts | Expand only for a named unresolved question and a measured resource budget |
| G2 — local reconstruction | Common-ancestor factorial arms, paired quality and byte results, selected method on held-out data | Try specified cheap reconstruction or profile Phase 2 only if the quality margin fails |
| G3 — training | Frozen recipe, measured memory/throughput, quality-vs-cost curve and held-out scale check | Continue only within the frozen compute ceiling |
| G4 — transfer | Unseen target, exact types/bytes, independent function checks, quality and runtime resource measures | Claim functional replication only if all prespecified tolerances pass |

Hash downloaded files, revisions, checkpoints, corpora, token IDs, scripts, manifests, and results. Freeze formulas, exclusions, uncertainty units, margins, and analysis code before each held-out gate. Preserve prior versions and date amendments. Report books, registries, benchmark items, and training seeds as their respective independent units; tokens and adjacent coefficients are not independent replicates. Tag every conclusion measured, inferred, or proposed.

**Resource planning:** the previously reported free-space number is stale; check disk and available RAM before any download. A whole Qwen3.8 artifact is 55.56 decimal GB, not the first step; the MLX weights are unnecessary for the small metadata consistency check. A decoded F32 endpoint matrix can be about 5.09 decimal GB **alone**, so tile by rows and 1024/128 boundaries and measure peak RSS including source, output, sorting, and code buffers. A full 27B BF16 forward exceeds local 12 GB VRAM and 30 GB RAM without streaming/offload or larger hardware. Profile every training and evaluation path before promising throughput or cost. Keep decimal GB and binary GiB explicit.

**Remaining limits:** the exact ancestor could be unreleased; parameter symmetries can mask weight changes; the publisher's full benchmark harness or data may not be recoverable; a matched small-model result may fail on GDN, tools, long context, or vision; and legal rights for a commercial implementation have not been established by the model licenses. Record these as limits on claims, not as inferred outcomes.

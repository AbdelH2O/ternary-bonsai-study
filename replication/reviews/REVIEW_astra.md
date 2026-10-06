# Adversarial review of the Bonsai replication plan

The plan is a useful starting point for **characterizing a released checkpoint and finding a competitive compression recipe**, but it is not yet a valid way to identify PrismML's historical procedure. Its central decision table makes non-identifiable causal claims, its alignment gate can reject the very drift it seeks to detect, and its proposed converter setup contains a concrete naming error that would corrupt the GDN comparison. The format/header claims are largely sound, and the measurement discipline is unusually good, but the benchmark target, packing instructions, success criteria, and resource model need revision before expensive work begins. I would proceed with a small, corrected Phase 0 focused on rejecting explicit candidate algorithms; I would not authorize the current H1–H4 classification or use it to require end-to-end QAT.

## Evidence scope

Reviewed on 2026-09-30. `P` below denotes `analysis/bonsai2/replication/REPLICATION_PLAN.md`; `S` denotes `/tmp/llama.cpp-842b1880415d6f508f03b789e5ce70194def7bfd`. Source citations such as `S/conversion/qwen.py:622` refer to that supplied local tree. It has no usable Git metadata (`git -C … rev-parse HEAD` failed), so its correspondence to the named commit is **unverified independently**; the actual files were inspected.

I read the entire plan, the prior structural analysis and methodology brief, converted all four root whitepapers with `pdftotext -layout`, and checked the relevant paper sections and source paths. PDF text line citations refer to `/tmp/review-astra-<PDF-basename>.txt`, reproducible with that command. I read only the GGUF header/descriptors, not its weight payload. Small HF API/config/index/rotation-metadata requests downloaded no model weights. No inference, quantization, training, or new weight-comparison experiment was run. Publisher benchmark scores remain published claims, not reproduced measurements.

## Findings ranked by severity

### Critical 1 — H1–H4 are overlapping histories, not identifiable alternatives

**Plan:** §§2, 3.3–3.4, 5, 10; P:60–63, 104–123, 150.

**Problem:** T1 equality does not imply “never trained,” and T1 inequality or small-magnitude sign flips do not imply end-to-end training. QAT can freeze every high-precision tensor, train only scales or a subset of matrices, and finish by recomputing closed-form scales. Local block reconstruction can optimize normalization/gate parameters without end-to-end training. Data-free iterative optimization, weight equalization, channel permutation, and a fine-tuned intermediate base are omitted or misclassified. H4 is not exclusive of H1–H3: any quantizer can start from another base, and a nearby fine-tune need not have poor agreement everywhere. Neither a drift map nor its magnitude estimates training compute.

**Evidence:** These implications are explicit in P:104–119. A direct counterexample is training latent weights while freezing norms, then projecting onto ternary codes; T1 stays exactly equal under H3. A sufficiently small update can stay inside the same quantization cells, leaving *all* T1–T4 unchanged. Conversely, rescaling a norm and inversely rescaling its downstream projection changes T1 without changing the full-precision function. Even an exact closed-form projection conditional on the published signs would not show that selecting those signs was data-free: fixed at inference does not mean randomly chosen without calibration. The papers do not supply training history that resolves these ambiguities.

**Fix:** Separate axes: candidate ancestor/revision; functional reparameterization; assignment/scale objective; data use; optimization scope; and export arithmetic. Treat tests as rejecting specified mappings, not identifying unique histories. Replace the decision table with conclusions such as “compatible with this exact deterministic projection,” “requires additional transformations or optimization relative to this ancestor,” and “unresolved.” Keep hybrid methods and unknown ancestors explicit. Try inexpensive reconstruction regardless of a purported H3 result; reserve QAT for a demonstrated quality/cost need.

### Critical 2 — The folded-tensor names proposed for the converter are in the wrong namespace

**Plan:** §3.1 step 2; P:87.

**Problem:** Setting the converter's folded set equal to GGUF `prism.hadamard.weight_names` will not preserve grouped `ssm_out` columns as claimed. The converter tests **HF source names**, not GGUF names. This is a direct route to a convincing but incorrect drift signature.

**Evidence:** `S/conversion/base.py:620–632` reads source names from `hadamard_packing.json`. `S/conversion/qwen.py:568–570` matches source-name suffixes; at :617–629 a failed match reorders `out_proj` columns. The actual GGUF lists `blk.0.ssm_out.weight`; the pinned HF index lists `model.language_model.layers.0.linear_attn.out_proj.weight`. These do not suffix-match. The independent MLX metadata uses another wrapper, `language_model.model.layers.0.linear_attn.out_proj.weight`, also requiring canonicalization. `S/conversion/base.py:720–739` maps source names to GGUF names only later.

**Fix:** Build an explicit, checked HF→canonical-source→GGUF manifest, including splits/fusions and wrapper removal, then supply the proper source-name folded set. Assert the branch actually sets `_hadamard_gdn_v_grouped`, and test it with labeled head/channel fixtures. Import the applicable converter class and preprocessing pipeline, not just `modify_tensors`: `S/conversion/base.py:1068–1081` promotes BF16 to F32 before modification. The current 27B index already has separate `in_proj_qkv` and `in_proj_z`; do not unconditionally apply the Qwen3-Next fused-`qkvz` branch.

### Critical 3 — The positive alignment control is circular, and the similarity gate censors drift

**Plan:** §§3.1–3.2 and G0b; P:88, 95–98, 197.

**Problem:** Packing and recovering numbers transformed by one's own implementation validates serialization, not whether the transform matches the runtime. A wrong direction or axis can pass perfectly. Choosing the direction with the best Bonsai correlation fits the convention to the outcome. Requiring similarity to the base before interpreting a tensor excludes genuinely changed or unrelated tensors; “alignment failed” cannot always mean “fix the pipeline.” H4 becomes difficult or impossible to reach under its own gate.

**Evidence:** The runtime determines the convention without statistical selection. `S/src/llama-graph.cpp:1561–1572` permutes tiled V features to grouped order, multiplies by signs, then applies normalized H. At :2398–2408, embedding lookup applies H first, then signs. `S/src/llama-model.cpp:2054–2064` constructs Sylvester H explicitly; :2122–2131 configures the GDN permutation. With column activations, define `R = H S / sqrt(n)`. The correct stored projection is `W Rᵀ`; a row-major embedding table also stores `E Rᵀ`, since the looked-up column is restored by `Rᵀ`. The inverse *lookup operation* does not imply the opposite table-folding direction.

**Fix:** Before any ternarization, compare independent Python and pinned-runtime operations on labeled basis vectors and random inputs: `(W Rᵀ)(R x) = W x`, embedding lookup restoration, and GDN tiled→grouped→rotation→matmul. Cover rectangular matrices, all three widths, 1024-block boundaries, 128-group boundaries, and signed non-symmetric rotations. Use an ordinary FP reference and an independently folded FP reference before testing packed tensors. Move the Phase 3 forward/logit gate into Phase 0. Establish layout from these tests; report actual checkpoint similarity separately, retaining low-similarity tensors. Negative controls should be empirical nulls, not an assumed universal 50%: signs can be imbalanced, nearby layers related, and wrong axes dimensionally impossible or accidentally equivalent.

### Major 4 — T5 has a basis error as written and cannot discriminate PTQ from QAT

**Plan:** §3.3 T5, §3.4; P:108, 111, 118.

**Problem:** `W′` and the decoded Bonsai matrix live in rotated input coordinates, but `X` is described as the base model's input activations. Multiplying their difference by unrotated X is wrong. Even corrected, lower reconstruction error is compatible with PTQ, local training, QAT, distillation, and a lucky data-free rule. Failure to improve on today's base activations does not show that the method was not output-error-driven on its own calibration distribution or student trajectory.

**Evidence:** The sign/H/permutation order is fixed by `S/src/llama-graph.cpp:1561–1578`. For ordinary projections, the comparison is `||(W Rᵀ − Ŵ) R X||_F²`, equivalently `||(W − Ŵ R) X||_F²`. For `ssm_out`, inputs must be the actual post-normalization/post-gating features with the correct V-head order. P:108 itself admits “H2 (or H3 with data),” contrary to the stronger decision-table routing. Channel energy measures only the diagonal of `X Xᵀ`; correlated features can dominate reconstruction error.

**Fix:** Recast T5 as held-out functional reconstruction evidence. Use normalized error, paired document-level uncertainty, several data domains, and full second-moment structure where feasible. Compare original Bonsai scales and separately refitted scales, holding zero budgets/bit budgets constant. Include data-free baselines and a known local reconstruction baseline. Distinguish base-input, quantized-prefix-input, and whole-block/end-to-end tests. No T5 outcome should uniquely select H2 or H3. It is also useful as an early implementation/quality diagnostic, not only after every unspecified data-free rule has failed.

### Major 5 — T4's “least-squares optimum given codes” is mathematically wrong when signs disagree

**Plan:** §3.3 T4; P:107.

**Problem:** For fixed signed codes `t`, the unconstrained least-squares scale is `sum(t_i w′_i) / sum(t_i²)`, not the mean absolute base weight over nonzeros. The latter is correct only when every nonzero sign matches. The intended difficult cases explicitly contain sign flips. A nonnegative-scale constraint additionally clips the signed optimum at zero. All-zero groups need separate treatment.

**Evidence:** Expanding `sum_i (w′_i − s t_i)²` and differentiating gives the signed formula. P:105 anticipates sign mismatches, so the simplifying condition cannot be assumed. `S/ggml/src/ggml-quants.c:126–129` stores a nonnegative FP16 absolute-maximum scale.

**Fix:** Report both the optimum for Bonsai's actual codes and the optimum for newly assigned sign-consistent codes. Freeze any fitted constant `c` on discovery tensors and test it on held-out tensors; fitting one constant per group would make `absmax*c` tautological. Test FP16-representable neighboring scales under the actual objective, not merely an unrounded formula. Add a decisive cheap baseline: exact independent-group least-squares ternarization. For sorted descending magnitudes `a_j`, choose support size `m` maximizing `(sum_{j≤m} a_j)²/m`; its scale is their mean. Compare this unconstrained optimum with the fixed-42-zero optimum and signed-rank/balanced-code rules.

### Major 6 — Similarity thresholds conflate numerical ambiguity, model drift, and statistical evidence

**Plan:** §§3.2–3.4, §11 question 3.

**Problem:** No universal 99% or 99.9% threshold identifies a process. FP16 scale rounding normally does not change code signs or the magnitude ordering of the original base weights. BF16 makes source values discrete, producing ties; unknown pre-export precision and transform arithmetic are separate uncertainties. T3 conditions on Bonsai's observed zero count, so even perfect support agreement does not explain how that count was chosen. T4 formula searches and selecting “best” controls create multiple-comparison/selection effects.

**Evidence:** The codec computes codes using the F32 `d` before storing its FP16 conversion (`S/ggml/src/ggml-quants.c:126–142`). Thus scale rounding and code assignment are distinct. The old sampler is tensor-balanced, not parameter-balanced (`analysis/bonsai2/README.md:30–34`). T3 and T4 definitions are P:106–107.

**Fix:** For a fully specified deterministic candidate, demand exact codes on every numerically unambiguous tested position and exact exported scale bits under the replicated dtype chain. Use separate arithmetic fixtures to bound transform error. Report sign disagreements outside a predeclared near-zero uncertainty band; report top-k equality modulo ties/uncertainty at the support boundary. Publish the excluded fraction. If invoking an unknown higher-precision ancestor, bound that uncertainty separately; do not silently absorb it into tolerance. Use absolute and norm-normalized errors for T1, with quantiles and counts, not unstable relative errors near zero. Report positive/zero/negative histograms jointly, rank violations, per-tensor and parameter-weighted summaries, and held-out mismatch rates. Cluster uncertainty by meaningful blocks/documents as appropriate; billions of correlated coefficients are not independent trials proving historical provenance.

### Major 7 — “Lossless packing” is conditional, and upstream ternarization is inferred rather than proved

**Plan:** §§1.2, 3.2, 6; P:31, 95, 160–161.

**Problem:** The packing routine also ternarizes arbitrary finite real inputs. Its existence therefore does not prove that ternarization occurred in an upstream HF checkpoint. Exact value recovery additionally requires an FP16-representable scale, compatible group boundaries, and the intended tensor type. An arbitrary F32 scale is rounded; an all-zero group cannot preserve an independently chosen nonzero latent scale.

**Evidence:** `S/ggml/src/ggml-quants.c:120–142` takes a maximum and rounds any input vector; :129 explicitly rounds the scale to FP16. PTQ1_0 does the same at :2205–2218. `general.name=Hf` and `general.basename=folded` were confirmed in the header but are labels, not provenance records. More seriously, `S/src/llama-quant.cpp:489–505` chooses **Q4_K for untied token embeddings** under bare PQ2_0/PTQ1_0 selection. Type overrides and `--pure` control this at :680–719. The default quantization eligibility logic at :286–366 does not generally preserve `ssm_alpha`/`ssm_beta` just because the input is BF16.

**Fix:** Mark upstream ternarization as an inference. Specify the exact conversion/quantization commands, explicit embedding/output/tensor types, and BF16/F32 exception map. Pre-round group scales to the intended export format and test degenerate groups. Assert the final header's per-tensor types, dimensions, counts, and byte budget, then compare effective decoded values and codes. `--pure` can suppress automatic embedding promotion but still needs explicit high-precision exceptions. Test both PQ2_0 and PTQ1_0 codecs on compact adversarial fixtures before exporting a whole model.

### Major 8 — The gen-1 pilot is useful engineering work, but cannot validate the gen-2 pipeline or ancestor

**Plan:** §§1.1, 1.3, 3.0, 4; P:25, 50, 73–78.

**Problem:** The pilot does not exercise packed decoding, Hadamard direction, GDN grouping, `A_log`, qwen35 norm offsets, or the hybrid architecture. It therefore cannot exercise “every analysis script.” Better matching one of two related Qwen releases identifies the closer candidate, not the actual ancestor. Stock HF execution rules out a required missing online transform, but does not rule out offline function-preserving parameter reparameterizations. The premise that gen-1 had no high-precision tensors is factually false.

**Evidence:** `ternary-bonsai-8b-whitepaper.pdf` §2.1 (text:179–182) explicitly retains normalization parameters in higher precision. `bonsai-27b-whitepaper.pdf` text:283–289 likewise retains a normalization/scale tail. The supplied 1.7B HF config declares ordinary `Qwen3ForCausalLM`, 28 layers, and tied embeddings; 27B declares a 64-layer qwen3_5 hybrid with untied embeddings. T1 will inspect a much narrower parameter class in the pilot.

**Fix:** Keep Phase 0a as a modest, time-boxed gen-1 study with no recipe conclusion carried into gen-2 by default. Add an independent synthetic gen-2 layout fixture plus an early real 27B comparison covering one linear block, one full-attention block, and endpoints before fully pursuing the small-model recipe. Label the winner of the two-base test “closest tested ancestor.” Correct the high-precision claim and inventory actual exceptions rather than requiring every matrix to be ternary in advance.

### Major 9 — The 1.7B quality target is internally inconsistent in the paper, and the harness gate is underspecified

**Plan:** §§4, 9, 10; P:127, 143, 188, 199.

**Problem:** The headline 57.5 is not the six-benchmark average supported by the detailed tables. A tolerance centered on it could send the project into an endless “fix harness” loop. Reproducing an aggregate also does not establish correct scoring: offsetting benchmark errors can cancel. The small-model protocol differs from the gen-2 thinking-mode protocol.

**Evidence:** `ternary-bonsai-8b-whitepaper.pdf` headline text:40 says 57.5, but Table 7 (text:359–373) says **58.47** for Ternary Bonsai 1.7B and **66.57** for Qwen3-1.7B. The six applicable Table 10 entries (text:469–476) reproduce those means: `(52.9 + 50.8 + 74.2 + 51.8 + 70.1 + 51.0)/6 = 58.4667`. Its §3 directs readers to Appendix B of `1-bit-bonsai-8b-whitepaper.pdf`. That appendix specifies EvalScope 1.4.2/vLLM 0.15.1 (text:686–693), **thinking disabled and greedy decoding** for these six tasks (:728–748), 13 single-turn BFCL subsets (:790–794), code-extraction hotfixes (:804–809), and an external judge fallback (:811–817). Gen-2 uses thinking/xhigh (`bonsai-2-27b-whitepaper.pdf` text:279–285).

**Fix:** Register the table-derived per-task targets and record the headline discrepancy. Pin dataset versions, prompts, templates, parsers, BFCL subset weights, token limits, runtime dtypes, and judge behavior. Separate an honest local matched-reference comparison from exact reproduction of an incompletely recovered publisher harness. If the latter remains unavailable, report that limitation instead of assuming every mismatch is an implementation bug. The external judge/software/container requirements also need reconciliation with §8's HF-only network budget. Availability of the publisher's exact hotfixes and evaluation repository is **unverified** here.

### Major 10 — Phase 1 does not define “gap closure,” and the gates can pass without useful quality

**Plan:** §§4, 7, 10; P:137–146, 200–202.

**Problem:** “Fraction of the FP16→Bonsai gap closed” lacks an origin and denominator. Matching Bonsai, approaching FP16, and improving over naive ternary are different objectives. FP16 is a reference, not a guaranteed upper bound on every metric. Ratios become unstable when references tie or reverse order. “Gap closure is measured” and “QAT ladder shows gap closure” do not impose a practically meaningful success threshold. Selecting the best arm on the same evaluation set overstates performance.

**Evidence:** P:146 supplies no formula or threshold. P:200–202 require measurement/packing rather than non-inferiority. Even the paper's MuSR scores favor Bonsai 1.7B over Qwen (50.8 versus 50.1, Table 10), demonstrating that per-task “upper bound” language can fail.

**Fix:** Make the primary target a paired non-inferiority comparison with Bonsai at a fixed packed-size/precision budget, using separately chosen tolerances in nat/token, nat/answer, and task points. Report absolute scores and differences. For optional recovery versus naive ternary N, define, for a higher-is-better utility U, `C = (U_arm − U_N)/(U_Bonsai − U_N)`; matching Bonsai is 1 and naive is 0. For a residual-loss ratio versus full precision F, use `R = (L_arm − L_F)/(L_Bonsai − L_F)` and call it that; matching Bonsai is 1 and matching F is 0. Do not report either ratio when its denominator is indistinguishable from zero, and do not clip values outside [0,1]. Predeclare family aggregation, independent sample counts, power/equivalence margins, and fallback rules. Select methods on validation data and evaluate the chosen method once on held-out test data. Preserve paired book/registry uncertainty and keep it distinct from seed-to-seed training variability.

### Major 11 — Phase 1's arms and code comparisons mix incompatible bases and precision policies

**Plan:** §4 arms and code-level agreement; P:131–144.

**Problem:** The ladder cannot isolate interactions among rotation, the assignment rule, and precision exceptions. “All matrices” usually already excludes 1D norms, making the proposed norm-restoration arm redundant unless specified otherwise. Qwen3 lacks the GDN tensors that motivate most gen-2 exceptions. Rotated ternary codes cannot meaningfully be compared coordinate-by-coordinate with gen-1 unrotated codes. Rotating an already-ternary checkpoint and requantizing also differs from rotating the FP ancestor and quantizing once.

**Evidence:** The architecture and precision maps differ as documented in finding 8 and `analysis/bonsai2/README.md:21–28`. P:133 adds rotation to “Arm 2,” without defining which weights are rotated. The runtime's basis contract is explicit, so coordinate equality across different rotations is not a similarity measure of the same coefficient.

**Fix:** Generate every arm from the identical hashed FP ancestor. Specify clipping/rounding/tie rules and tensor policy. Run at least a crossed rule×rotation comparison with several frozen sign seeds; evaluate exceptions only where they actually differ. Compare exact codes only in the same basis, grouping, and row order. For other arms compare effective unrotated matrices, output error, and behavior. Include a gen-2-style hybrid model bridge, if an appropriate small artifact is available, before claiming transfer of the protected recurrent-path recipe; availability and suitability of such a target are **unverified** here.

### Major 12 — Phase 2 is too underspecified to falsify a method or support compute extrapolation

**Plan:** §5; P:152–155.

**Problem:** “STE QAT plus KL” describes a family, not a reproducible experiment. Missing choices include trainable parameters, latent dtype, scale parameterization, how fixed sparsity is enforced, whether signs/rotations can change, optimizer/schedule, sequence lengths, loss temperature/direction, hard-label loss, clipping, and teacher placement. Ten million, 100 million, and one billion tokens alone do not establish a scaling law; three noisy points can plateau or reverse, and 1.7B does not extrapolate reliably to hybrid 27B. Raw open text and local KL may preserve NLL while failing tool formatting or instruction following.

**Evidence:** These implementation and budget choices are absent from P:152–155. The review's identifiability counterexamples mean static drift cannot resolve them. The plan's own target includes coding/tool use, beyond the raw-text metric.

**Fix:** Freeze a complete training recipe and an initialization/teacher-only baseline. Compare cheaper scale-only, selective-weight, and blockwise reconstruction options before full-weight QAT. Use short profiled runs to measure peak memory and tokens/second, several seeds for selected budgets, fixed evaluation checkpoints, and a real held-out scale-up point before extrapolating. Specify an affordable corpus mixture for the target capabilities, documented deduplication against evaluation, and a ceiling in GPU-hours as well as tokens. Unknown overlap with Prism's original training data remains unresolvable from the artifacts and must be labeled so.

### Major 13 — The resource plan omits the working set, teacher, and realistic 27B activation capture

**Plan:** §§3.0–3.1, 5, 8.

**Problem:** One F32 endpoint matrix is indeed about 5.09 GB, but decoding, both rotation directions, magnitude/order workspaces, codes, and the mapped ancestor can multiply that footprint. “One tensor at a time” is insufficient on a 30 GiB host. A 27B BF16 forward cannot be resident on a 12 GB GPU, or wholly resident in 30 GiB system RAM, merely because T5 is GPU-enabled. The 25–30 GB full-Adam estimate for 1.7B is reasonable for one common implementation before activations, but omits the teacher, fake-quant buffers and full-vocabulary KL. The absolute claim that local training fits “only about 0.6B” is unsupported without profiling specific offload/dtype implementations.

**Evidence:** Header descriptors give 1,271,398,400 elements per endpoint. `free -h` during review showed 30 GiB total but about **18 GiB available**, with swap already used. HF's pinned Qwen index reports `total_size = 55,562,855,904` bytes for the full artifact. The proposed CPU minutes, training throughput, and GPU fit were not measured and are **unverified**. A common 16-byte/parameter Adam ledger is 27.2 GB for 1.7B, before teacher/activations. At 1B tokens, the rough dense student-training term `6*N*T` alone is approximately `1.02e19` FLOPs, before teacher and QAT overhead.

**Fix:** Stream bounded row tiles aligned to both 1024 rotation blocks and 128 groups; compute one convention at a time and avoid whole-tensor sorting/index arrays. Profile peak RSS and throughput before all-block expansion. For T5, implement disk/CPU/GPU layer streaming or rent an appropriately sized machine; a quantized substitute teacher must be explicitly treated as a changed experiment. Budget checkpoint/optimizer copies, activation caches, teacher logits or on-the-fly teacher inference, KL chunking, conversion scratch, evaluation generations, and recovery checkpoints. Give measured wall-time/cost projections and available-memory limits. Distinguish decimal GB from GiB: current `df -h` showed 323 GiB free, not a directly comparable 323 GB.

### Major 14 — Transfer can pass every current gate without achieving the stated goal

**Plan:** §§6, 10; P:159–162, 200–202.

**Problem:** An allowed architecture name does not ensure every width, tensor role, or tied-weight case works. The actual 1.7B target ties embeddings and the output head, whereas the reviewed 27B artifact does not. Two implementations can match each other's logits while sharing a wrong reference transformation. Passing serialization and measuring quality on a single small full-attention model does not demonstrate Bonsai-2-level quality, comparable practical memory/latency, or transfer to another family.

**Evidence:** `S/conversion/base.py:741–756` requires **schema 3 and `tied_output=true`** for tied latent embeddings, and forbids a separate output tensor. `S/src/llama-model.cpp:2006–2009` requires every folded input width to be divisible by the chosen block size. Architecture support alone is therefore insufficient. Gen-2's language-only structural analysis explicitly warns that vision scores involve a separate tower (`analysis/bonsai2/README.md:49`). The inherited small-model BFCL protocol excludes multi-turn tasks.

**Fix:** Preflight shapes, ties, routers/shared experts, tensor roles, tokenizer IDs, context configuration, and packed-size budget before target selection. Add an independently checked FP-function-preservation gate, followed by full-vs-folded-vs-packed layer/logit comparisons across decode/prefill/chunk boundaries. Freeze transfer targets and permitted tuning before results. Require behavioral non-inferiority and actual bit/byte budgets, plus measured memory and latency, on at least an unseen architecture/scale and a hybrid bridge if claiming gen-2 transfer. Add real multi-turn/tool-use and long-context generation tests; define whether vision is excluded or separately controlled. Packing failures should be diagnosed as codec/manifest/dtype/reference errors before assuming runtime graph work.

### Minor 15 — Several factual/provenance claims need narrower wording; downloads can be much cheaper

**Plan:** §§1.3, 3.1, 8.

**Problem:** The advertised ~54 GB is approximately the language-model baseline size, not the full current HF artifact. Block 0 and token embeddings are not in one shard. An 8.6 GB MLX pack is unnecessary to cross-check a small JSON contract. Two export metadata copies agreeing confirms consistency, not independent evidence of training history. “No unpacked Bonsai 2 checkpoint is published” and the exact gen-1 no-rotation statement are broader than the checked sources establish.

**Evidence:** The pinned Qwen index puts language block 0 in `model-00001-of-00018.safetensors` and `model.language_model.embed_tokens.weight` in `model-00003-of-00018.safetensors`; the full index size is 55.56 GB. I fetched **297,903 bytes** of MLX `hadamard.json`; all 28,672 sign values, widths, transform, axis, version, and GDN flag matched the GGUF. The folded count is 401; source names differ by export convention. No MLX weights were needed. The gen-1 papers support ordinary architecture and no disclosed online rotation, but do not establish the full absence of offline reparameterization. A comprehensive current search for unpublished/other Bonsai-2 unpacked artifacts was not performed: **unverified**.

**Fix:** Download config, index, tokenizer metadata, and rotation JSON first, then only the shards required for a prespecified pilot. Start with T1 and representative complete rows/rotation blocks before scanning all ~210 million ternary groups. Consider range requests for headers or selected tensor data during the future experiment if the host supports them, with explicit range checks and provenance; otherwise use whole selected shards. Mark negative publication claims with the searched repositories/date and leave them open to revision.

## Explicit answers to section 11

1. **Are H1–H4 complete/discriminating; can a method fool T1–T4?** No. Frozen-norm QAT, scale-only optimization, blockwise calibration that adjusts norms, data-free iterative/equalized quantization, final reprojection, and quantization of an intermediate fine-tune are counterexamples. Even exactly reproducing final weights establishes an effective mapping for those inputs, not a unique history. Use candidate rejection and compatibility classes.

2. **Are alignment controls sufficient, especially V grouping and embeddings?** No. Correct the HF/GGUF name mismatch first. Determine conventions from source and independent functional identities, not correlation. Projection and row-major embedding folding both use `W Rᵀ`/`E Rᵀ`; runtime activation rotation and embedding restoration have opposite operation order. For the actual GDN geometry, test 16 K heads × 3 V heads per K × 128 channels, including the permutation before signs/H. Prove mapping on labeled synthetic fixtures and an unquantized FP forward, then interpret real-weight differences without censoring low-similarity tensors.

3. **What thresholds for approximately 100% signs and exact top-k?** For an exact claimed deterministic mapping: 100% agreement on positions/groups outside an independently justified numerical ambiguity set; top-k equality modulo tied/uncertain boundary positions; exact exported FP16 scale bits under the specified dtype chain. These are *mapping* criteria, not historical-training tests. Publish all mismatch counts and excluded fractions. Use discovery/hold-out splits, empirically calibrated error floors and block-level summaries for approximate candidates. There is no defensible universal 99.x% provenance threshold. Scale rounding alone does not justify sign or support mismatches.

4. **Is the gen-1 1.7B pilot sound?** Yes as a cheap gen-1 format/statistics/harness pilot; no as validation of gen-2 rotation, GDN, high-precision policy, or training recipe. It should be accompanied early by synthetic and selected-shard gen-2 tests. Both generations retain some high-precision tensors. The closer Qwen variant remains only the closer tested candidate ancestor.

5. **Is T5 a discriminator between PTQ and QAT?** No. Both can reduce held-out local output error, and both can fail to do so on the base's current activation distribution. Correct its basis and input site, evaluate covariance-aware normalized errors with uncertainty, and describe the outcome as functional reconstruction evidence. Add whole-block/end-to-end checks for compensating changes.

6. **Is the Phase 1 success metric well-defined across families?** No. Specify the origin and denominator, orientation, aggregation, and near-zero-denominator rule separately for each family. Prefer absolute paired non-inferiority to Bonsai at the same packed budget; use the recovery and residual formulas in finding 10 only as secondary descriptors. Do not average arbitrary normalized NLL, margins, and task points into one success score. Select the best arm before the final test.

7. **What could cause failure even if every phase passes?** Shared reference bugs; a codec that silently promotes embeddings or quantizes protected gates; tied-output mishandling; untested tensor widths/roles; a small full-attention result that fails on recurrent models; an unreleased ancestor; a favorable book/retrieval distribution that misses tool-use/long-horizon failures; evaluation selection or contamination; unaccounted teacher/training cost; and no enforced quality/bit/latency threshold on the transfer model. Most fundamentally, a competitive alternative algorithm can pass everything without being Prism's original method. Define successful functional replication and historical identification as distinct deliverables.

## What the plan gets right

- It correctly makes basis/layout alignment a prerequisite for interpreting raw differences, and recognizes the special GDN `out_proj` case even though the proposed implementation is wrong.
- The checked GGUF header agrees with the core structural claims: 851 tensors; 402 PQ2_0 tensors with 26,869,760,000 elements; 96 BF16 and 353 F32 tensors totaling 26,238,464 higher-precision parameters; 401 folded projections plus an inverse-lookup embedding. The read stopped after 11,120,982 header bytes. Rotation metadata matched the separate MLX JSON exactly after accounting for tensor-name namespaces.
- The 97.2%/42-zero statement accurately reports the prior **sample**, and the plan correctly says a fixed zero count alone cannot distinguish PTQ from training. I verified the existing report (`analysis/bonsai2/README.md:30–34`), not the payload again.
- The gen-2 83.9/85.4/75.2 comparison and gen-1 27B 80.49/85.07 comparison match the cited paper tables; their different benchmark suites should remain separate. No disclosed weight-production recipe was found in the examined paper material; KV-cache calibration is a separate method.
- Hashing, per-gate freezes, fresh registries, position-matched target IDs, shortcut/value-swap checks, units, and independent book/registry intervals preserve the strongest lessons of the earlier investigation (`LEARNING_REFERENCE.md:55–77`).
- Streaming static comparisons and putting cheap reconstruction before costly training are sensible. With the fixes above, Phase 0 can substantially narrow the effective recipe while honestly leaving its historical origin unresolved.

## Small metadata sources checked

These links pin the artifacts used for metadata checks; no corresponding weights were downloaded:

- [Qwen3.8-27B index, revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/model.safetensors.index.json) and [config](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json).
- [Bonsai-2 MLX rotation contract, revision fcba37d2117a7077eac6b613b2668d14d9779edd](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit/blob/fcba37d2117a7077eac6b613b2668d14d9779edd/hadamard.json) and [model card declaring the Qwen ancestor](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit/blob/fcba37d2117a7077eac6b613b2668d14d9779edd/README.md).
- [Qwen3-1.7B config, revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e](https://huggingface.co/Qwen/Qwen3-1.7B/blob/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/config.json), [Qwen3-1.7B-Base config, revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1](https://huggingface.co/Qwen/Qwen3-1.7B-Base/blob/ea980cb0a6c2ae4b936e82123acc929f1cec04c1/config.json), and [Bonsai-1.7B config, revision 3aca840085293d026ce6f6b80fafdae937fd2eeb](https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-unpacked/blob/3aca840085293d026ce6f6b80fafdae937fd2eeb/config.json).

The HF API declared these repositories public/not gated and Apache-2.0. This verifies their metadata declarations, not any legal conclusion about reproducing the method. Patent status and commercial rights were not investigated and remain **unverified**.

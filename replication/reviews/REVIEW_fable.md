# Review of `REPLICATION_PLAN.md` (draft 2026-09-30) — reviewer: Fable

**Scope.** Independent, adversarial review of `analysis/bonsai2/replication/REPLICATION_PLAN.md`. I did not read any other review. Every claim below tagged **[checked]** was verified against a local file, the pinned Prism source at `/tmp/llama.cpp-842b1880415d6f508f03b789e5ce70194def7bfd` (release `prism-b10735-842b188`, matching `scripts/download_binaries.sh:15`), the local `models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf` header, or small Hugging Face API/range requests (headers, configs, and a few kilobytes of specific tensors; no shard was downloaded). Claims I could not check are marked **[unverified]**. Numbers in the appendix come from the commands I ran on 2026-09-30; they are single-row / few-group probes, not full-tensor statistics, and should be read as such.

---

## Overall verdict

The plan is careful, well-instrumented and honest about what a static comparison cannot reveal, but it is aimed at a question that a 100-kilobyte probe already answers, and it stakes its most expensive phase on a pilot that does not share the recipe it is meant to illuminate. Streaming the small tensors of `Qwen/Qwen3.8-27B` and comparing them with the local GGUF (the plan's own T1) shows that **every high-precision tensor I sampled has been changed**: layer norms in blocks 0, 3, 31 and 63, q/k norms, the GDN norm, `A_log`, `dt_bias`, `conv1d`, `in_proj_a` and `in_proj_b` all differ from Qwen3.8-27B by small, highly correlated amounts (corr 0.88–0.9999, 85–99 % of elements changed, see Appendix A). Under the plan's own decision table (§3.4) that is the H3 row ("end-to-end training touched the network") — a result the plan says "doesn't yield a recipe". The gen-1 1.7B pilot is a poor proxy: its ternary codes have *variable* zero counts (23–60 per group, not the fixed 42 of Bonsai 2), tied embeddings, and only 77–91 % sign agreement with its base, so it will land in H3 for a different reason and on a different recipe. Meanwhile the plan misses two published artifacts that change the economics: the 53.8 GB `Ternary-Bonsai-2-27B-F16.gguf` (the folded, pre-packing checkpoint; I verified it stores exactly the ternary values the PQ2_0 file decodes to) and `prism-ml/Ternary-Bonsai-27B-unpacked` (gen-1 27B on the public `Qwen/Qwen3.6-27B`, same hybrid architecture, no rotation, everything ternary), which shows a strikingly different fingerprint from the 1.7B (98–99 % sign agreement, zero-set Jaccard ≈ 0.55 versus smallest-magnitude) and is the most informative public artifact for the "method family" question. The plan also has three factual errors in its tensor-mapping description, a circular "independent contract cross-check", an evaluation-path gap (rotated arms cannot run on the EvalScope/vLLM harness it wants to reproduce), and a positive control that will silently fail with the pinned `llama-quantize` defaults. My recommendation: **do not start the 54 GB download or the rotation pipeline as the first step.** Run a streamed T1 over all 449 non-ternary tensors (≈57 MB of range requests) and a streamed T2/T3 on two or three full matrices as a Phase-0-pre gate; re-scope Phase 0b from "identify the closed-form rule" to "quantify the drift fingerprint and test data-aware zero-selection criteria"; replace the 1.7B pilot with gen-1 27B for the family question (keep 1.7B as a script smoke test); and put the design effort into Phase 2, where a drift-calibration ladder can turn the "H3 gives no recipe" outcome into an estimate of how much training Prism did.

---

## Findings, ranked by severity

### Critical

**C1. Phase 0's headline question is already answered by T1, and the plan's decision rule then discards most of Phase 0's value.** (§2, §3.3 T1, §3.4)

*Problem.* The plan treats T1 ("were the high-precision tensors changed?") as one of six tests to run after a 54 GB download, a full PQ2_0 decode and a rotation pipeline. But T1 needs none of that: the 353 F32 + 96 BF16 tensors in the GGUF are tiny, Qwen's counterparts can be fetched by HTTP range request from the safetensors shards, and the converter's transforms are known (§1.2). I ran it on a sample (Appendix A). Result: **changed everywhere**, with the signature of light end-to-end training (small correlated deltas in every parameter class, including ones no PTQ method touches: `conv1d` mean |Δ| = 16 % of mean |w|, `in_proj_b` 33 %, norms 0.3–1 %). The plan's §3.4 then routes to "H3: Phase 2 required; use Phase 0's drift map to size the training", and §3.4's last line concedes H3 yields no recipe. T3 and T4 (closed-form zero set / scale) test functions of the *public* base weights; under any training hypothesis the relevant latent weights are unknown, so those tests are uninterpretable exactly in the regime we are now in.

*Evidence.* Appendix A tables; `conversion/base.py:1068-1081` (BF16 → F32 cast happens *before* `modify_tensors`, so bit-exact `+1` / `−exp` comparison is valid); GGUF norm values minus 1 are 100 % BF16-representable and `log(−ssm_a)` is 100 % BF16-representable (my check), so the source checkpoint stored BF16 values and the differences are real, not rounding. Qwen3.8-27B's weight files have not changed since upload (2026-08-13; the only later commits are README/LICENSE) and Bonsai 2 was uploaded 2026-09-17 **[checked via commits API]**.

*Fix.* (1) Add a **Phase 0-pre gate**: streamed T1 over all 449 non-ternary tensors (≈57 MB), reported per tensor kind and block as `exact / max|Δ| / mean|Δ| / corr / frac_changed`, plus a reordered-vs-unreordered control on `conv1d` (which discriminates 0.995 vs 0.72 in my probe). (2) Re-state Phase 0b's goal as *quantifying* the drift of the ternary codes from a projection of public Qwen (T2 sign-flip rate by magnitude decile; T3 Jaccard; per-block/kind map) and testing *data-aware* zero-selection criteria (see M6/M10), not finding a closed form. (3) Make T3/T4 conditional on T1/T2 showing no training. (4) Move the bulk download behind the streamed gate: two or three full matrices (e.g. `blk.0.ffn_down` = 178 MB in BF16) give the drift fingerprint before committing 54 GB.

**C2. The gen-1 1.7B pilot is not a proxy for the gen-2 recipe, and a better public proxy exists.** (§3.0, §11 Q4)

*Problem.* Bonsai 2's most distinctive structural fact is the fixed 42-zero budget (97.2 % of groups). `Ternary-Bonsai-1.7B-unpacked` does **not** have it: zero counts per group in the rows I sampled span 23–60 (`q_proj` 38–58, `down_proj` 43–59, `embed_tokens` 23–44). It has tied embeddings (`tie_word_embeddings: true`, no `lm_head` in the safetensors header; 151,669 embedding rows vs Qwen3-1.7B's 151,936), so the embedding/LM-head comparison is a different object. Its drift is heavy: nonzero-sign agreement with Qwen3-1.7B is 77.5 % (`q_proj` row 0) and 91.5 % (`down_proj` row 0); correlation of the dequantized ternary with the base is 0.52 / 0.73; zero-set Jaccard against the k-smallest-|w| rule is 0.27 / 0.42; the stored scale is 1.3× (down) to 2–3× (q) the least-squares candidate. A data-free projection would give 100 % / ≈0.9 / 1.0 / ≈1.0. So the pilot will land in H3 immediately and teach nothing about the 42-budget or the rotation. It will also anchor thresholds on a recipe that Prism visibly changed twice (gen-1 1.7B → gen-1 27B → gen-2).

By contrast `prism-ml/Ternary-Bonsai-27B-unpacked` (12 BF16 shards, 54.7 GB) is gen-1 27B on the public `Qwen/Qwen3.6-27B` (55.6 GB, not gated): same Qwen3.5-family hybrid architecture, no rotation (no Hadamard mention anywhere in the gen-1 27B whitepaper; the Bonsai 2 paper calls it the "pre-rotation build"), and everything ternary including `in_proj_a` (verified ternary, variable zeros). Its fingerprint on the rows I probed is *different and more informative*: nonzero-sign agreement 97.9 % (`down_proj`) / 99.0 % (`out_proj`), correlation 0.85, zero-set Jaccard 0.55–0.60, norms changed (corr 0.984–0.999, 98–100 % elements). That is "signs preserved, zero selection not magnitude-based, norms lightly trained" — closer to a data-aware or lightly-trained assignment than to either H1 or heavy H3.

*Fix.* Keep the 1.7B as a **script smoke test only** (it is cheap and unrotated). For the "method family" question use gen-1 27B vs Qwen3.6-27B; it can be done by streaming the same tensors as the Bonsai 2 comparison (no 110 GB download needed for the gate-level questions). Record that the recipe differs across the three releases and treat gen-1 fingerprints as lower/upper bounds, not as the gen-2 rule.

**C3. Phase 1/3 have no evaluation path for rotated arms on the harness the plan wants to reproduce.** (§4 metrics, §9 evaluation cost, G1)

*Problem.* G1 requires reproducing Prism's EvalScope + vLLM numbers, then scoring each arm on the same harness. Arms 3–5 are Hadamard-folded; stock HF/vLLM cannot run them (the MLX pack's own `PACK-RUNTIME.md`: "Ordinary MLX loaders do not apply the required transforms"; `KNOWN_ISSUES.md`: the F16 GGUF "loads in stock llama.cpp but produces garbled output"). The plan does not say how folded arms reach EvalScope. Separately, Prism's protocol (81,920-token output budgets, AIME avg@8, GPQA ×5, thinking mode, `xhigh`) is far beyond a 12 GB GPU for a 27B model and heavy even for 1.7B.

*Fix.* Two options, both cheap: (a) **unfold for evaluation** — store `W_eval = W_t · R` (dense F16) so the model is function-identical to the folded one (the runtime computes `W_t (R x) = (W_t R) x`) and runs on stock HF/vLLM; report the F16-rounding floor of the unfold; (b) serve arms through the pinned `llama-server` (OpenAI-compatible) and point EvalScope at it. State the benchmark subset and budgets you can actually afford locally, and treat Prism-number reproduction as a harness sanity check on the two reference models only.

### Major

**M1. §1.2 tensor-mapping description has three errors for Qwen3.8.**
- "`in_proj_qkvz` split and fused into qkv and z": that is the **Qwen3-Next** path (`conversion/qwen.py:397-425`). Qwen3.8-27B's HF checkpoint already stores `linear_attn.in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj` separately **[checked: `model.safetensors.index.json`]**, and `_LinearAttentionVReorderBase.modify_tensors` (`qwen.py:572-631`) only reorders V rows/columns.
- Attention `q_proj` carries the output gate interleaved per head (`attn_output_gate: true`; GGUF `attn_q` is `[5120, 12288]` = 24 heads × 2 × 256). The converter does not split it; `src/models/qwen35.cpp:333-361` views it as `[n_embd_head, 2, n_head]`. The comparison must keep that row layout and must not compare only the first half.
- The GGUF has 851 tensors = language model only: the 15 `mtp.*` tensors (`mtp_num_hidden_layers: 1`) and the vision tower are absent. Exclude them from the map, and note the GGUF carries no `nextn` metadata.
Also worth recording: `attn_k`/`attn_v` are `[5120, 1024]` (4 KV heads × 256); `ssm_out` and `attn_output` share the width-6144 sign vector.

**M2. "No unpacked Bonsai 2 checkpoint is published" is wrong, and the missing artifact is useful.** (§1.3)
`prism-ml/Ternary-Bonsai-2-27B-gguf` contains `Ternary-Bonsai-2-27B-F16.gguf` (53,808,408,928 bytes) with the same 851 tensors, the same `prism.hadamard.*` contract, `general.basename = folded`, 498 F16 + 353 F32 tensors **[checked: header]**. I range-read 12 groups (`blk.0.attn_gate`, `token_embd`, `output`): all exactly ternary `{−s, 0, +s}` with 42 zeros, and bit-identical to the local PQ2_0 decode. `blk.0.ssm_alpha` row 0 in the F16 file equals the local BF16 to within 3e-8 (F16 re-export of BF16), so for T1 use the PQ2_0 file's BF16 tensors, not the F16 file. Uses: (i) a **T0 gate** — decoder correctness and packing losslessness by per-tensor range requests, no local packing needed; (ii) T4 reads the upstream FP16 scale directly; (iii) a stronger positive control: our own PQ2_0 packer applied to F16 tensors must reproduce the published PQ2_0 bytes. The `Ternary-Bonsai-1.7B-gguf` repo likewise ships an F16 "re-quantization source".

**M3. The "contract cross-check" against the MLX pack is circular.** (§1.3, §3.2)
`runtime/runtime.py` in the MLX pack builds the model by reading `prism.hadamard.*` fields from a GGUF (`load(gguf_path, ...)`, lines 73-110) and transcoding PQ2_0/PTQ1_0 blocks (`runtime/codec.py`); `hadamard.json` is that metadata re-emitted with HF-style names. It is derived from the GGUF, not an independent copy. Drop the independence claim. The cross-check that actually matters — GGUF metadata ↔ runtime semantics — I did: llama.cpp `build_lora_mm` applies `signs` then the blockwise WHT (`src/llama-graph.cpp:1551-1573`), the rotation matrix is the normalized Sylvester Hadamard `(−1)^{popcount(row&col)}/√n` (`src/llama-model.cpp:2040-2050`), the embedding inverse applies WHT then signs (`llama-graph.cpp:2401-2410`), and MLX `fwht` does the same (`runtime.py:16-28`). So `R = H·S/√n` exactly as the whitepaper states.

**M4. Rotation and embedding directions are derivable, not open; specify them and use "both directions" only as a control.** (§3.1 step 3, §11 Q2)
Runtime: `y = W_t (R x)`; for `y ≈ W x`, `W_t = W R⁻¹ = W Rᵀ = (W S) H/√n`, i.e. **flip column signs first, then WHT each 1024-block of columns**. Embedding: runtime returns `h = S (H z)/√n` for stored row `z`, so `z = R e` and `E_t = E Rᵀ` — the *same* right-multiplication as the weights, on the embedding dimension. Ordering is Sylvester/natural (`scipy.linalg.hadamard(1024)` matches; the "sequency" ordering would be wrong). Keep the wrong-direction run as negative control (b) but state the expected answer. Add negative control (e): unreordered V heads for `attn_qkv`/`conv1d`/`attn_gate` — my `conv1d` probe gives corr 0.995 reordered vs 0.72 unreordered, so this control is sharp.

**M5. Parameter symmetries can produce misleading T2/T3/T4 signatures; the plan does not list them.** (§2, §11 Q1-Q2)
- *Norm-γ column scaling.* `W (γ ⊙ n(x))` is invariant under `γ → γ′, W → W·diag(γ/γ′)`. Since Bonsai's γ differ from Qwen's by 0.3–1 % on average (up to 7–13 %), the right base for T2–T4 on every matrix that consumes a normed input (`attn_qkv`, `attn_gate`, `ssm_alpha/beta`, `attn_q/k/v`, `ffn_gate/up`) is `W_qwen · diag(γ_qwen/γ_bonsai)` before rotation; for `ssm_out` it is the GDN norm γ. Run both plain and γ-corrected candidates.
- *Sign symmetries.* Joint per-channel sign flips of q/k rows (through `q_norm`/`k_norm` channels) and of FFN hidden units (`ffn_up` row *i* with `ffn_down` column *i*) leave the function unchanged. Sign agreement on `q_proj` alone is therefore weak evidence (the 1.7B gives 77 % on `q_proj` but 91 % on `down_proj`). Use residual-writing matrices (`ffn_down`, `ssm_out`, `attn_output`, `output`) as the primary T2 tensors.
- *Residual-stream rotation (SpinQuant R1).* Would have folded γ to 1; excluded by T1 closeness — state this as a check, it is a real alternative the whitepaper's citation [28] invites.

**M6. Hypothesis set is incomplete, and the decision table has an ambiguous cell.** (§2, §3.4)
Add: **H2b** data-aware zero selection with sign-preserving assignment (Wanda/SparseGPT-style `|w|·‖x_j‖` criteria) — the gen-1 27B fingerprint (99 % signs, Jaccard 0.55) fits this; **H3b** train-then-project (a fine-tuned Qwen followed by a closed-form projection) — statically indistinguishable from H3; **H5** QAT with latent weights + STE — the final codes are a projection of *latent* weights, so T3/T4 against public Qwen are uninformative. The observed combination "T1 changed, T2 ≈ 99 %" (gen-1 27B) falls between rows 2 and 3 of §3.4; add a "light training / data-aware" row and make H3 vs H4 a matter of measured correlation, not a binary.

**M7. The 42-zero fact is a *zero budget*, not balanced thirds, and it is not MSE-optimal — say what T3 will test.** (§1.1, §3.3 T3)
In 200 groups each from four tensors, zeros are fixed at 42 while (neg, pos) vary: (43,43), (41,45), (45,41), (44,42) **[checked]**. So the constraint is "exactly 86 nonzeros per group", consistent with "zero the 42 smallest |w′|" or with an importance-ranked selection of 86. Note 32.8 % zeros is well below the MSE-optimal zero fraction for Gaussian-like rotated weights (≈46 % at threshold 0.61σ), so the budget was chosen for a reason other than per-group MSE (entropy per weight, hardware, or data). The 2.8 % non-42 groups are spread over every matrix kind (`ffn_up` 42 tensors, `ffn_down` 41, `ffn_gate` 39, `ssm_out` 33, …), not concentrated — record their counts (41/43?) as a clue about ties or training. T3 should test: (a) 42 smallest |w′|; (b) 42 smallest importance under Bonsai-captured activations (M10); and report the per-decile flip profile.

**M8. Phase 1 target and success metric are weak.** (§4, §11 Q6)
Ternary-Bonsai-1.7B retains 86 % of Qwen3-1.7B on the 6-benchmark suite (57.5 / 66.57, ternary 8B whitepaper Table 7) while Bonsai 2 27B retains 98.2 %; matching the 1.7B is a low bar and says nothing about the gen-2 recipe. Gap-closure fractions in nat/token, nat/answer and benchmark points are not commensurable: report each family with its own interval and do not average. Qwen3-1.7B's tied embeddings require the converter's schema-3 `tied_output` path for folding (`base.py:741-758`; runtime version 2, `llama-model.cpp:1198-1207`) — plan for it. Add an "arm 0": the pinned quantizer's own absmax rule (`llama-quantize PQ2_0` on rotated Qwen), and an arm with the fixed-86-nonzero budget, since that is the observed gen-2 constraint.

**M9. The positive control will fail with default `llama-quantize` behaviour.** (§3.2)
- `quantize_row_pq2_0_ref` (`ggml/src/ggml-quants.c:113-143`) is itself an absmax-round ternarizer: lossless only if the input is exactly `{−s,0,+s}` with `s` FP16-representable (BF16 scales are). `quantize_pq2_0` ignores the importance matrix.
- Default type selection under `PQ2_0`/`PTQ1_0` sends `token_embd` → Q4_K and `output` → Q6_K, and would quantize 2-D `ssm_alpha/ssm_beta` (`src/llama-quant.cpp:461-505`, `:700-718`). The published file keeps those BF16 and the embedding/output PQ2_0, so use `--pure` plus `--tensor-type` overrides (or `--token-embedding-type`/`--output-tensor-type`).
- The manifest must say `"name": "normalized-signed-sylvester-walsh-hadamard"`, `"status": "requires-matching-runtime"`, `"axis": -1`, roles `fold-before-matmul` / `inverse-after-lookup`, schema 1–3 (`base.py:644-720`), whereas the GGUF stores `normalized-sylvester-walsh-hadamard`; copying the GGUF string into the manifest fails validation.
- The binary exists locally (`bin/cuda/llama-quantize`, release `prism-b10735-842b188`) **[checked]**.

**M10. T5 is not feasible as written and does not discriminate H2 from H3.** (§3.3 T5, §8, §11 Q5)
Qwen3.8-27B in BF16 (54 GB) fits neither the 12 GB GPU nor 30 GB RAM; "on GPU for 27B" needs layer-streamed CPU forward passes or rented hardware. Cheaper proxy: capture per-layer inputs from **Bonsai 2 itself** via the pinned llama.cpp on the 12 GB GPU (it was built to match the base's activations). And since training with data also produces activation-structured deviations, T5 cannot separate H2 from H3; its real use is testing whether a *specific* importance criterion reproduces the zero set (H2b), which is worth doing given the gen-1 27B Jaccard of 0.55.

**M11. Phase 2 feasibility and extrapolation risk are understated.** (§5, G2)
Locally only ≈0.6B is trainable, and no Prism reference exists at 0.6B, so the ladder measures gap closure vs FP16 only. Published conversions to ~1.58 bit use tens of billions of tokens; a 10M–1B ladder can estimate a slope, not the endpoint. Define G2 as "slope of gap closure vs log(tokens) with intervals", and add a **drift-calibration** output: measure sign-flip rate and zero-set Jaccard vs tokens in your own QAT, then map Bonsai's measured fingerprint (Appendix A/B) onto that curve — this is the only way the H3 outcome yields a quantitative estimate of Prism's training dose.

### Minor

- **m1.** §1.1 "gen-1: no high-precision tensors" — gen-1 kept "a negligible tail of normalization and scale parameters in higher precision" (Bonsai 27B whitepaper §3, text line ≈287); what gen-2 added is the recurrent-state path. The gen-1 27B unpacked stores `in_proj_a/b`, `conv1d`, `A_log`, `dt_bias` as BF16 (`in_proj_a` row 0 is ternary-valued; the others **[unverified]**).
- **m2.** §1.3 sizes: Qwen3.8-27B is 55.6 GB (includes 0.47B vision tower and one MTP layer); block 0 is in shard 1 (3.97 GB), `embed_tokens` in shard 3 (2.54 GB), `lm_head` in shard 18 (3.39 GB), final norm in shard 16. Safetensors supports per-tensor HTTP ranges, which makes the "download order" moot for the gate.
- **m3.** §8: 323 GB free now (`df`), not 345. The sandbox blocks `huggingface.co` and the `*.hf.co` CDN unless allow-listed; the GPU is hidden in the sandbox (`nvidia-smi` fails), consistent with "host-device CUDA outside the sandbox".
- **m4.** T1 tolerance: bit-exact after the converter's F32 transforms is the right rule (cast order verified); allow ≤ 2 ulp for `exp` libm differences. Report `corr` and `frac_changed`, not just pass/fail, because the answer is "changed".
- **m5.** T4: by construction the PQ2_0 scale is the group absmax of the folded F16 checkpoint, so "stored scale" = upstream `s` in FP16 exactly; the least-squares candidate given codes is `mean|w′|` over the 86 nonzeros. In the 1.7B the ratio `s / mean|q|_nz` was 1.3 (`down_proj`) and 2–3 (`q_proj`) — not a closed form of the public base.
- **m6.** Whitepaper citations check out (Tables 7–10: 83.9/85.4, IQ2_XXS 75.2 at 7.3 GB; 26,238,464 = 0.0976 %; gen-1 80.49/85.07 on 15 benchmarks; 1.7B 57.5 on IFEval/GSM8K/HumanEval+/BFCL/MuSR/MMLU-Redux, with Qwen3-1.7B at 66.57 on that suite). The same paper's Table 10 uses a 10-benchmark suite (49.58 vs 58.24) — cite one suite consistently.
- **m7.** The MLX repo is tagged `base_model:finetune:Qwen/Qwen3.8-27B` (the GGUF repo: `quantized`) — weak, but consistent with T1.
- **m8.** §6 targets: sign widths must be multiples of the block size (`base.py:670-673`, `llama-model.cpp:1251`); check hidden/FFN/attention-output widths of each candidate (Qwen3-0.6B 1024/3072 ✓, Qwen3-1.7B 2048/6144 ✓, Llama-3.2-1B 2048/8192 ✓). `gdn_v_grouped` handling exists only for the QWEN35-family `ssm_out`.
- **m9.** No `Qwen/Qwen3.8-27B-Base` or `Qwen3.6-27B-Base` is public (API returns an auth error, i.e. private or nonexistent) **[checked]**; "post-trained" is the only public base, and the Qwen3.8 model card says so explicitly. The H4 variant "Qwen-internal intermediate checkpoint" therefore cannot be excluded by any static test; say so.
- **m10.** Phase 0a: Qwen3-1.7B (post-trained) is closer to Ternary-Bonsai-1.7B than `-Base` on every norm I checked (e.g. layer-0 `post_attention_layernorm` 9.8 % of elements changed vs 72 %; layer-14 `input_layernorm` 0.2 % vs 20 %; layer-27 exact for both) — the "which variant is the base" question is answered before the pilot runs.
- **m11.** The MLX pack's runtime code is Apache-licensed (`runtime/LICENSE`) and is a compact reference for the transform semantics; cite it in §1.2.

---

## Answers to §11

**Q1 — Are H1–H4 complete and discriminating? Could a method fool T1–T4?**
Not complete: add H2b (data-aware zero selection, sign-preserving), H3b (train, then project), H5 (QAT with latent weights). Only T1 separates "some parameters were optimized with data" from "pure projection of public weights", and it already reads *changed* for every tensor sampled. T2–T4 can be fooled by: train-then-project (looks like H3 although a closed form was used); γ-scaling and sign symmetries (M5); and QAT latents (T3/T4 against public Qwen are uninformative). A "norm-tweaking"-style PTQ would change norms but not `conv1d`/`in_proj_b`, which are changed here, so the H2-with-norm-refit loophole is closed by the probe.

**Q2 — Are the alignment controls sufficient?**
For layout errors, yes, with two additions: the unreordered-V-head control (sharp: 0.995 vs 0.72 on `conv1d`) and the γ-corrected candidate. Specifics: `ssm_out` stays in grouped V order and the runtime permutes the activation tiled→grouped before the transform (`llama-graph.cpp:1560-1568`, `llama-model.cpp:2124-2133`; MLX pack: "GDN activations are already grouped"), so compare Bonsai `ssm_out` with Qwen `out_proj` unpermuted — the plan has this right. Embedding inverse: `E_t = E Rᵀ`, same direction as weights (M4). Axis: `input-last-dimension` = GGUF `ne[0]` = HF last dimension. Ordering: Sylvester. Direction: derived, not free.

**Q3 — Thresholds.**
T1: bit-exact after F32 transforms (≤ 2 ulp for `exp`); anything else is "changed" — report corr and `frac_changed`. T2 (nonzero-sign agreement, primary on residual-writing matrices): ≥ 99.9 % with residual flips only where |w′|/s < 0.05 → projection-consistent; 95–99.9 % with flips concentrated in the lowest |w′| decile → light training; < 95 % → heavy; 50 % is chance (negative controls). T3: per-group Jaccard = 1 in ≥ 99 % of groups → exact (F32 ties are negligible; scale rounding does not affect ranking). T4: |s − fp16(candidate)| ≤ 1 FP16 ulp in ≥ 99 % of groups → formula. BF16 base weights are exact in F32, so "BF16 base" adds no tolerance; FP16 scale rounding affects only T4.

**Q4 — Is the gen-1 1.7B pilot sound?**
As a script exercise, yes. As a recipe proxy, no (C2): variable zero counts, tied embeddings, heavy drift, and a recipe that changed at gen-1 27B and again at gen-2. Use gen-1 27B vs Qwen3.6-27B for the family question.

**Q5 — Is T5 a valid discriminator?**
No: data-driven training produces the same activation-structured deviations. T5 is worth running only as a test of specific importance criteria (H2b) with Bonsai-captured activations as a proxy for the base's (M10).

**Q6 — Is the Phase 1 success metric well defined across families?**
No. Define per family (benchmark points on the harness you can actually run; nat/token NLL; nat/answer margins), each with an interval over independent units, and never average across families. Report gaps to FP16 and to Bonsai-1.7B separately; the latter is a low bar (86 % retention).

**Q7 — What would make the replication fail even if every phase passes?**
(a) The recipe is training whose data/schedule no static test reveals; Phase 1 (PTQ) can pass its own gates and still not reproduce Bonsai. (b) Rotated arms have no path onto the standard harness (C3) — fixed by unfolding. (c) Prism's evaluation protocol is unaffordable locally; reproduction at Prism's budgets is not possible, so G1 must be redefined. (d) The runtime supports folding only for the listed architectures and `gdn_v_grouped` only for QWEN35-family; any other target needs graph work before Phase 3. (e) Extrapolating QAT cost from 0.6B/≤1B tokens to 27B may be off by orders of magnitude (M11). (f) Bonsai 2 may derive from a Qwen-internal checkpoint (m9). (g) The 42-budget and rotation may interact with the training (e.g. the "pre-rotation build" was re-trained after rotation) so that a projection-only Phase 3 packaging of a QAT model trained without the budget in the loop loses quality. (h) Legal: the method is described as proprietary Caltech IP.

---

## Cheaper or more decisive experiments (in order)

1. **Streamed full T1** (≈57 MB): all 449 non-ternary tensors, per kind/block, with the reorder control. Today, on CPU.
2. **Streamed T0/T4 on the F16 GGUF**: per-tensor range reads, verify ternary-ness, 42-count, bit-equality with local PQ2_0, and read the FP16 scales.
3. **Streamed T2/T3 on 2–3 full matrices** (`blk.0.ffn_down`, `blk.3.attn_output`, `blk.47.ssm_out`, ≈180 MB each): rotate `W_qwen·diag(γ_q/γ_b)` with `(W S)H/√n`, report sign agreement by |w′| decile, Jaccard, and negative controls. This is the drift fingerprint; decide the 54 GB download afterwards.
4. **Gen-1 27B vs Qwen3.6-27B on the same tensors** (no rotation): family fingerprint (my probe: 98–99 % signs, Jaccard 0.55–0.60).
5. **Importance-criterion test** (H2b) using Bonsai-captured per-layer inputs from the pinned llama.cpp on the 12 GB GPU: does `|w′|·‖x_j‖` (or `|w′|²·‖x_j‖²`) reproduce the 86-nonzero set better than |w′| alone?
6. **Unfold trick** (`W_t R` in F16) for evaluating any folded arm on stock HF/vLLM; measure its numerical floor once.
7. **Drift calibration in Phase 2**: sign-flip rate and Jaccard vs tokens on your own QAT ladder; map Bonsai's fingerprint onto it.

---

## What the plan gets right

- The format, rotation contract and structure facts in §1.1–1.2 are correct **[checked]**: ternary g128 with FP16 scale; `R = H·S/√n`, block 1024, explicit signs for widths 5120/6144/17408 (28,672 values); 401 folded weights + `token_embd` inverse; `gdn_v_grouped`; 402 PQ2_0 / 96 BF16 / 353 F32 tensors; 26,238,464 high-precision parameters; `general.name = Hf`, `basename = folded`, `version = v5`.
- PQ2_0 packing is lossless for ternary input and the ternarization happened upstream — now directly verified against the published F16 GGUF.
- The converter's transform list (`−exp(A_log)`, `dt_bias → dt_proj.bias`, `conv1d` squeeze, `+1` on all norms except `linear_attn.norm`, V-head reorders, folded `out_proj` kept grouped) and the architecture/tensor-kind restrictions are accurate (`qwen.py:387-395, 572-631`; `base.py:687-739`).
- `W_t ≈ W Rᵀ` is the right derivation; streaming one tensor at a time is the right memory strategy; the positive/negative control discipline, pre-freezing of thresholds, hashing/pinning, disjoint data, independent units (books/registries/items) and the "measured / inferred / proposed" tagging are exactly what the earlier `analysis/bonsai2` work validated.
- The plan is honest that an H3 outcome yields no recipe and that Phase 0a may not transfer — it just needs to act on those caveats earlier.
- Whitepaper numbers and the "no training procedure disclosed" claim are accurate.

---

## Unverified

- Final `output_norm` T1 (my script errored before that line); per-tensor T1 over all blocks (I sampled blocks 0, 3, 31, 63).
- Whether gen-1 27B's `conv1d`, `A_log`, `dt_bias` are ternary-valued.
- All T2/T3/T4 numbers here are from single rows / ≤32 groups; full-tensor statistics may differ.
- Whether Bonsai 2's differences could come from an unreleased Qwen3.8 revision (no weight commits after 2026-08-13, but internal checkpoints are unknowable).
- The meaning of the whitepaper's "earlier pre-rotation build" of Bonsai 2 (a pre-release, not a published artifact).
- Phase 2 memory/throughput figures (plausible: 1.7B × 16 B/param ≈ 27 GB with Adam) — not measured.

---

## Appendix A — 27B probe: Qwen3.8-27B (BF16, HTTP range) vs local Bonsai 2 PQ2_0 GGUF

Transforms as the converter does them (F32): norms `+1`, `A_log → −exp`, V-head reorder grouped→tiled (nk=16, rep=3) for `A_log`/`dt_bias`/conv V channels. `frac_changed` = fraction of elements not bit-identical.

| Tensor (Qwen name → GGUF) | n | max|Δ| | mean|Δ| | mean|Δ|/mean|w| | corr | frac_changed |
|---|---:|---:|---:|---:|---:|---:|
| L0 `input_layernorm`(+1) → `attn_norm` | 5120 | 0.0713 | 0.0062 | 0.0064 | 0.98185 | 0.988 |
| L0 `post_attention_layernorm`(+1) | 5120 | 0.0371 | 0.0023 | 0.0030 | 0.99716 | 0.852 |
| L0 `linear_attn.norm` → `ssm_norm` | 128 | 0.238 | 0.0088 | 0.010 | 0.87548 | 0.703 |
| L0 `A_log`(−exp, reordered) → `ssm_a` | 48 | 0.00265 | 0.00025 | 0.0051 | 0.99993 | 0.521 |
| L0 `dt_bias`(reordered) → `ssm_dt.bias` | 48 | 0.0039 | 8e-5 | 1e-5 | 1.00000 | 0.021 |
| L0 `conv1d`(V reordered) → `ssm_conv1d` | 40960 | 0.0449 | 0.0026 | 0.158 | 0.99504 | 0.992 |
| L0 `conv1d` **unreordered (control)** | 40960 | 0.6045 | 0.0117 | 0.698 | 0.72253 | 0.995 |
| L0 `in_proj_a` row 0 (BF16 both) → `ssm_alpha` | 5120 | 0.0081 | 0.0017 | 0.129 | 0.99161 | 0.985 |
| L0 `in_proj_b` row 0 (BF16 both) → `ssm_beta` | 5120 | 0.0094 | 0.0020 | 0.328 | 0.94872 | 0.995 |
| L3 `input_layernorm`(+1) | 5120 | 0.0625 | 0.0037 | 0.0030 | 0.99914 | 0.926 |
| L3 `q_norm`(+1) | 256 | 0.0166 | 0.0056 | 0.0046 | 0.99855 | 0.953 |
| L3 `k_norm`(+1) | 256 | 0.0215 | 0.0060 | 0.0049 | 0.99954 | 0.961 |
| L31 `input_layernorm`(+1) | 5120 | 0.1309 | 0.0072 | 0.0073 | 0.99895 | 0.976 |
| L63 `input_layernorm`(+1) | 5120 | 0.0625 | 0.0060 | 0.0054 | 0.99957 | 0.969 |

Representability: GGUF `attn_norm − 1`, `q_norm − 1` are 100 % BF16-representable; `log(−ssm_a)` 100 %; `ssm_dt.bias` 100 % — the source checkpoint stored BF16 values and the converter added `+1`/`−exp` in F32, exactly as `base.py:1071-1081` implies.

F16 GGUF (`Ternary-Bonsai-2-27B-F16.gguf`): 4 groups each of `blk.0.attn_gate`, `token_embd`, `output` — all exactly `{−s,0,+s}`, 42 zeros, equal to the local PQ2_0 decode (max diff 0). `blk.0.ssm_alpha` row 0: max diff vs local BF16 = 2.98e-8.

Bonsai 2 code balance (200 evenly spaced groups per tensor, `blk.0.ffn_up`, `blk.31.attn_q`): zeros fixed at 42; (neg,pos) most common (43,43), (44,42), (45,41), (41,45).

## Appendix B — gen-1 probes

**Ternary-Bonsai-1.7B-unpacked (F16, tied embeddings, 151,669 rows) vs Qwen3-1.7B / -Base**

| Tensor | vs Qwen3-1.7B corr / frac_changed | vs -Base corr / frac_changed |
|---|---|---|
| L0 `input_layernorm` | 0.99950 / 0.976 | 0.99933 / 0.827 |
| L0 `post_attention_layernorm` | 0.99999 / 0.098 | 0.99990 / 0.723 |
| L0 `q_norm` | 1.00000 / 0.062 | 0.99999 / 0.359 |
| L14 `input_layernorm` | 1.00000 / 0.002 | 0.99999 / 0.200 |
| L27 `input_layernorm` | exact | exact |
| final `norm` | 1.00000 / 0.002 | 0.99997 / 0.081 |

Matrices (row 0 = 16 groups): `q_proj` zeros 38–58 per group, nonzero-sign agreement 77.5 % (Base 77.2 %), corr 0.524, zero-set Jaccard vs k-smallest-|w| 0.273, `s/mean|q|_nz` = 2.06 / 3.02 / 2.02. `down_proj` zeros 43–59, sign agreement 91.5 % (91.3 %), corr 0.734, Jaccard 0.423, ratio 1.29–1.33. `embed_tokens` row 0 zeros 23–44. All groups exactly ternary.

**Ternary-Bonsai-27B-unpacked (BF16, 12 shards, 54.7 GB) vs Qwen/Qwen3.6-27B (public, 55.6 GB)**

Norms: L0 `input_layernorm` corr 0.98448, frac 0.983, max 0.052; L0 `linear_attn.norm` corr 0.99571, frac 1.000; final `norm` corr 0.99902, frac 1.000, max 0.086. Matrices (row 0, 32 groups): `down_proj` sign agreement 97.9 %, corr 0.848, Jaccard 0.553, zeros 32–44; `out_proj` 99.0 %, corr 0.854, Jaccard 0.596, zeros 33–48. `in_proj_a` row 0 is ternary-valued with variable zeros (35–44).

## Appendix C — commands used (abridged)

- Whitepapers: `pdftotext -layout *.pdf`, grep for train/distill/calibrat/hadamard/rotation.
- Pinned source: `conversion/base.py` (Hadamard manifest 620-773; dtype cast 1068-1081), `conversion/qwen.py` (387-427, 446-659), `ggml/src/ggml-quants.c` (105-143, 494-512, 2198-2300), `src/llama-graph.cpp` (1546-1580, 2398-2410), `src/llama-model.cpp` (1196-1355, 1972-2140), `src/llama-impl.h` (57-75), `src/models/qwen35.cpp`, `src/llama-quant.cpp` (385-410, 461-505, 700-718), `tools/quantize/quantize.cpp` (37-38).
- Local GGUF header via `analysis/bonsai2/inspect_gguf.HeaderReader` (metadata + 851 tensor descriptors; ≤ 40 KB of weight bytes read for group probes).
- HF: `api/models/{repo}?blobs=true`, `commits/main`, `config.json`, `model.safetensors.index.json`, safetensors headers, and ≤ 200 KB of tensor bytes per repo via `curl -r`; F16 GGUF header (11.1 MB) plus 6 KB of tensor bytes.

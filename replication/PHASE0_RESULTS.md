# Phase 0 execution record — 2026-09-30

**Status:** streamed fingerprint, native FP pilot, and held-out tests of the frozen simple candidate rules complete. The small-model runtime contract has passed; the exact 27B FP comparison remains open. These are comparisons with the pinned *public* Qwen3.8-27B revision, not a reconstruction of Prism's unpublished training process. This Phase 0 record has no quality result; the subsequent [0.8B quality pilot](QUALITY_PILOT.md) reports one.

## Frozen inputs and bounded reads

- Local model: `models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf`, SHA-256 `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1`.
- Public candidate: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0), revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The local copy contains its 1199-name safetensors index and config, not its 55.56 GB checkpoint.
- Folded reference: [Prism Bonsai 2 F16 GGUF](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/tree/b072e1d3b35a0a630cece372c2127528e0994386), revision `b072e1d3b35a0a630cece372c2127528e0994386`; only bounded header and tensor ranges were read.
- Pinned runtime and codec: `prism-b10735-842b188` source snapshot and local `bin/cuda/libggml-base.so`; the source tree has no independently usable Git metadata. File hashes are in [freeze.json](results/freeze.json).
- Range reader requires HTTP 206, an exact `Content-Range`, and exact requested length before accepting a response. The T1 run read 52.6 MB across 485 range requests; the three matrix runs read 304.1 MB across 122 requests. No whole model shard was downloaded.

## Gate 0: verified and still pending

The [source-name map](results/mapping.csv) covers all 851 GGUF language tensors, one to one with 851 source names. All 851 source dtypes, shapes, and byte counts passed a [remote-header validation](results/layout_validation.json) across 18 shards. The public index has 348 additional names, including vision and MTP tensors absent from this GGUF. The [folded source-name set](results/source_folded_names.json) has 401 projection names, including all 48 grouped GDN out-projections; embedding lookup uses its separate inverse role. The map follows the pinned converter's BF16→F32, `−exp(A_log)`, norm `+1` (except GDN norm), conv squeeze, and V-head permutation rules. Qwen3.8 already has separate `in_proj_qkv` and `in_proj_z` tensors.

Independent [numeric fixtures](results/contract_fixtures.json) show projection and embedding identities at <3×10⁻¹³ absolute error for widths 5120, 6144, and 17408, including 1024/128 boundaries and asymmetric signs. A labeled GDN tiled→grouped fixture passes at <1×10⁻¹³; omitting that permutation yields ≈30.5 absolute error on the same synthetic example. A wrong sign/H operation order produces large errors. This agrees with the inspected pinned runtime graph; **the fixture has not been run through a separately constructed native FP model**.

The [real V-order control](results/layout_control.json) favors the declared permutation: block-0 conv1d correlation with local Bonsai is 0.9950 after reorder versus 0.7225 without it; for `ssm_a`, 0.99993 versus 0.0512. These correlations are diagnostics, not how the operation order was selected.

The pinned C PQ2_0 quantizer/decoder and independent Python decoder agree on adversarial all-zero, signed, near-threshold, and FP16-rounding fixtures. Sixteen bounded groups across `blk.0.attn_gate`, `blk.0.ffn_down`, `token_embd`, and `output` match **exactly** between local PQ2_0 decoding and the publisher's folded F16 GGUF; re-quantizing those F16 values with the pinned C quantizer reproduces the published packed bytes ([codec record](results/codec_probe.json)). This verifies those sampled codec/packing paths, not every group or the full runtime rotation.

### Native FP function check

The pinned [Qwen3.5-0.8B checkpoint](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17) provided a small 24-layer hybrid model with tied embeddings. The [fold script](make_folded_pilot.py) made explicit signed Hadamard variants, and the pinned converter exported ordinary and folded F32 GGUFs with `--no-mtp`. The [validation records](results/fp_pilot_validation.json) confirm 320 language tensors, a schema-3 tied-output contract, 150 folded projections plus the embedding for H512, and exactly 169 byte-identical unaffected tensors. An H1024 variant folded 126 projections plus the embedding; the 24 FFN down tensors have width 3584 and stayed unrotated because 3584 is not divisible by 1024 ([H1024 validation](results/fp_h1024_validation.json)).

The [native comparison harness](compare_fp_logits.cpp) used the same pinned `libllama`, F32 storage, CPU, 8 threads, 10 prompt tokens, all 10 prefill-logit rows, and one forced decode token for each pair. The [full per-position metrics](results/fp_pilot_results.json) are:

| Pair | Max prefill logit difference | Max decode logit difference | Largest relative L2 | Top token agreement |
|---|---:|---:|---:|---:|
| Public base vs H512 full fold | 0.00242 | 0.00292 | 0.000280 | 10/10 prefill; decode yes |
| Public base vs H1024 partial fold | 0.00299 | 0.00299 | 0.000229 | 10/10 prefill; decode yes |
| Same GGUF vs itself | 0 | 0 | 0 | 10/10 prefill; decode yes |

The public 0.8B checkpoint has equal numbers of K and V heads, so it cannot test the nontrivial GDN permutation. The [grouped-V fixture](make_grouped_v_pilot.py) crops 144 GDN tensors in a derived checkpoint from 16 K/16 V heads to 4 K/12 V heads, giving **three V heads per K head**. Its weights are a graph fixture, not a quality-preserving checkpoint ([derivative record](results/fp_grouped_derivative.json)). The pinned converter emitted `prism.hadamard.gdn_v_grouped=true`; the [grouped export validation](results/fp_grouped_validation.json) confirms 151 expected changed tensors and 169 untouched tensors. Native H512 base-versus-folded comparison has max prefill difference 0.0432, max decode difference 0.0208, largest relative L2 0.00182, and matching top tokens at 10/10 prefill positions and decode. The larger numerical gap may reflect amplification in the cropped recurrent network; it is not a quality metric.

A strong negative control changes **only** the folded GGUF's `prism.hadamard.gdn_v_grouped` boolean from true to false; a full bytewise comparison found exactly one changed byte. Its max prefill difference jumps to 25.16, largest relative L2 to 1.53, and top-token agreement falls to 0/10; decode also disagrees. The same-file grouped control is bit-exact. Together these runs validate that the pinned converter's grouped source-name branch and the native runtime permutation are functionally consequential in a model graph.

**Still required for the exact Bonsai 2 contract:** full-precision 27B prefill/decode comparison against an independently folded unrotated ancestor, if that ancestor becomes available and a streamed or larger-memory execution path is provisioned. The public 0.8B and synthetic grouped fixture validate the runtime mechanism, not Bonsai 2's unreleased latent weights. More boundary codec cases, PTQ1_0, and each future target's export policy still require their own checks. Pilot weight files and GGUFs are retained under ignored `work/qwen35_08b/` (about 27 GB logical file size, including a copy-on-write negative-control GGUF) and are absent from Git status.

### Packed export smoke test

The pinned quantizer's dry run predicted 193.35 MiB of tensor payload for a folded H512 PQ2_0 pilot. The actual file is **213,731,552 bytes** including its GGUF header. `--pure`, an explicit PQ2_0 embedding type, and the [54 recurrent-tensor overrides](results/quantizer_overrides_h512.txt) yielded exactly 151 PQ2_0, 36 BF16 (`ssm_alpha/beta`), and 133 F32 tensors. The tied-output/Hadamard contract was retained; nine sampled packed groups exactly match the pinned C reference quantizer applied to the folded F32 source ([packed validation](results/fp_packed_pilot.json)). The model loaded and completed prefill and decode in the pinned runtime.

On the single 10-token smoke prompt, this naive absmax arm changes every top token and reaches ≈24.9 maximum prefill logit difference from the F32 base. This is a **diagnostic of this one arm and prompt**, not a quality benchmark or evidence about Prism's training recipe. It confirms that valid format and runtime execution alone do not meet Phase 1's quality target; matched, predeclared evaluations are still needed.

## T1: all 449 protected tensors

After validated source transforms, **442 of 449 tensors differ** from this public Qwen revision. Seven 48-element `ssm_dt.bias` tensors are bit-exact: blocks 2, 6, 13, 25, 29, 33, and 41. Across 26,238,464 protected elements, 98.13% differ bitwise. The complete [per-tensor T1 data](results/t1.json) include exact counts, mean/max and 50th/95th/99th percentile absolute differences, relative L2, and correlation.

| Tensor class | Tensors | Changed elements | Mean absolute difference |
|---|---:|---:|---:|
| `attn_norm` | 64 | 95.16% | 0.00385 |
| `post_attention_norm` | 64 | 97.68% | 0.00834 |
| `ssm_alpha` | 48 | 97.66% | 0.00122 |
| `ssm_beta` | 48 | 98.72% | 0.00149 |
| `ssm_conv1d` | 48 | 98.42% | 0.00195 |
| `ssm_a` | 48 | 53.91% | 0.000442 |
| `ssm_dt.bias` | 48 | 4.99% | 0.0000839 |

The measured non-identity rules out an **unchanged protected-parameter projection of this exact public revision** under the checked conversion map. It does not distinguish an intermediate fine-tune, selective reconstruction, functional reparameterization, QAT, or another ancestor. The earlier review's small probe said every sampled tensor differed; the full scan narrows that statement by finding seven exact biases.

## T2–T4: complete discovery matrices

The candidate is `W_qwen S H/√1024`, derived from the pinned runtime's `R = H S/√1024` convention. Each result covers every row and 128-weight group in its matrix. Sign agreement is restricted to Bonsai nonzero positions; zero-set Jaccard uses the deterministic 86-largest-`|w|` support with stable index tie breaking. Scale-bit agreement is against the nonnegative signed least-squares optimum **given Bonsai's actual codes**, rounded to FP16. These are exact-rule tests conditional on the public candidate weights.

| GGUF tensor | Groups | Groups with 42 zeros | Nonzero-sign agreement | Fixed-86 support Jaccard | Exact fixed-86 groups | Signed-LS scale bits equal |
|---|---:|---:|---:|---:|---:|---:|
| `blk.0.ffn_down.weight` | 696,320 | 97.15% | 99.9816% | 0.8674 | 15 | 0.35% |
| `blk.3.attn_output.weight` | 245,760 | 97.49% | 99.99993% | 0.9099 | 147 | 1.81% |
| `blk.0.ssm_out.weight` | 245,760 | 97.26% | 99.99998% | 0.9143 | 429 | 2.00% |

The non-42 groups in these matrices all have **fewer** than 42 zeros; there are no >42-zero groups in this selection. The fixed-86 boundary ties are rare (32, 7, and 6 groups respectively), so ties cannot explain the large exact-support mismatch. The pinned absmax ternarizer applied to the same rotated public weights matches **zero complete groups** and zero exported scale bits in all three matrices. Detailed code agreement, scale errors, zero histograms, correlations, and sign agreement by within-group magnitude rank are in [matrix_probe.json](results/matrix_probe.json).

The GDN out-projection also has a valid norm-gamma symmetry. Replacing public columns by `W_qwen·diag(γ_qwen/γ_Bonsai)` before folding changes its support Jaccard from 0.9143 to 0.9083 and gives 279 exact fixed-86 groups, 37 nonzero-sign mismatches, and 3.03% signed-LS scale-bit agreement. The [gamma record](results/gdn_gamma_probe.json) retains the raw and corrected cases separately. Neither case is an exact simple projection.

### Held-out test of the frozen simple rules

Before reading the complete payloads, the [held-out manifest](results/heldout_manifest.json) fixed later-layer tensors `blk.32.ffn_down`, `blk.35.attn_output`, and `blk.32.ssm_out` and recorded the candidate formulas and analysis-script SHA-256. The script hash was unchanged after the run. The [complete held-out result](results/matrix_heldout.json) read 304.1 MB of exact ranges and found:

| Held-out GGUF tensor | Groups with 42 zeros | Nonzero-sign agreement | Fixed-86 support Jaccard | Exact fixed-86 groups | Signed-LS scale bits equal |
|---|---:|---:|---:|---:|---:|
| `blk.32.ffn_down.weight` | 96.56% | 99.9410% | 0.8451 | 1 / 696,320 | 0.92% |
| `blk.35.attn_output.weight` | 96.04% | 99.9941% | 0.8894 | 45 / 245,760 | 2.53% |
| `blk.32.ssm_out.weight` | 96.52% | 99.9887% | 0.8857 | 30 / 245,760 | 2.27% |

The pinned absmax rule again matches zero full groups and zero FP16 scale bits in every held-out matrix. This confirms rejection of those **exact rules relative to the pinned public revision** without selecting thresholds on the same matrix payloads. The lower support Jaccard and 42-zero rate at these later layers also show that a block-0-only summary would overstate uniformity.

## Candidate-rule decision at this gate

| Specified rule relative to this public revision | Result |
|---|---|
| Protected parameters left unchanged | Rejected by 442 changed tensors. |
| Pinned absmax ternarization of the rotated public weights | Rejected as an exact code-and-scale mapping on all three discovery and three held-out matrices. |
| Exactly 86 largest magnitudes per group | Rejected as an exact support rule on all six complete matrices; it remains a useful approximate baseline. |
| Nonnegative signed-LS scale from the public weights and Bonsai codes | Rejected as an exact exported-scale rule on all six complete matrices. |
| Predominantly sign-preserving assignment | Descriptively compatible, particularly for residual-writing attention/GDN outputs; no deterministic rule is identified. |

These decisions do **not** reject analogous rules applied to an unpublished, changed latent ancestor. Next: test other explicit support/scale rules with the same streamed method, finish the original 27B function gate if a suitable ancestor/execution path becomes available, then choose the cheapest matched quality experiment. Training has no entry criterion from these static differences alone.

## Reproduction

From the repository root, with Python 3.11+ and NumPy:

```bash
python3 analysis/bonsai2/replication/phase0.py inventory
python3 analysis/bonsai2/replication/phase0.py validate
python3 analysis/bonsai2/replication/contract_fixtures.py
python3 analysis/bonsai2/replication/codec_probe.py
python3 analysis/bonsai2/replication/layout_control.py
python3 analysis/bonsai2/replication/phase0.py t1
python3 analysis/bonsai2/replication/matrix_probe.py
python3 analysis/bonsai2/replication/gdn_gamma_probe.py
```

The scripts fail closed if a remote host ignores byte ranges. Each weight read is capped at 16 MiB; no checkpoint download is initiated by these commands.

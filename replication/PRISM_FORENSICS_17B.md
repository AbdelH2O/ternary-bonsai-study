# What Prism's 1.7B releases reveal about their method — static and behavioral forensics, 2026-10-06

**Why this exists:** before committing Modal credit to the [1.7B training design](QAT17B_DESIGN.md), the user asked whether anything more about Prism's quantization could be learned. This note compares Prism's three Qwen3-1.7B releases with each other, with the Qwen3-1.7B ancestor, and with our own 0.8B training runs. It is descriptive and changed no frozen experiment. Evidence tags: 🟢 measured; 🟡 inferred; 🟣 proposed.

Scripts and records:
- [prism_variants_17b.py](prism_variants_17b.py) → [results/prism_variants_17b.json](results/prism_variants_17b.json): all 197 quantized tensors, 1.72B weights.
- [prism_lineage_17b.py](prism_lineage_17b.py) → [results/prism_lineage_17b.json](results/prism_lineage_17b.json): Prism's layers 0/9/18/27, plus our 0.8B control.
- [teacher_style_probe.py](teacher_style_probe.py) → [results/teacher_style_probe.json](results/teacher_style_probe.json): Qwen3-4B/8B Q8_0 GGUFs from `Qwen/Qwen3-{4B,8B}-GGUF`, in `work/qwen3_teachers/`.

Files used:
- `Ternary-Bonsai-1.7B-PQ2_0.gguf` (Phase 1 copy);
- `Ternary-Bonsai-1.7B-Q2_0_g64.gguf` (sha256 `6d0ecb3d…`);
- the 1-bit `Bonsai-1.7B-Q1_0.gguf` (sha256 `3d7c6c90…`), both downloaded today into `work/prism17_variants/`;
- the pinned Qwen3-1.7B@`70d244cc`.

## Findings

### 1. The g64 file is a lossless repack, not a second projection 🟢

The g64 ternary file has the same codes as the g128 file in every one of 1.72B positions. Both g64 halves of each 128-group carry the g128 scale (100% of sampled groups in two blocks and the embedding; 99.998% in `blk.0.ffn_up`). It carries no information about the underlying weights.

### 2. Every scale is a BF16 value 🟢

All ternary scales (13.4M groups) are FP16 numbers exactly representable in BF16; for the 1-bit file it's 99.997%. Phase 1 had found this only for the embedding (scale = mean |w| rounded to BF16, then stored as FP16).

🟡 Prism's projection scales were therefore produced in BF16 too, consistent with BF16 training on TPUs, where the absmean of BF16 latent weights is itself BF16. Our quantizer rounds the FP32 mean straight to FP16.

### 3. Both releases use the same data-free embedding 🟢

The ternary and 1-bit embeddings have identical scales in every group (s₁/s₁₂₈ = 1.000) and 99.9% identical signs. Both equal the ancestor's sign and its group mean |w|. Neither model trained the embedding; both projected it from the ancestor with the same scale rule.

### 4. Prism's training reshaped the weights; ours barely moved them 🟢

[CODE_DRIFT.md](CODE_DRIFT.md) (2026-10-05) already showed the size of the movement gap at the code level: 39% of Prism's codes differ from plain rounding of the ancestor, against 14% for our students, and 2.9% vs 0.04% outright sign flips. It recommended a full-length higher-learning-rate arm. What's new here is the *shape* of Prism's change: its scales and zero fraction show the underlying weights grew about 2× and became more peaked. Bins by ancestor size show its flips reach weights of average and above-average size.

| | Untrained Qwen3-1.7B, absmean | Prism ternary 1.7B | Our 0.8B QAT, absmean, 65.5M tokens (`qat_free`) |
|---|---:|---:|---:|
| Zero fraction, projections | 31.7% (30.9% in our H1024 basis) | **39.9%** (38.2–40.9% by role) | 30.9% → **31.8%** |
| Nonzero sign agreement with ancestor | 100% | **89.4%** | **99.3%** |
| Flip rate of nonzero weights by ancestor size, 1–2× group mean / 0.5–1× | 0 | **4.6% / 15.6%** (layers 0/9/18/27) | 0.0% / 0.2% |
| Group scale vs ancestor group mean \|w\| | 1× | **≈ 2×** (median 1.73–2.14 by depth) | — |
| Latent weight change ‖L − W‖/‖W‖ | 0 | — (latents not released) | 0.09 at 8M tokens → 0.195 at 65.5M |

- **The scale doubled and zeros rose from 32% to 40%.** Under absmean (scale = mean |L|, zero iff |L| < scale/2), both changes mean the underlying weights grew about 2× and became more peaked. 40% zeros is what a Laplace-shaped distribution gives (39.3%); a Gaussian gives 31%, which is exactly the untrained ancestor and the untouched embedding.
- 🟡 **One plausible mechanism.** Long straight-through training with Adam and no weight penalty lets strongly pushed weights keep growing. That raises the group mean, and more small weights fall below the zero threshold. Our run shows the same trend, very weakly: +0.9 points of zeros in 65.5M tokens. An explicit regularizer, such as the mirror-descent potential in the Hassibi–Lale patent lead ([research notes](research_notes/Bonsai%20quantization%20models%20and%20compute/prism_bonsai.md)), could produce the same fingerprint. Static data can't tell these apart.
- 🟡 **Prism's training moved the weights roughly an order of magnitude further than ours.** Ours flipped almost no weight above half the group mean; Prism flipped 4.6% of weights at 1–2× the group mean. This is a correlate of their quality, not a proven cause. It is the best available measure of how far our run is from their regime.
- 🟢 **The 1-bit sibling moved even further:** scales 1.9–2.5× the ancestor's, and 23.3% of signs flipped overall (62.7% agreement on the weights ternary zeroed).

### 5. No evidence that the ternary model was started from the 1-bit model 🟢

- **Raw coincidence.** Where the ternary model flipped an ancestor sign, the 1-bit model flipped it too 80% of the time, versus 7% where ternary kept it. Ternary nonzero signs agree more with the 1-bit model (91.8%) than with the ancestor (89.4%).
- **Control.** Two *independent* runs of ours (absmean and top-86, same init, data and objective) show the same pattern, even stronger: within each ancestor-size bin, P(other flips | one flips) is 0.89 / 0.79 / 0.67 / 0.53, against Prism's 0.80 / 0.78 / 0.76 / 0.71.
- **Reading.** A shared starting point and objective explain the coincidence. It doesn't show a shared training run.

### 6. Prism answers GSM8K in a different voice from its ancestor 🟢 (Phase 1 outputs, test set, descriptive)

- **Ability.** Prism solves 34% (81/238) of the items Qwen3-1.7B fails; it fails 16.4% of the items Qwen3-1.7B solves. When both are wrong, they give the same wrong number 34% of the time (54/157).
- **Style.** Only 0.2% of answers share their first 200 characters.
  - Qwen3-1.7B's commonest opening: "We are given the following information:" (622 / 1,319).
  - Prism's: "Let's break down the problem step by step." (606 / 1,319), an opening Qwen3-1.7B uses 138 times.
  - Format is preserved: boxed answers 98–99%, markdown headers, similar length (301 vs 308 tokens).
- 🟡 **Two readings:**
  - (a) a small nudge between two openings that are nearly tied for Qwen3-1.7B;
  - (b) training on responses from a different teacher, e.g. a larger Qwen3, which shares the vocabulary.

  The teacher-style probe below tests this.

### 7. Prism doesn't answer like a larger Qwen3 either 🟢

Greedy openings on 100 GSM8K *train* questions (items 100–199), Phase 1 prompt, thinking off, CPU. First-token probabilities were requested but not captured (the server's response field didn't match the probe), so only greedy text is compared.

Counts are summed over each model's five commonest first lines ("+" marks a lower bound).

| Model | "We are given…" openings | "Let's …" openings | Top opening |
|---|---:|---:|---|
| Qwen3-1.7B (F32) | 64 | 12 | "We are given the following information:" (43) |
| Qwen3-4B (Q8_0) | 39+ | 21+ | "We are given the following information:" (23) |
| Qwen3-8B (Q8_0) | 35+ | 19+ | "We are given the following information:" (19) |
| **Prism 1.7B** | **0** in its top 5 | **70** | "Let's break down the problem step by step." (51) |

Same first line: Qwen3-1.7B ~ 4B 46%, 1.7B ~ 8B 32%, 4B ~ 8B 32%; **Prism ~ 1.7B 12%, Prism ~ 4B 15%, Prism ~ 8B 4%.**

- 🟢 **Every Qwen3 size shares one house style; Prism has left it.** The larger-Qwen3-teacher reading of finding 6 is not supported for 4B or 8B. 14B and 32B are untested.
- 🟡 **This is hard to get from pure KL to Qwen3.** At the first response token, KL to any Qwen3 teacher targets the teacher's own distribution, which puts most of its weight on "We are given". A ternary student under that loss should keep the same greedy opening unless the options are nearly tied. Here they aren't: 64 vs 12 for Qwen3-1.7B.
- 🟡 **A switch to "Let's …" in 70 of 100 answers suggests something else in Prism's training targets.** Candidates: hard-label (cross-entropy) training on response data in another style; a teacher outside the tested Qwen3 sizes; or preference/RL tuning. "Let's break down the problem step by step" is a common opening in widely used synthetic math corpora. That is a hint, not evidence of a specific dataset.

## What this changes in the 1.7B design (🟣 proposals, for the user to accept or reject)

1. **Keep absmean, and round scales through BF16** (mean |L| → BF16 → FP16) to match Prism's stored scales exactly. Cost-free. It also aligns the Question A mapping.
2. **Measure how far the weights travel.** At every checkpoint, log:
   - zero fraction;
   - median scale relative to the ancestor's group mean |w|;
   - nonzero sign agreement with the ancestor;
   - flip rate by ancestor-size bin.

   Report the trajectory against Prism's 39.9% / ≈2× / 89.4%. This tells us whether more tokens or larger steps are the missing ingredient, and feeds the extension rule.
3. **Revisit the learning-rate rule.** A 150-step monitor-KL probe favours safe, small steps. Prism's fingerprint suggests far larger cumulative movement than the 0.8B recipe produced. Proposal:
   - extend the probe to 300 steps;
   - break ties toward the larger rate when final KLs are within 2%;
   - add 4e-4 to the grid.

   About $1 more.
4. **Keep the Qwen3-1.7B teacher** (finding 7 doesn't favour 4B or 8B). Record as a known difference that Prism's targets likely included non-Qwen-style responses, perhaps hard labels. Our KL-only student will keep Qwen3's style; style itself isn't scored, but it's a visible difference from Prism to report.

## Limits

- Prism's latent weights are unreleased; every statement about them goes through the absmean rule, which fits the embedding near-exactly but is assumed for the projections.
- Our control is a different model (Qwen3.5-0.8B, hybrid), measured in its folded H512 basis, and trained 65.5M tokens. It shows what a shared objective produces, not what Prism's process was.
- GSM8K style evidence comes from greedy decoding on one prompt format.

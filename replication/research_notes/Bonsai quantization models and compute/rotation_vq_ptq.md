# Rotation, incoherence and vector/trellis PTQ (plus fine-tuned PTQ) at ~1.5-2.5 bpw: state of the art as of Oct 2026, and what is worth combining with ternary QAT

Research date: 2026-10-05. How the sources were checked: items marked **[fetched]** were read this session (the arXiv HTML or abstract, or the GitHub page). Items marked **[abstract, from prior knowledge; not re-fetched]** are well-known abstract-level claims, cited to their canonical URL. Re-check those before quoting the numbers in a final report.

Conversion used throughout: a perplexity ratio r equals ln(r) nat/token of extra NLL. This lets the paper numbers be compared with the team's "+1.5-1.6 nat/token" residual.

---

## 1. Rotation methods (QuaRot, SpinQuant, FlatQuant, DuQuant, OSTQuant, KurTail, QuIP#/QTIP RHT): which help weight-only low-bit, and learned vs fixed

### Takeaway
At <=2.5 bpw weight-only, the best methods all use a **fixed random Hadamard transform (RHT)** for incoherence. Learned rotations (SpinQuant, OSTQuant, FlatQuant, KurTail) were designed and reported for W4A4(KV4), where activation outliers are the main problem. None of them reports a weight-only gain at ~2 bits large enough to matter next to the choice of codebook and fine-tuning. Rotation is necessary but not sufficient: the 2-bit frontier comes from vector/trellis codes plus fine-tuning on top of RHT.

### Cited Findings
- QTIP uses incoherence processing by RHT, W~ <- V_m S_m W S_n V_n^T (Hadamard matrices times random sign vectors), to make weights approximately i.i.d. Gaussian before trellis quantization **[fetched]**. — [QTIP arXiv HTML](https://arxiv.org/html/2406.11235)
- SpinQuant (ICLR 2025; submitted May 26 2024) learns rotations (Cayley optimization) for W4A4KV4. It narrows the zero-shot gap to FP to 2.9 points on LLaMA-2 7B, beats LLM-QAT by 19.1 and SmoothQuant by 25.0 points, and on LLaMA-3 8B "reduces the gap to full precision by up to 45.1% relative to QuaRot". The abstract makes no weight-only claim **[fetched]**. — [SpinQuant arXiv](https://arxiv.org/abs/2405.16406)
- QuaRot (2024) uses fixed (randomized) Hadamard rotations folded into weights so that W4A4KV4 works end to end; on LLaMA-2 70B it loses at most ~0.47 WikiText-2 PPL and keeps ~99% of zero-shot accuracy **[abstract, from prior knowledge; not re-fetched]**. — [QuaRot arXiv](https://arxiv.org/abs/2404.00456)
- FlatQuant (Oct 2024) learns per-layer affine (Kronecker-factored) transforms to flatten weights and activations; the abstract claims <1% accuracy drop for W4A4 on LLaMA-3-70B, 7.5% better than SpinQuant **[abstract, from prior knowledge; not re-fetched]**. — [FlatQuant arXiv](https://arxiv.org/abs/2410.09426)
- DuQuant (NeurIPS 2024) combines block rotations with a zigzag channel permutation to spread "massive" activation outliers across blocks. It targets W4A4 / W6A6 **[abstract, from prior knowledge; not re-fetched]**. — [DuQuant arXiv](https://arxiv.org/abs/2406.01721)
- OSTQuant (ICLR 2025) learns orthogonal plus scaling transforms with a KL-Top loss. Reported: ~99.5% of FP accuracy for W4-only and a ~32% smaller gap for W4A4KV4 on LLaMA-3-8B than prior SOTA **[abstract, from prior knowledge; not re-fetched]**. — [OSTQuant arXiv](https://arxiv.org/abs/2501.13987)
- KurTail (Mar 2025) learns rotations by minimizing activation kurtosis (single-GPU training) and claims gains over QuaRot and SpinQuant for W4A4 **[abstract, from prior knowledge; not re-fetched]**. — [KurTail arXiv](https://arxiv.org/abs/2503.01483)
- HIGGS (Nov 26 2024) uses Hadamard rotation plus MSE-optimal grids, **data-free**. It proves a "linearity theorem": perplexity increase is linear in the layer-wise l2 reconstruction error. That gives optimal non-uniform per-layer bit allocation via dynamic programming. Evaluated on Llama-3.1/3.2 and Qwen **[fetched]**. — [HIGGS arXiv](https://arxiv.org/abs/2411.17525)

### Inferences
- In the weight-only regime, a rotation only needs to make weights Gaussian-like and incoherent. A random Hadamard already does this near-optimally. Learned rotations mostly solve activation outliers, which are irrelevant for a W1.6/A16 llama.cpp runtime. **Expect little weight-only gain from replacing H512/H1024 with learned rotations.** A cheap check: measure weight kurtosis and max/RMS per block before and after the transform.
- Rotation block size is a real knob. QuIP#/QTIP rotate the full row/column dimension (Kronecker of Hadamards), whereas Bonsai uses H512/H1024 blocks. Smaller blocks give weaker incoherence for channels with heavy outliers. The DuQuant-style permutation-before-block-rotation trick spreads outliers across blocks and costs nothing at inference if it is folded in.
- HIGGS's linearity theorem justifies allocating bits per layer from measured sensitivity. Its data-free result implies that **a better grid** (MSE-optimal for Gaussian) matters more than **more data**. That matches the team's finding that 4x calibration data gave no gain.

### Gaps
- I found no paper that directly ablates learned vs random rotation for **weight-only ~2-bit ternary**. OSTQuant reports W4-only only.
- I did not retrieve KurTail, DuQuant or FlatQuant tables this session.

---

## 2. Vector / trellis / codebook quantization at <=2 bits (AQLM, QuIP#, QTIP, PV-tuning, VPTQ, GPTVQ, HIGGS, EXL3, llama.cpp IQ/TQ types)

### Takeaway
At 2.0 bpw the PTQ-plus-light-fine-tune frontier on Llama-3-8B is about **+0.23 to +0.35 nat/token on WikiText-2** (PPL 6.99-7.84 vs BF16 5.54). Llama-3-70B is proportionally worse (+0.57 to +0.65 nat). High-dimensional codes (E8 lattice, additive codebooks, trellis) beat scalar grids by a consistent margin at 2 bits, and QTIP (trellis) is the strongest published pure-PTQ representation. EXL3 brings QTIP to a production inference stack. For comparison, llama.cpp's ternary types TQ1_0 and TQ2_0 are plain scalar ternary with one fp16 scale per 256 weights.

### Cited Findings
QTIP (NeurIPS 2024 Spotlight; submitted June 17 2024, revised June 18 2025) **[fetched]**, from the [QTIP arXiv HTML](https://arxiv.org/html/2406.11235), WikiText-2 / C4 PPL at 2 bits:

| Model (ctx) | FP | QTIP 2-bit | QuIP# 2-bit | AQLM 2-bit |
|---|---|---|---|---|
| Llama-2 7B (4096) | 5.12 / 6.63 | 5.86 / 7.73 | 6.19 / 8.16 | 6.14 / 8.09 |
| Llama-2 70B (4096) | 3.12 / 4.97 | 3.70 / 5.48 | 3.91 / 5.71 | 3.83 / 5.62 |
| Llama-3 8B (8192) | 5.54 / 7.10 | 7.33 / 8.62 | 7.84 / 9.06 | — |
| Llama-3 70B (8192) | 2.59 / 5.78 | 4.97 / 6.80 | 5.77 / 7.46 | — |

- In nats (my conversion): Llama-3 8B is +0.28 for QTIP and +0.35 for QuIP#. Llama-3 70B is +0.65 for QTIP and +0.80 for QuIP#.
- QTIP decode throughput at 2-bit on an RTX 6000 Ada: 188 tok/s for Llama-2 7B and 23.5 tok/s for 70B. Its lookup-free "1MAD"/"3INST" codes decode in about 2-3 ALU instructions per weight. QTIP "generally matches or exceeds QuIP# and AQLM" on zero-shot tasks (ArcC/ArcE/PiQA/Wino). — [QTIP arXiv HTML](https://arxiv.org/html/2406.11235)
- PV-tuning (May 2024; NeurIPS 2024) **[fetched]** is a representation-agnostic fine-tuning method that alternates continuous (P) and discrete (V) updates. It replaces straight-through estimation (STE), which the authors call sub-optimal. It claims "the first Pareto-optimal quantization for Llama 2 family models at 2 bits per parameter". WikiText-2 results:
  - Llama-2 7B: 5.84 at ~2 bits (QuIP# 6.19) and 8.28 at ~1 bit (OneBit 9.73).
  - Llama-2 70B: 3.78 at ~2 bits.
  - Llama-3 8B: 6.99 at ~2 bits, i.e. +0.23 nat vs 5.54.
  - Llama-3 70B: 4.57, i.e. +0.57 nat vs 2.59.
  - Mistral-7B: 5.29.
  Cost is up to ~1.5x the prior fine-tuning; 7B fits on 1 GPU and 70B needs 8xA100. — [PV-tuning arXiv HTML](https://arxiv.org/html/2405.14852), [abstract](https://arxiv.org/abs/2405.14852)
- QuIP# (ICML 2024) uses RHT incoherence, an E8-lattice "E8P" codebook at 2 bits, and inter-layer fine-tuning **[abstract, from prior knowledge; not re-fetched]**. — [QuIP# arXiv](https://arxiv.org/abs/2402.04396), [code](https://github.com/Cornell-RelaxML/quip-sharp)
- AQLM (ICML 2024) uses additive multi-codebook quantization with a global fine-tune, and claims Pareto optimality below 3 bpw **[abstract, from prior knowledge; not re-fetched]**. — [AQLM arXiv](https://arxiv.org/abs/2401.06118), [code](https://github.com/Vahe1994/AQLM)
- VPTQ (EMNLP 2024) is second-order-guided vector PTQ at ~2 bits. It reports large gains on LLaMA-3 over prior 2-bit PTQ and a fraction of AQLM's quantization time **[abstract, from prior knowledge; not re-fetched]**. — [VPTQ arXiv](https://arxiv.org/abs/2409.17066)
- GPTVQ (2024) extends GPTQ's Hessian-aware column sweep to vector quantization **[abstract, from prior knowledge; not re-fetched]**. — [GPTVQ arXiv](https://arxiv.org/abs/2402.15319)
- PTQTP (arXiv 2509.16989, v3 dated Jan 1 2026) **[fetched]** is a training-free ternary-native method; details are in section 4. It reports 73.4x faster quantization than AQLM and 1.57x faster than AWQ on LLaMA-7B. — [PTQTP arXiv HTML](https://arxiv.org/html/2509.16989v3)
- EXL3 (exllamav3) is "a streamlined variant of QTIP". It computes Hessians on the fly with a fused Viterbi kernel. Conversion takes "a couple of minutes for smaller models, up to a few hours for larger ones (70B+) on a single high-end consumer GPU". The repo hosts a Llama-3.1-8B-Instruct bits-per-weight vs quality chart, but I could not extract the numbers **[fetched]**. — [exllamav3 GitHub](https://github.com/turboderp-org/exllamav3)
- llama.cpp TQ1_0 / TQ2_0 (PR #8151, merged Sept 6 2024) **[fetched]**, from the [llama.cpp PR #8151](https://github.com/ggml-org/llama.cpp/pull/8151):
  - TQ1_0 is 1.6875 bpw: 256-element blocks, 240 trits packed 5 per byte plus 16 packed 4 per byte, and one fp16 scale (54 bytes per block).
  - TQ2_0 is 2.0625 bpw: 2 bits per trit plus an fp16 scale (66 bytes per block). It is the fastest of the types tested, about 2x Q4_K throughput on an AVX2 laptop.
  - Both target BitNet b1.58 and TriLM. They store one scale per 256 weights, with no sub-block scales and no codebook.
- llama.cpp's i-quants (IQ2_XXS ~2.06 bpw, IQ1_S ~1.56 bpw, IQ1_M ~1.75 bpw) are importance-matrix-weighted lattice/grid codebooks by ikawrakow. IQ2_XXS uses a 256-entry grid derived from the E8 lattice, inspired by QuIP# **[from prior knowledge; not re-fetched; PR numbers not verified this session]**. — [llama.cpp quantize README](https://github.com/ggml-org/llama.cpp/tree/master/tools/quantize)
- "Any-Precision LLM" (ICML 2024) overlays 3-8-bit models in one memory footprint, so it is not relevant at <2.5 bpw **[from prior knowledge]**. — [arXiv](https://arxiv.org/abs/2402.10517)

### Inferences
- **Scale check against the team's residual.** The team reports +1.5-1.6 nat/token (a PPL ratio of about 4.5-5x). The best published 2-bit PTQ+FT is +0.23-0.35 nat on Llama-3-8B and +0.57-0.80 on 70B. Even the training-free ternary PTQTP is +0.33 nat on Llama-3.1-8B (see section 4). Unless the team's nat figure is measured differently (instruction or reasoning data, KL to the teacher, a 27B hybrid model), their one-shot result is **roughly 3-5x worse in nats** than the 2-bit frontier. That points to a representation or optimization bottleneck, not a calibration-data one, which is consistent with 4x data giving nothing.
- **The rate-distortion argument for why scalar ternary loses** (my derivation from textbook rate-distortion facts, not cited):
  - For i.i.d. Gaussian weights after RHT, the Shannon bound at R bits is D = 2^(-2R). At 1.58 bits that is about 0.11 sigma^2, and at 2 bits about 0.0625 sigma^2.
  - The best 3-level scalar (Lloyd-Max) quantizer reaches only about 0.19 sigma^2, with thresholds near ±0.61 sigma, about 46% zeros, and entropy about 1.54 bits.
  - Trellis/lattice codes such as QTIP and E8P get within a fraction of a dB of the bound. That is the structural reason they beat scalar grids at equal bpw.
  - A fixed budget of 42 zeros per 128 weights (32.8%) is fewer zeros than the Gaussian-MSE-optimal ~46%. If the budget is enforced exactly rather than as a maximum, it adds distortion. Check whether "42" is an encoding necessity: log2 C(128,42) + 86 sign bits is about 1.55 bits/weight, plus a fp16 scale per 128 gives about 1.68 bpw.
- The team owns its llama.cpp fork (PQ2_0 and PTQ1_0 are already custom types), so a trellis or vector type is technically possible. EXL3 shows that QTIP-class decoding is practical on GPUs. The open question is CPU/Metal decode cost, since QTIP's codes are designed around GPU ALU throughput.

### Gaps
- I could not extract the EXL3 vs GGUF chart numbers or any public EXL3 results below 2 bpw.
- I did not retrieve QTIP's 1-bit / 1.5-bit tables or any QTIP / AQLM / PV-tuning results on Qwen3 or Qwen3.5.
- MLX and vLLM kernel support:
  - vLLM has had AQLM and (via HF) QuIP#-style kernels; status not verified this session.
  - MLX supports affine 1-8-bit quantization natively. I found no MLX trellis or VQ kernels.

---

## 3. Fine-tuning after PTQ (PV-tuning, AQLM global fine-tune, EfficientQAT Block-AP / E2E-QP): tokens, compute and how much gap it closes

### Takeaway
At 2 bits, fine-tuning is where most of the remaining gain comes from. A cheap recipe is block-wise reconstruction followed by end-to-end tuning of only the continuous parameters (scales, norms). PV-tuning-style discrete updates add a further step beyond STE. Both cost GPU-days, not GPU-months, for 7-70B models.

### Cited Findings
- PV-tuning vs alternatives (Table 1) **[fetched]**:
  - Optimizing only continuous parameters is worse by roughly 2-3 PPL points at extreme bits.
  - PV combined with STE gives a further ~0.3-0.4 PPL.
  - Compute is up to 1.5x prior fine-tuning; 7B fits on one GPU and 70B on 8xA100.
  - The token count and calibration size were not in the extracted tables.
  — [PV-tuning arXiv HTML](https://arxiv.org/html/2405.14852)
- EfficientQAT (2024) has two phases, Block-AP (block-wise training of all parameters) then E2E-QP (end-to-end training of only the quantization step sizes). It reports a 2-bit Llama-2-70B in ~41 h on a single A100-80GB, with <3 points average zero-shot loss (69.48 vs 72.41) **[abstract, from prior knowledge; not re-fetched]**. — [EfficientQAT arXiv](https://arxiv.org/abs/2407.11062)
- PTQTP frames its pitch as "single-hour quantization versus 10-14 GPU days for training-based methods". This is a useful reference for what the field considers QAT cost at 7-8B **[fetched]**. — [PTQTP arXiv HTML](https://arxiv.org/html/2509.16989v3)
- ParetoQ (Feb 2025) unifies QAT across 1, 1.58, 2, 3 and 4 bits. It reports a learning transition between 2 and 3 bits: below it, the representation changes drastically during QAT instead of staying close to the pre-trained weights. Its ternary 600M model beats prior ternary 3B models **[abstract, from prior knowledge; not re-fetched]**. — [ParetoQ arXiv](https://arxiv.org/abs/2502.02631)

### Inferences
- The team's PTQ start point leaves +1.5 nat, so QAT is justified. The literature says that below ~2 bits, QAT moves weights far from initialization (ParetoQ), so a better PTQ start mainly saves early QAT steps. It does not change the final quality much.
- What **is** worth taking from the PTQ-fine-tuning literature into their QAT:
  1. **PV-style discrete updates.** Periodically re-assign ternary codes (subject to the zero budget) by minimizing the loss linearized at the current weights, while continuous scales are trained by gradient. This beats pure STE at 1-2 bits.
  2. **The EfficientQAT two-phase schedule.** Block-wise distillation first (cheap, layer-local, parallel across blocks), then a short global KD that tunes only scales, norms and any fp16 side parameters.
  3. **Logit KD to the FP teacher** (as AQLM/PV do for the global fine-tune), which they already do.

### Gaps
- I could not confirm exact token budgets (AQLM / PV-tuning use on the order of 10^7-10^8 tokens of RedPajama-style data, but this was not verified this session).
- I found no published PV-tuning results on ternary (3-level scalar) codes specifically.

---

## 4. Ternary-specific PTQ and QAT (PTQ1.61, PTQTP, Tequila, ParetoQ, BitNet-style)

### Takeaway
Training-free ternary PTQ improved a lot in 2025-2026. PTQTP reports Qwen3-14B keeping about 95-96% of MMLU / GSM8K / Math-500. Its bit accounting ("1.58-bit" for two trit-planes) needs scrutiny before comparing it with Bonsai's 1.75 bpw. On the QAT side, Tequila (ICLR 2026) names and fixes the exact pathology a ternary STE QAT will hit: **deadzone trapping**.

### Cited Findings
- **PTQTP** (arXiv 2509.16989; v3 Jan 1 2026; HKU / PolyU / UCSB) **[fetched]**, from the [PTQTP arXiv HTML](https://arxiv.org/html/2509.16989v3):
  - Method: it decomposes each weight matrix into **two** ternary {-1,0,1} trit-planes with row-wise fp16 scales and group size 128, using a progressive approximation algorithm. Inference is multiplication-free.
  - WikiText-2 PPL:
    - LLaMA-3.1-8B: 6.14 to 8.51, i.e. +0.33 nat (my conversion).
    - LLaMA-3-70B: 2.86 to 6.12, i.e. +0.76 nat.
    - LLaMA-2-7B: 6.30, while AWQ at 2 bits collapses to about 2.2e5.
  - Qwen3-14B:
    - MMLU 76.20 vs 79.38 FP (−3.2 points).
    - GSM8K 85.44 vs 89.39 (−4.0 points).
    - Math-500 82.40 vs 86.60 (−4.2 points).
  - Speed and cost: 4.63x decode speedup over FP16 on an RTX 3090, and about 1 hour to quantize.
  - **Caution:** the paper labels it "b1.58-Dual". Two trit-planes naively need about 3.17 bits/weight (2 x log2 3) plus scales, so "1.58 bpw" is likely a per-plane label. Confirm the storage accounting before treating it as comparable to Bonsai's 1.75 bpw.
- **PTQ1.61** (arXiv 2502.13179; Feb 18 2025, revised Aug 2025) **[fetched]**:
  - Uses a 1-D structured mask that keeps salient input channels at 4-bit for about 0.0002 bit/weight of overhead.
  - Binarizes the other channels with block-wise optimized scales.
  - Adds a "quantization preprocessing" step that reshapes the pretrained weight distribution before quantizing.
  - Claims SOTA for extremely low-bit PTQ. Numbers were not in the abstract.
  — [PTQ1.61 arXiv](https://arxiv.org/abs/2502.13179), [code](https://github.com/zjq0455/PTQ1.61)
- **Tequila** (arXiv 2509.23809; Sept 28 2025; ICLR 2026) **[fetched via search abstract]**:
  - Ternary QAT suffers "deadzone trapping": many weights sit at the deadzone boundary and get only noisy, uninformative STE gradients, so they cannot escape.
  - Tequila repurposes the trapped weights as **dynamic biases**. They provide a continuous forward signal and get direct gradients, with near-zero inference overhead.
  - Reports a >4% ARC gain over the previous SOTA ternary method, within <1% of FP, with a 3.0x inference speedup.
  — [Tequila arXiv](https://arxiv.org/abs/2509.23809), [ICLR 2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5b555804d495321df2e3208cc27f4fbc-Abstract-Conference.html)
- **ParetoQ**: see section 3 (ternary and 2-bit QAT scaling; transition below ~2-3 bits) **[abstract, from prior knowledge]**. — [ParetoQ arXiv](https://arxiv.org/abs/2502.02631)
- **BitNet b1.58 2B4T** (Apr 2025) is native ternary pretraining (4T tokens) that reaches FP-competitive quality at 2B. It is the reference point for trained-from-scratch ternary **[abstract, from prior knowledge; not re-fetched]**. — [arXiv 2504.12285](https://arxiv.org/abs/2504.12285)

### Inferences
- Bonsai's runtime is a single scaled trit-plane per group of 128 with a zero budget. PTQTP's evidence suggests that a **second, sparse residual trit-plane** (applied only to the most sensitive layers or rows) is the most "ternary-native" way to buy quality, with multiplication-free kernels preserved. The cost is extra bits: each plane adds about 1.6 bpw, so it is only viable as targeted mixed precision.
- Tequila is directly relevant to the team's QAT. With an enforced 42-zero budget, the weights ranked near the zero/nonzero boundary are exactly the trapped population. Possible mitigations:
  - Tequila-style bias reactivation.
  - PV-style periodic hard re-assignment instead of STE flipping.
  - Annealing the threshold or budget.
- PTQ1.61's salient-channel idea (input channels kept at higher precision, chosen with an imatrix-like activation statistic) maps onto Bonsai's rotated basis only partly. After a Hadamard rotation, salience is spread across channels by design. Salience-based mixed precision therefore probably works better at the **layer or matrix level** than at channel level in a rotated model.

### Gaps
- I found no published head-to-head of PTQTP vs Tequila vs PTQ1.61 on the same model.
- I found no ternary PTQ results on hybrid GDN models such as Qwen3.5/3.6-27B.
- I did not find a "sparsity-aware ternary" paper with a fixed per-group zero budget like Bonsai's 42/128.

---

## 5. What should stay full precision; mixed-precision allocation

### Takeaway
Standard practice at ~1.5-2 bpw:
- Keep embeddings, lm_head, norms and biases in higher precision.
- Spend more bits on the first/last blocks and on attention v/o and MLP down projections.
- Allocate by measured sensitivity (imatrix or linearity-theorem l2 error × sensitivity).
The cost is about 0.05-0.2 bpw on large models, and it consistently buys more than further per-layer optimization.

### Cited Findings
- HIGGS's linearity theorem gives a principled per-layer allocation: perplexity increase is approximately the sum over layers of a_l × (l2 error at layer l), with bitwidths chosen by DP under a total budget **[fetched]**. — [HIGGS arXiv](https://arxiv.org/abs/2411.17525)
- PTQ1.61 keeps a structured salient-channel subset at 4-bit for about 0.0002 extra bit/weight **[fetched]**. — [PTQ1.61 arXiv](https://arxiv.org/abs/2502.13179)
- TQ1_0 / TQ2_0 in llama.cpp only cover the linear weights. Whatever is not converted (token embedding, output) stays in another GGUF type, which in practice means high precision for ternary-trained models **[fetched]**. — [llama.cpp PR #8151](https://github.com/ggml-org/llama.cpp/pull/8151)
- In hybrid GDN models, the gate projections (decay and write-strength) were assumed fragile but turned out to be the **least** sensitive layers at 4-bit; see section 6 **[fetched via search abstract]**. — [arXiv 2609.04098](https://arxiv.org/pdf/2609.04098)

### Inferences
- For a 27B hybrid model, embeddings plus lm_head are about 2 × vocab (~150-250k) × hidden (~5k), roughly 1.5-2.5B parameters, or 5-10% of all parameters. Keeping them at Q4-Q6 instead of ternary is the single largest "free" quality lever if the size budget allows.
- GDN gate projections are probably safe to ternarize. Spend the extra bits on the softmax-attention v/o projections and the MLP down projections instead.

### Gaps
- I did not obtain a fetched source with measured numbers for "which Qwen-family layers are most sensitive at <2 bpw".

---

## 6. Quantizing hybrid linear-attention / SSM models (Mamba, Gated DeltaNet, Qwen3-Next / Qwen3.x-27B) at low bits

### Takeaway
All the evidence I found is at 4-8 bits; I found nothing at <=2 bpw. The strongest recent result (Sept 2026) is that Gated DeltaNet layers, gates included, survive **NVFP4 W4A4** with no measurable loss, and that recurrence noise plateaus rather than compounding over 32K tokens. This removes the main worry specific to hybrid models.

### Cited Findings
- "Why Gated DeltaNet Survives 4-Bit Quantization: NVFP4 W4A4 for the Recurrent Half of a Hybrid 27B LLM" (arXiv 2609.04098, Sept 3 2026) **[fetched via search snippet; full paper not read]**. — [arXiv PDF](https://arxiv.org/pdf/2609.04098), [paper page](https://paperswithcode.co/paper/2609.04098)
  - The model is the hybrid 27B (48 GDN plus 16 attention layers; the snippet calls it "Qwen3.8-27B").
  - Their recipe "Minima" quantizes all 496 linear layers to NVFP4 W4A4, GDN included. It matches BF16 within seed noise on 4K/32K perplexity, MMLU-Pro, GSM8K, AIME'25, GPQA-Diamond, LiveCodeBench and RULER to 64K, at 17.5 GiB with +14-19% prefill speed.
  - Mechanisms:
    - Gate projections compress ~11% GEMM error to ~2% output error through softplus/exp/sigmoid.
    - The delta-rule recurrence holds injected noise at a flat plateau over 32K tokens and forgets state impulses within hundreds of steps.
    - Per-token error washes out instead of compounding.
  - Community 4-bit quants had kept GDN at 8/16 bit out of caution.
- Quamba (Oct 2024; W8A8 for Mamba) and Quamba2 (Mar 2025; W4A8 / W4A16 for Mamba1/2) found SSM input/output activation outliers to be the main obstacle. Neither goes to 2 bits **[abstract, from prior knowledge; not re-fetched]**. — [Quamba](https://arxiv.org/abs/2410.13229), [Quamba2](https://arxiv.org/abs/2503.22879)
- LeapQuant (arXiv 2609.38166, title only) targets quantization of the recurrent *state* in linear attention, which matters for KV/state cache rather than weights **[title only, not read]**. — [listing](https://www.opentrain.ai/papers/leapquant-efficient-linear-attention-with-accurate-recurrent-state-quantization--arxiv-2609.38166/)

### Inferences
- Weight-only ternary on GDN layers should not suffer special long-context error accumulation, by the 2609.04098 mechanism. The team should still check long-context (RULER-style) degradation separately from short-context PPL, since this has not been shown at 1.6 bits.

### Gaps
- I found no study of weight quantization at <=2.5 bpw on any hybrid GDN or Mamba model. This is genuinely unexplored in the published literature I could find.

---

## 7. Unsloth Dynamic quants / imatrix practice for very-low-bit GGUFs: measured downstream quality

### Takeaway
I could not retrieve primary data this session: the Unsloth blog returned 403 and the Dynamic 2.0 docs page returned 404, pointing to a "Dynamic 3.0" page. The known practice is layer-selective mixed precision (sensitive layers at higher bits, the rest IQ1/IQ2) driven by an imatrix built from large chat-style calibration sets. It should be treated as a mixed-precision heuristic rather than a new quantizer.

### Cited Findings
- The Unsloth Dynamic 2.0 docs URL now redirects ([docs.unsloth.ai](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs) to [unsloth.ai/docs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs)), which returned 404 and mentions "Unsloth Dynamic 3.0 GGUFs". The full docs export is listed at [unsloth.ai/docs/llms-full.txt](https://unsloth.ai/docs/llms-full.txt) **[fetched]**.
- llama.cpp IQ1_S / IQ1_M / IQ2_XXS are the base types Unsloth mixes, and they rely on an importance matrix from calibration text **[from prior knowledge]**. — [llama.cpp quantize README](https://github.com/ggml-org/llama.cpp/tree/master/tools/quantize)

### Inferences
- The Unsloth-style lesson for Bonsai is item 5 above: choose which matrices get more bits using measured KL to the FP model, rather than applying a uniform type.

### Gaps
- No measured MMLU or KL numbers for Unsloth Dynamic 2.0/3.0 at IQ1/IQ2 were retrieved. To fill this, fetch the Unsloth Dynamic 3.0 docs page or llms-full.txt, or the DeepSeek-R1 1.58-bit dynamic blog (unsloth.ai/blog/deepseekr1-dynamic). That blog was not fetched this session, so none of its claims are included here.
- I found no community (r/LocalLLaMA) benchmark this session.

---

## 8. Synthesis: which ideas could close a ternary model's gap cheaply, and what to combine with QAT

### Takeaway
Ranked by expected value per unit of compute for a team already running QAT distillation:
1. Fix the QAT optimizer for ternary-specific pathologies (Tequila, PV-tuning discrete updates).
2. Use the cheap EfficientQAT-style schedule: block-wise distillation, then global tuning of scales only.
3. Do sensitivity-driven mixed precision (embeddings / lm_head / a few layers; possibly a second sparse trit-plane).
4. Revisit the 42-zero budget and the rotation block size.

Switching to a trellis/VQ representation (QTIP/EXL3) is the largest representational gain at 2 bpw, but it breaks the ternary multiplication-free runtime and needs new kernels.

### Cited Findings
- STE is sub-optimal for extreme-bit fine-tuning. PV adds ~0.3-0.4 PPL over STE and much more over continuous-only tuning **[fetched]**. — [PV-tuning](https://arxiv.org/html/2405.14852)
- Deadzone trapping is the dominant failure mode of ternary QAT; Tequila fixes it at near-zero inference cost (ICLR 2026) **[fetched]**. — [Tequila](https://arxiv.org/abs/2509.23809)
- Two-plane ternary PTQ keeps about 95-96% of reasoning accuracy on Qwen3-14B, in about an hour **[fetched]**. — [PTQTP](https://arxiv.org/html/2509.16989v3)
- Trellis (QTIP) beats E8 VQ (QuIP#) and additive codebooks (AQLM) at 2 bits, e.g. Llama-3-70B WikiText-2 4.97 vs 5.77 **[fetched]**. — [QTIP](https://arxiv.org/html/2406.11235)
- Layer-wise l2 error maps linearly to perplexity, which licenses DP-based bit allocation **[fetched]**. — [HIGGS](https://arxiv.org/abs/2411.17525)

### Inferences
- **The PTQ result is not wasted, but do not iterate on it further.** More calibration data or better sequencing gives nothing because one-shot GPTQ is capped by the scalar ternary codebook and the zero budget, not by Hessian estimation. The literature's 2-bit gains come from codebook dimension plus fine-tuning, not from GPTQ refinements.
- **Compare against the nats yardstick.** If the QAT run lands well below about +0.3-0.5 nat on a comparable eval, it is at or beyond the published 2-bit PTQ+FT frontier for 8B-class models (70B-class is about +0.6). Note that larger models lost *more* nats in QTIP, PV-tuning and PTQTP at 2 bits.
- **Cheap experiments, in priority order:**
  - (a) Fp16/Q6 embeddings and lm_head plus imatrix/KL-ranked higher precision for the top ~5% most sensitive matrices.
  - (b) A sweep of the zero budget (if the format allows variable zeros) or anneal it during QAT.
  - (c) A random sign flip plus a full-dimension (Kronecker) Hadamard instead of H512 blocks, or a permutation before the block rotation, measured by weight kurtosis and the one-shot residual.
  - (d) Tequila-style or PV-style discrete updates inside QAT.
  - (e) A sparse second trit-plane on the top-k sensitive matrices.
- **Learned rotations (SpinQuant/OSTQuant/FlatQuant) are probably not worth it.** The runtime is weight-only, and their published gains are for activation quantization. The exception is an all-integer W-ternary/A8 path.

### Gaps
- I found no published method that reaches FP-parity at ~1.6-1.75 bpw post-hoc on a 27B instruction or reasoning model. All near-parity ternary claims involve training (BitNet, Tequila) or report retention on easy benchmarks.
- I found no MLX or vLLM kernel support for QTIP-style trellis codes. vLLM has AQLM support historically; that was not verified this session.

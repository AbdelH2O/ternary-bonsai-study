# Extreme low-bit (ternary / 1-bit / 2-bit) QAT and distillation for LLMs: what recovers chat, reasoning and knowledge

Researched 2026-10-05. Scope: 2024 to Oct 2026, weighted toward 2025-2026. Some numbers below come from a summarizing fetch tool reading arXiv HTML pages, not from a manual table read; those are flagged "(as extracted)" where I had reason to doubt them. Team context: Qwen3.5-style 0.8B hybrid (Gated DeltaNet + full attention), ternary g128 with 42 zeros/group, Hadamard basis, 65.5M-token forward-KL QAT distillation. Result: 66% of the loss gap closed, MMLU at chance, GSM8K degenerate, copying an answer from the prompt at 27% vs 99% for FP.

## Q1. Method landscape: from-scratch vs. conversion, token budgets, losses, MMLU/GSM8K/chat results

### Takeaway
The only ternary models with solid MMLU/GSM8K/chat numbers were either trained natively for trillions of tokens (BitNet b1.58 2B4T: 4T tokens, MMLU 53.2, GSM8K 58.4) or converted from FP checkpoints with at least about 10B tokens, more often 30B to 400B (ParetoQ, HF/Falcon3, BitCPM4, BitDistill). Papers that use roughly 0.1B tokens report perplexity and zero-shot multiple-choice numbers only, never MMLU or GSM8K.

### Cited Findings

**Microsoft BitNet family**
- BitNet b1.58 (Feb 2024): absmean ternary weights {-1,0,+1} with 8-bit activations. It matches FP LLaMA perplexity and end-task scores from 3B parameters up when both are trained from scratch on the same data. — [arXiv 2402.17764](https://arxiv.org/abs/2402.17764)
- BitNet b1.58 2B4T (Apr 2025), native from-scratch training on 4T tokens:
  - Pretraining has two stages: high LR with cosine decay and WD peaking at 0.1, then a "cooldown" at much lower LR with zero WD on curated data plus synthetic math. — [arXiv 2504.12285](https://arxiv.org/html/2504.12285)
  - SFT sums the per-token CE instead of averaging it, and uses a larger LR and more epochs than FP models need. DPO follows (2 epochs, LR 2e-7, beta 0.1). — [arXiv 2504.12285](https://arxiv.org/html/2504.12285)
  - Embeddings and output layer stay at standard precision. The model uses squared ReLU and subln. — [arXiv 2504.12285](https://arxiv.org/html/2504.12285)
  - Scores: MMLU 53.17, GSM8K 58.38, IFEval 53.48, MT-Bench 5.85. That matches Qwen2.5-1.5B on average (55.23) and beats Qwen2.5-1.5B INT4 GPTQ on GSM8K (58.38 vs 50.57) while losing on MMLU (53.17 vs 57.43). — [arXiv 2504.12285](https://arxiv.org/html/2504.12285)
  - Converted 1-bit models in the same comparison score lower: Falcon3-1.58bit 7B averages 50.76 and Llama3-8B-1.58 (100B tokens) 49.75. — [arXiv 2504.12285](https://arxiv.org/html/2504.12285)
- BitNet a4.8 (Nov 2024): 4-bit activations on attention/FFN inputs plus sparsified 8-bit intermediate states; about 55% of parameters active; 3-bit KV cache. — [arXiv 2411.04965](https://arxiv.org/abs/2411.04965)
- BitNet v2 (Apr 2025): H-BitLinear applies an online Hadamard transform before activation quantization, which enables native 4-bit activations. — [arXiv 2504.18415](https://arxiv.org/abs/2504.18415)
- bitnet.cpp (Oct 2024): ternary CPU kernels, 1.37x to 5.07x speedup on ARM and 2.37x to 6.17x on x86. — [arXiv 2410.16144](https://arxiv.org/abs/2410.16144); TL kernels — [arXiv 2502.11880](https://arxiv.org/abs/2502.11880)
- BitNet Distillation, "BitDistill" (Oct 15 2025), converts FP Qwen3 0.6B/1.7B/4B, Qwen2.5-0.5B and Gemma3-1B to ternary for specific downstream tasks:
  - Stage 1 inserts SubLN before the output projections of attention and FFN.
  - Stage 2 is 10B tokens of continued pretraining on the FALCON corpus.
  - Stage 3 combines logit KL (temperature 5) with MiniLM multi-head attention-relation distillation on a single late layer.
  - Results at 4B: MNLI 91.40 vs FP 91.48, against 76.11 for plain ternary SFT. — [arXiv 2510.13998](https://arxiv.org/html/2510.13998)
  - Ablation on Qwen3-0.6B MNLI: full pipeline 88.17; without SubLN 76.30; without continued pretraining 86.73; without distillation 86.73. — [arXiv 2510.13998](https://arxiv.org/html/2510.13998)
  - Without stage 2, the gap to FP grows with model size (13.9% at 0.6B, 15.3% at 4B). Later layers are the best attention-distillation targets, better than early layers or all layers. A larger FP teacher (1.7B or 4B) lets the 0.6B student beat the same-size FP baseline. — [arXiv 2510.13998](https://arxiv.org/html/2510.13998)
  - Evaluations are GLUE and CNN/DM only; no MMLU, GSM8K or chat numbers. — [arXiv 2510.13998](https://arxiv.org/html/2510.13998)

**Meta ParetoQ (Feb 2025; NeurIPS 2025)**
- One framework covers 1, 1.58, 2, 3 and 4 bits. With a fixed 100B-token budget, about 90% FP pretraining followed by 10% QAT beats both PTQ and from-scratch QAT. — [arXiv 2502.02631](https://arxiv.org/html/2502.02631)
- QAT saturates around 10B tokens at 3 and 4 bits but needs about 30B tokens at 1, 1.58 and 2 bits. — [arXiv 2502.02631](https://arxiv.org/html/2502.02631)
- At 3 bits and above, weights move 10-20% ("compensation"). At 2 bits and below they move about 40% ("reconstruction"), meaning representations change drastically. — [arXiv 2502.02631](https://arxiv.org/html/2502.02631)
- Quantizers: SEQ (Stretched Elastic Quant) for 1.58 and 2 bits, LSQ for 3 and 4 bits. Training uses AdamW with zero WD and LR 2e-5 for ternary, 1e-5 for 3 and 4 bits, with cosine decay. — [arXiv 2502.02631](https://arxiv.org/html/2502.02631)
- Zero-shot averages: ternary MobileLLM-600M 55.5 (beats the previous SoTA 3B ternary model); LLaMA-3 3B 61.9 ternary vs 65.4 at 4-bit; LLaMA-3 8B 69.0 ternary. — [arXiv 2502.02631](https://arxiv.org/html/2502.02631)

**Spectra / TriLM**
- Spectra 1.1 (Jun 2025) trains TriLMs of 1.5B, 2.5B and 3.6B on about 1.2T tokens. Its scaling law is L ≈ 2.19 + 4.73/N^0.32 + 5.18/D^0.81: data helps far more than parameters. — [arXiv 2506.23025](https://arxiv.org/html/2506.23025v1)
- Knowledge-heavy tasks (MMLU, TriviaQA) show larger ternary deficits than commonsense tasks. Spectra-1.1-3B scores about 42-43 on MMLU (as extracted). — [arXiv 2506.23025](https://arxiv.org/html/2506.23025v1)
- Original Spectra/TriLM suite, Jul 2024; ICLR 2025. — [arXiv 2407.12327](https://arxiv.org/abs/2407.12327), [ICLR 2025](https://iclr.cc/virtual/2025/33505)

**Hugging Face, Falcon3 and Falcon-Edge (TII)**
- HF blog (Sep 2024): converted Llama3-8B to 1.58-bit with 10B and 100B tokens.
  - Quantizing directly with no warmup sent the loss from about 2 to about 13.
  - The fix was a "warmup quantization" λ schedule that mixes the FP and quantized weights. A linear schedule λ = min(2·step/total, 1) worked best. Exponential schedules gave about 15 perplexity; sigmoid was unstable.
  - LR 1e-4 was best, and LR choice was described as crucial.
  - Training from scratch reached WikiText perplexity 26; the fine-tuned conversion reached about 12.2.
  - The 10B-token conversion roughly matched Llama-1-7B on MMLU.
  - For SmolLM-135M, warmup barely helped. — [HF blog](https://huggingface.co/blog/1_58_llm_extreme_quantization)
- Falcon3-1.58bit (Dec 2024) was converted from Falcon3 using the HF-blog and BitNet recipe (details in the Falcon3 tech report, "Compression" section). — [Falcon3-7B-Instruct-1.58bit card](https://huggingface.co/tiiuae/Falcon3-7B-Instruct-1.58bit/blob/main/README.md). Its 7B average is 50.76, below native BitNet 2B4T (60.68 on the 1-bit comparison set). — [arXiv 2504.12285](https://arxiv.org/html/2504.12285)
- Falcon-Edge (May 2025): 1B and 3B ternary models trained from scratch on about 1.5T tokens with WSD scheduling. TII ships bf16, ternary and "pre-quantized" checkpoints and the `onebitllms` library for fine-tuning. — [HF blog](https://huggingface.co/blog/tiiuae/falcon-edge); Axolotl ternary fine-tuning — [HF blog](https://huggingface.co/blog/axolotl-ai-co/finetuning-ternary-llms-tii-axolotl)

**OpenBMB BitCPM4 (Jun 2025)**
- Ternary 0.5B and 1B models built by two-stage continual QAT from a high-precision MiniCPM checkpoint (first FP8 training, then ternary QAT).
- About twice the LR-decay-phase tokens of QAT was enough.
- Total cost was about 10% of BitNet-2B's tokens (that is, on the order of 400B if measured against 4T), and the models were "competitive especially on knowledge tasks". — [arXiv 2506.07900](https://arxiv.org/pdf/2506.07900), [BitCPM4-1B card](https://huggingface.co/openbmb/BitCPM4-1B/blob/refs%2Fpr%2F1/README.md)

**Tencent Tequila (Sep/Oct 2025; ICLR 2026)**
- Weights trapped in the ternary deadzone get noisy, uninformative STE gradients. Tequila reactivates them as dynamic biases at nearly zero inference cost.
- Setup: LLaMA-3.2-1B/3B, 10B UltraFineWeb tokens, LR 1e-4. Results are more than 4% better than SoTA on ARC and within 1% of BF16.
- Only PIQA, ARC, HellaSwag and WinoGrande are reported; no MMLU or GSM8K.
- Its baseline table lists token budgets: BitNet 100B, Spectra 100B, ParetoQ 10B, TernaryLLM 10B. — [arXiv 2509.23809](https://arxiv.org/html/2509.23809v2), [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5b555804d495321df2e3208cc27f4fbc-Abstract-Conference.html)

**Earlier conversion methods (2023-2024), all FP-teacher distillation**
- LLM-QAT (2023): data-free distillation on sequences generated by the FP model itself; works down to 4-bit. — [arXiv 2305.17888](https://arxiv.org/abs/2305.17888)
- OneBit (NeurIPS 2024): 1-bit weights plus two FP value vectors, initialized by SVID matrix decomposition and trained with KD. Retains "at least 81%" of FP performance on LLaMA. — [arXiv 2402.11295](https://arxiv.org/abs/2402.11295)
- BitDistiller (Feb 2024): asymmetric quantization and clipping plus Confidence-Aware KL (CAKLD), which blends forward and reverse KL. It self-distills on teacher-generated data at 2 and 3 bits. — [arXiv 2402.10631](https://arxiv.org/abs/2402.10631)
- DB-LLM (Feb 2024): Flexible Dual Binarization plus Deviation-Aware Distillation, which up-weights the uncertain samples where teacher and student disagree, at 2 bits. — [arXiv 2402.11960](https://arxiv.org/abs/2402.11960)
- TernaryLLM (Jun 2024): Dual Learnable Ternarization (learnable scale and shift) plus Outlier-Friendly Feature distillation (cosine similarity on hidden features). — [arXiv 2406.07177](https://arxiv.org/abs/2406.07177)
- EfficientQAT (Jul 2024; ACL 2025): block-wise training of all parameters (Block-AP), then end-to-end training of only the quantization parameters (E2E-QP). 2-bit Llama-2-70B takes 41 h on one A100 and scores 69.48 vs FP 72.41. — [arXiv 2407.11062](https://arxiv.org/abs/2407.11062)

**Newer 2025-2026 work**
- UPQ (Jun 2025): FP16 to INT4 by block-wise PTQ, then INT2 by distillation-QAT with generalized JSD. Billed as the first 2-bit instruction-tuned open LLMs without proprietary post-training data, competitive on MMLU and IFEval. — [arXiv 2506.09104](https://arxiv.org/abs/2506.09104)
- "1-Bit Wonder" (Feb 2026): k-means (non-uniform) weight formats beat integer formats in low-bit QAT. At equal memory, a 31B 1-bit model is reported at MMLU 51.6 and GSM8K 45.3, against 50.9 and 48.5 for a 12B 4-bit model. Caution: the 16-bit baseline extracted as "44B, MMLU 33" looks garbled; verify in the paper. The paper also argues that loss-only evaluations hide downstream differences. — [arXiv 2602.15563](https://arxiv.org/html/2602.15563v1)
- PTQTP (Sep 2025): post-training quantization to trit-planes on LLaMA3.x and Qwen3 from 0.6B to 70B. It is reported to approach 1.58-bit QAT in about an hour, against 10-14 GPU-days for QAT. — [arXiv 2509.16989](https://arxiv.org/abs/2509.16989v1)
- Prism ML Bonsai 27B / Bonsai 2: per PrismML docs, based on Qwen3.6-27B, with 1-bit or ternary weights across embeddings, attention, MLPs and LM head; the vision tower is 4-bit. — [PrismML docs](https://docs.prismml.com/models/bonsai-27b). A third-party writeup calls it post-training (not from scratch) and says the exact recipe is "the program's least-disclosed component". — [local-ai-zone blog (secondary)](https://local-ai-zone.github.io/blog/bonsai-2-27b-ternary-quantization-deep-dive.html)

### Inferences
- Every ternary result with strong MMLU or GSM8K (BitNet 2B4T, Falcon-Edge, BitCPM4, the HF Llama3-8B conversion) used at least 10B tokens, and the ones that also recover reasoning and chat used SFT/DPO-style post-training as well. Distillation-only papers at 0.1B tokens (Ternary Mamba) report only perplexity and zero-shot MC.
- Prism quantizes embeddings and lm_head too, which almost every academic recipe keeps in FP. The team's student may legitimately differ from Bonsai there.

### Gaps
- I could not find "PT-BitNet" as a distinct paper; searches returned PTQTP, TWLA and BitNet results. Its details are unverified.
- Prism ML's training recipe (token count, losses) is not public.
- The Falcon3-1.58bit token count is in the Falcon3 tech report, which I did not fetch.

## Q2. Converting a pretrained FP model to ternary: how many tokens? Is 65.5M too few?

### Takeaway
Yes. 65.5M tokens is 2-3 orders of magnitude below every published ternary conversion that recovers downstream ability (10B to 30B tokens is the floor; ~100B to 400B for strong general models). Even at 4-bit, NVFP4 QAD on post-trained models used 0.3B to 6B tokens. ParetoQ's mechanistic finding, that 2 bits and below means "reconstruction" with about 40% of weights moving, explains why loss can partly close fast while capabilities do not.

### Cited Findings
- **Ternary needs about 30B QAT tokens to saturate** (3 and 4 bits: about 10B), and ~90% FP / 10% QAT of a 100B budget is optimal. — [ParetoQ, arXiv 2502.02631](https://arxiv.org/html/2502.02631)
- **10B tokens of continued pretraining** is BitDistill's stage 2. Removing it costs about 1.4 points on MNLI at 0.6B, and the gap without it grows with model size. — [arXiv 2510.13998](https://arxiv.org/html/2510.13998)
- **10B and 100B tokens** were used for the HF Llama3-8B conversions. At 10B the model only reached Llama-1-7B MMLU level. — [HF blog](https://huggingface.co/blog/1_58_llm_extreme_quantization)
- **10B tokens** is the budget for Tequila, TernaryLLM and ParetoQ in Tequila's comparison; BitNet and Spectra used 100B. — [arXiv 2509.23809](https://arxiv.org/html/2509.23809v2)
- **About 400B tokens** for BitCPM4: about 10% of BitNet-2B's 4T, and QAT about 2x the LR-decay phase. — [arXiv 2506.07900](https://arxiv.org/pdf/2506.07900)
- **102M tokens** (the closest budget to the team's) was used for ternary Mamba-2 1.3B with KD.
  - Results: 48.1% zero-shot average vs FP16 55.5%, +4.2 WikiText PPL.
  - CE-only training gave PPL 87.6, KD-only 38.0, and a 0.5/0.5 KD+CE mix was best.
  - No MMLU, retrieval or copying evaluations. — [Ternary Mamba, arXiv 2606.18114 (Jun 2026)](https://arxiv.org/html/2606.18114v1)
- **NVFP4 QAD (4-bit, much easier than ternary)** needed about 0.3B tokens (Llama-Nemotron-Super 49B), 0.8B (AceReason 7B), 2.5B (Nemotron-3-Nano 30B-A3B) and about 6B (Nemotron-Nano-9B-v2, a hybrid Mamba). — [NVIDIA QAD report, Mar 2026](https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf)
- **Optimal QAT fraction grows with compute.** The loss-optimal QAT/FP split rises with total tokens per parameter-byte, and fusing LR cooldown with QAT saves compute (Apple, Sep 2025; ICLR 2026). — [arXiv 2509.22935](https://arxiv.org/abs/2509.22935v1), [Apple ML](https://machinelearning.apple.com/research/compute-optimal)
- **Heavily trained models degrade more under low-bit quantization.** Quantization-induced degradation is larger for smaller models and for models trained on more tokens. — [Low-Bit Quantization Favors Undertrained LLMs, arXiv 2411.17691](https://arxiv.org/abs/2411.17691); PTQ degradation grows with pretraining data — [Scaling Laws for Precision, arXiv 2411.04330](https://arxiv.org/abs/2411.04330)
- **Switching from 16-bit to 1.58-bit mid-training** beats full 1.58-bit training on OLMo-1B (loss 3.088 vs 3.15; FP 2.95). Keeping the optimizer state and phasing in λ reduce the loss spike, but further training compensates either way. — [Nielsen et al., ACL Findings 2025, arXiv 2502.11895](https://arxiv.org/html/2502.11895v1)
- **Small models**: for SmolLM-135M, warmup quantization barely helped, and the loss curves matched immediate quantization. — [HF blog](https://huggingface.co/blog/1_58_llm_extreme_quantization)

### Inferences
- The team's 0.8B Qwen3.5-class teacher was likely trained on tens of trillions of tokens (Qwen3 used ~36T: [Qwen3 report, arXiv 2505.09388](https://arxiv.org/abs/2505.09388)). That is the very-high-tokens-per-parameter regime where QiD is worst and the compute-optimal QAT fraction is largest. Expect a 0.8B ternary student to need at least 10B tokens, plausibly 30B or more, before MMLU and GSM8K leave chance or degenerate output.
- "66% of the loss gap closed" is consistent with the known pattern: early QAT closes average CE quickly (frequent tokens, local syntax), while capabilities that hinge on low-frequency tokens and precise attention patterns (knowledge, copying, answer-letter read-out) come back last.
- Budget for at least 2-3 orders of magnitude more tokens before concluding that the architecture or recipe fails. A sensible ladder is 0.5B, 2B, then 10B, tracking MMLU, GSM8K, the copy probe and teacher-KL on held-out chat at each step.

### Gaps
- I found no paper reporting MMLU or GSM8K as a function of QAT tokens for a sub-1B ternary conversion. The token-to-capability curve at 0.8B is extrapolated, not measured.

## Q3. Distillation losses and data: what restores instruction following and reasoning?

### Takeaway
2025-2026 evidence converges on four points:
1. Pure KL to the FP teacher (QAD) beats CE/SFT-style QAT for post-trained chat and reasoning models, and it is robust to the data source.
2. Training only on fixed corpora leaves exposure bias: quantized students fall into repetition loops, which looks like "GSM8K degenerate". On-policy distillation (student rollouts scored with reverse KL from the teacher) fixes this.
3. A good PTQ initialization (GPTQ) matters.
4. Attention or feature-level distillation on a late layer adds measurable gains for ternary.

### Cited Findings
- **QAD vs QAT (NVIDIA, Mar 2026)**
  - QAT reaches nearly the same CE as BF16 but a large KL to the teacher, so it "effectively acts as an additional post-training stage". QAD reaches near-zero KL.
  - For RL-trained models, QAT on SFT data "can break the capabilities learned during RL training", while QAD recovers near-BF16 accuracy.
  - Settings: T = 1, LR 1e-5 to 1e-6, at or below the original post-training LR. — [NVIDIA QAD report](https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf)
- **QAD is robust to the data source.** On AceReason-7B, these all gave near-BF16 results: original SFT data, BF16 generations from RL prompts, and BOS-seeded generations. Keeping incorrect generations beat filtering to correct-only. Even random tokens did not break the model. Math-only or code-only data recovered both domains (AIME24 71.0-71.7 vs BF16 73.0 and PTQ 69.4). — [NVIDIA QAD report](https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf)
- **Reasoning QAT systematic study (Huawei/Tsinghua, Jan 2026)**
  - KD beats SFT as the QAT objective for both SFT- and RL-trained reasoning models (3-bit drops of 8.1-9.3% vs 10.5-21.4%).
  - GPTQ initialization beats RTN, converges faster and is more stable.
  - RL (GRPO) works only after a KD cold start; direct RL on heavily quantized models collapses.
  - Aligning the PTQ calibration domain with the QAT domain helps.
  - 2-bit W2G128 MATH-500: Qwen3-4B 78.3 vs 4.8 for GPTQ alone; R1-Qwen-1.5B 55.0 vs 3.7. — [arXiv 2601.14888](https://arxiv.org/html/2601.14888)
- **On-policy distillation for low-bit reasoning (Sep 2026)**
  - Quantization perturbs every next-token distribution, so deviations compound (exposure bias). Quantized models fall into repetitive loops and exhaust decode budgets: about 95% hit the limit on MATH-500 vs about 27% for BF16 (as extracted).
  - Fix: QAD initialization, then on-policy distillation on student rollouts at deployment precision, using reverse KL (estimable from student samples, penalizing continuations the teacher dislikes) plus verifier rewards.
  - At 2.79 and 1.88 effective bits on Qwen3 0.6B/1.7B/4B and Falcon3-1B: MATH-500 BF16 retention rises from 35% to 70% and HumanEval from 66% to 91%. Search snippets also mention GSM8K retention rising from 62.7% to 85.0%. — [arXiv 2609.26708](https://arxiv.org/html/2609.26708)
- **Foundations of on-policy and reverse-KL distillation**: GKD trains on student-generated sequences against teacher token distributions with generalized JSD or reverse KL. — [GKD, arXiv 2306.13649](https://arxiv.org/abs/2306.13649); reverse-KL MiniLLM — [arXiv 2306.08543](https://arxiv.org/abs/2306.08543). BitDistiller's CAKLD blends forward and reverse KL on teacher-generated data. — [arXiv 2402.10631](https://arxiv.org/abs/2402.10631)
- **Generalized JSD distillation** produced 2-bit instruction-tuned models with competitive MMLU and IFEval (UPQ). — [arXiv 2506.09104](https://arxiv.org/abs/2506.09104)
- **Attention-relation distillation**: MiniLM-style distillation on one late layer plus logit KL is BitDistill's stage 3 (about +1.4 MNLI at 0.6B). Later layers beat early or all-layer distillation. — [arXiv 2510.13998](https://arxiv.org/html/2510.13998)
- **Feature distillation**: TernaryLLM's OFF uses cosine similarity on hidden states, chosen to be robust to outliers. — [arXiv 2406.07177](https://arxiv.org/abs/2406.07177)
- **Loss mix at tiny budgets**: KD-only beat CE-only by a wide margin for ternary Mamba (PPL 38 vs 88), and KD+CE 0.5/0.5 was best. — [arXiv 2606.18114](https://arxiv.org/html/2606.18114v1)
- **Self-generated data**: LLM-QAT's data-free KD on the FP model's own generations works without original data. — [arXiv 2305.17888](https://arxiv.org/abs/2305.17888)
- **Native ternary chat** (BitNet 2B4T) needed SFT with summed (not averaged) CE, higher LR, more epochs, then DPO. — [arXiv 2504.12285](https://arxiv.org/html/2504.12285)

### Inferences
- The team's "GSM8K degenerate" output fits the exposure-bias and repetition failure documented in 2609.26708. All-position forward KL on fixed text (teacher forcing) never trains the student on its own prefixes. Adding on-policy reverse-KL (or JSD) on student rollouts of chat and math prompts, after a teacher-forced warm-up, is the most directly evidenced fix.
- Forward KL is mode-covering: it spreads a weak student's mass over all teacher modes. For read-out tasks such as an MMLU letter or a copied answer, reverse KL or a temperature-1 KL concentrated on response tokens may help more.
- Teacher-generated responses to chat and MC prompts (sequence-level KD) are cheap and, per the QAD report, about as effective as the original SFT data. Include incorrect samples.
- A late-layer attention-relation loss is cheap and directly targets the attention patterns behind copying. It belongs on the hybrid's full-attention layers (see Q4 and Q5).

### Gaps
- No paper compares forward vs reverse KL specifically for ternary conversion of a chat model. The evidence comes from 2-3 bit, NVFP4 or task-specific ternary work.

## Q4. Known failure modes: knowledge/MMLU loss, in-context copying, which components need higher precision or training

### Takeaway
Two patterns are documented. Knowledge-heavy benchmarks degrade more than perplexity or commonsense under low-bit quantization, and in-context and long-context retrieval is among the first abilities lost. Recipes almost universally keep embeddings, lm_head, norms and SSM dynamics parameters in FP (and trainable), and production hybrids keep their few attention layers and the first and last layers at higher precision. Ternary-specific stability needs extra norms (SubLN) before output projections.

### Cited Findings
- **Knowledge**
  - Ternary models lag more on MMLU and TriviaQA than on commonsense tasks. — [Spectra 1.1, arXiv 2506.23025](https://arxiv.org/html/2506.23025v1)
  - Factual recall drops consistently at 4-bit and is preserved at 8-bit. Relations where FP performance is not saturated degrade most. — [Through a Compressed Lens, arXiv 2505.13963](https://arxiv.org/html/2505.13963v3)
- **Loss vs downstream mismatch**
  - QAT can match BF16 cross-entropy while diverging strongly from the teacher's output distribution. — [NVIDIA QAD report](https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf)
  - Prior aggressive-quantization claims leaned on train and validation loss. — [1-Bit Wonder, arXiv 2602.15563](https://arxiv.org/html/2602.15563v1)
- **Long-context retrieval** is fragile even at 4-bit. 8-bit costs about 0.8% on average, while 4-bit loses up to 59% on long-context tasks and up to 16% on retrieval. — [Does quantization affect long-context tasks?, EMNLP 2025, arXiv 2505.20276](https://arxiv.org/html/2505.20276v3)
- **Induction heads** implement in-context copying and prefix matching. — [Olsson et al., arXiv 2209.11895](https://arxiv.org/abs/2209.11895); ablating them sharply cuts in-context learning in Llama-3-8B and InternLM2-20B — [arXiv 2407.07011](https://arxiv.org/html/2407.07011v3)
- **Reasoning degeneration**: quantized reasoning models loop and hit decode limits. — [arXiv 2609.26708](https://arxiv.org/html/2609.26708); math reasoning degrades under low-bit — [Quantization Meets Reasoning, arXiv 2501.03035](https://arxiv.org/abs/2501.03035)
- **What stays FP in the published recipes**
  - BitNet 2B4T: embeddings and output layer FP. — [arXiv 2504.12285](https://arxiv.org/html/2504.12285)
  - ParetoQ: embedding and output layers excluded from low-bit. — [arXiv 2502.02631](https://arxiv.org/html/2502.02631)
  - Ternary Mamba: SSM dynamics (A_log, D, dt_bias; under 0.3% of parameters), conv1d, embedding, lm_head and norms in FP16; only in_proj and out_proj ternary (85.7% of parameters). — [arXiv 2606.18114](https://arxiv.org/html/2606.18114v1)
- **What stays higher precision in production hybrids (NVIDIA)**
  - Nemotron-Nano-9B-v2 (4 attention + 52 Mamba layers): attention layers plus the first and last two layers kept at BF16.
  - Nemotron-3-Nano: its 6 attention layers and the Mamba layer before each kept at BF16. — [NVIDIA QAD report](https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf)
- **Norm placement**: SubLN before the attention-output and FFN-output projections is BitDistill's largest single ablation effect (88.17 with it, 76.30 without). BitNet 2B4T uses subln as well. — [arXiv 2510.13998](https://arxiv.org/html/2510.13998), [arXiv 2504.12285](https://arxiv.org/html/2504.12285)
- **Counterexample**: Prism's Bonsai quantizes embeddings and LM head as well. — [PrismML docs](https://docs.prismml.com/models/bonsai-27b)

### Inferences
- Copying an answer stated in the prompt at 27% (FP 99%) is a direct failure of induction or retrieval circuitry. In a Qwen3.5-style hybrid, exact retrieval sits mostly in the sparse full-attention layers (GDN state is compressed and lossy). Ternarizing their q/k/v/o projections is the likely root cause. Supporting evidence: NVIDIA keeps attention layers in BF16 even at 4 bits, attention-relation distillation helps ternary, and retrieval is fragile even at 4 bits.
- **Test 1**: keep the full-attention layers' projections (and perhaps the first and last layers) at FP or 4-bit and check whether copying and MMLU recover. That isolates whether the bottleneck is the ternary attention path or the token budget.
- **Test 2**: train all non-projection FP tensors (norms, q/k-norm, GDN gates, A_log, dt_bias, conv1d), consistent with every recipe above. Add SubLN-style norms before output projections if the architecture lacks them.
- MMLU at chance (rather than merely low) plus failed copying suggests broken answer-format read-out (letter-token selection) as well as lost knowledge. MC read-out distillation data will mainly help format. Recovering knowledge needs tokens; Spectra and Bonsai-style evidence says ternary keeps less knowledge per parameter even when fully trained.

### Gaps
- I found no study that measures induction-head or copy accuracy specifically in ternary or 1-bit LLMs. The link is inferred from 4-bit retrieval studies and mechanistic work on FP models.

## Q5. Low-bit QAT for linear-attention / hybrid / SSM architectures (Mamba, Gated DeltaNet, Qwen3-Next/3.5)

### Takeaway
Direct evidence is thin. Ternary Mamba-2 (102M-token KD) and Bi-Mamba (105B tokens from scratch) reach comparable zero-shot scores. Gated DeltaNet is unusually robust to 4-bit quantization (gates and the delta-rule recurrence damp noise). No published ternary QAT of Gated DeltaNet or Qwen3-Next/3.5 hybrids reports MMLU, GSM8K or retrieval.

### Cited Findings
- **Ternary Mamba (Jun 2026)**: Mamba-2 1.3B, g128 per-group absmean-style ternary on in_proj and out_proj; 102M tokens; KD (T = 1, α = 0.5 KL + 0.5 CE); LR 2.5e-4 cosine.
  - Results: 48.1% 7-task zero-shot average vs FP 55.5%; group size 64, 128 and 256 make no difference.
  - Bi-Mamba reaches 48.4% after 105B from-scratch tokens. — [arXiv 2606.18114](https://arxiv.org/html/2606.18114v1)
- **Earlier ternary SSM work**: Slender-Mamba (Fully Quantized Mamba in 1.58 bits, COLING 2025) is a smaller ternary Mamba. — [ACL Anthology 2025.coling-main.316](https://aclanthology.org/2025.coling-main.316.pdf)
- **Gated DeltaNet under NVFP4 W4A4 (Sep 2026)**, on a hybrid 27B with 48 GDN and 16 attention layers (reported as a Qwen3.x model):
  - Gate projections are robust: softplus/exponential and sigmoid parameterizations turn about 11% GEMM error into about 2% output error.
  - The delta-rule recurrence holds injected noise at a flat plateau over 32K tokens and forgets state impulses within hundreds of steps, so error does not compound with context.
  - The paper does not address sub-4-bit quantization. — [arXiv 2609.04098](https://arxiv.org/abs/2609.04098)
- **Hybrid Mamba-Transformer QAD at 4 bits** keeps the attention layers (and neighbouring layers) at BF16. — [NVIDIA QAD report](https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf)

### Inferences
- In hybrids, the recurrent GDN path may tolerate ternary better than the attention path. That is the reverse of the intuition that attention is the "easy" part. The 3:1 GDN:attention ratio means a few full-attention layers carry all exact-recall duty, so they deserve the precision budget (mixed precision) or dedicated distillation (attention-relation loss on those layers).
- GDN gate projections (alpha/beta, output gate) map through squashing functions that damp quantization error. Ternarizing them is probably fine, but their FP biases and norms should be trained.

### Gaps
- No public ternary or 2-bit QAT results for Gated DeltaNet, Qwen3-Next or Qwen3.5 hybrids with MMLU, GSM8K or retrieval metrics. Prism's Bonsai is the main existence proof, and its recipe is undisclosed.

## Q6. Optimizer / schedule tricks: STE variants, LR, gradual quantization, latent init, zeros (deadzone) handling

### Takeaway
1. Initialize from a good PTQ solution (GPTQ) rather than RTN.
2. Use a relatively high LR for ternary (1e-4 for the HF/Tequila 8B/1-3B conversions; ParetoQ used 2e-5 at small batch) with zero or low weight decay and a cooldown.
3. Phase quantization in (λ warmup) or keep the optimizer state to soften the loss spike, though these effects wash out with enough tokens.
4. Handle deadzone-trapped weights explicitly (Tequila), and pick a balanced ternary grid (ParetoQ's SEQ).

### Cited Findings
- **Gradual quantization**: linear λ warmup (mixing FP and ternary weights) avoids the 2 to 13 loss spike, and LR 1e-4 was best. — [HF blog](https://huggingface.co/blog/1_58_llm_extreme_quantization). λ phase-in and optimizer-state retention reduce spikes, but further training compensates. — [arXiv 2502.11895](https://arxiv.org/html/2502.11895v1)
- **Quantizer choice**: SEQ for ternary/2-bit, LSQ for 3/4-bit. AdamW with WD 0, LR 2e-5 for ternary, cosine schedule. — [ParetoQ](https://arxiv.org/html/2502.02631)
- **Deadzone trapping**: STE gives weights near the ternary threshold noisy gradients, so they get stuck at zero. Tequila reactivates them as dynamic biases: more than 4% better on ARC, within 1% of FP at 10B tokens. — [arXiv 2509.23809](https://arxiv.org/html/2509.23809v2)
- **Initialization**: GPTQ-initialized QAT converges faster and is more stable than RTN. — [arXiv 2601.14888](https://arxiv.org/html/2601.14888). OneBit's SVID initialization speeds convergence for 1-bit. — [arXiv 2402.11295](https://arxiv.org/abs/2402.11295). UPQ goes FP16 to 4-bit by PTQ before 2-bit distillation-QAT. — [arXiv 2506.09104](https://arxiv.org/abs/2506.09104)
- **Schedules**: native BitNet uses two-stage LR and WD, then cooldown at low LR and zero WD. — [arXiv 2504.12285](https://arxiv.org/html/2504.12285). Fusing cooldown with QAT saves compute. — [arXiv 2509.22935](https://arxiv.org/abs/2509.22935v1). Spectra 1.1 drops weight decay and clips gradients at 1.0. — [arXiv 2506.23025](https://arxiv.org/html/2506.23025v1)
- **Hadamard / rotation in QAT**: BitNet v2 applies an online Hadamard transform before activation quantization. — [arXiv 2504.18415](https://arxiv.org/abs/2504.18415). QuEST combines Hadamard normalization with a "trust" gradient estimator for stable low-bit QAT. — [arXiv 2502.05003](https://arxiv.org/abs/2502.05003)
- **Grid shape**: non-uniform (k-means) low-bit grids beat integer grids in QAT. — [arXiv 2602.15563](https://arxiv.org/html/2602.15563v1)
- **Group size**: no measurable effect across g = 64, 128 and 256 for ternary Mamba. — [arXiv 2606.18114](https://arxiv.org/html/2606.18114v1)
- **Learning rates for QAD on post-trained models**: 1e-5 to 1e-6, at or below the original post-training LR. Higher LR destabilized SFT-heavy models. — [NVIDIA QAD report](https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf). Ternary needs much larger weight movement (about 40% of weights change). — [ParetoQ](https://arxiv.org/html/2502.02631)

### Inferences
- A fixed 42-of-128 zeros constraint is close to the roughly one-third zeros of absmean ternary, but enforcing an exact per-group count makes the STE deadzone problem worse (hard top-k boundaries). Consider Tequila-style reactivation, or let the zero count float during QAT and re-impose it at the end.
- There is a tension between the conservative QAD learning rates (designed for 4-bit "compensation") and the higher ternary learning rates. ParetoQ's reconstruction finding argues for the ternary-style higher LR (about 1e-4 at large batch) on latent weights, with a lower LR on FP norms and gates.

### Gaps
- No direct comparison of STE variants (vanilla, clipped, trust or QuEST-style, Tequila) for ternary conversion of a chat model on MMLU or GSM8K.

## Q7. Synthesis for the team's specific failure modes: what to try next

### Takeaway
The literature points to three ranked causes:
1. Far too few tokens: 65.5M vs ≥10B.
2. A training distribution that never exposes the student to its own outputs: teacher-forced forward KL only, which explains degenerate generation.
3. Ternarized full-attention projections in a hybrid whose few attention layers carry all exact retrieval, which explains the 27% copy rate and the chance-level MC read-out.

### Cited Findings
- Ternary conversions need about 10-30B tokens to saturate. — [ParetoQ](https://arxiv.org/html/2502.02631), [BitDistill](https://arxiv.org/html/2510.13998), [Tequila](https://arxiv.org/html/2509.23809v2)
- Teacher-forced training leaves exposure bias and looping. On-policy reverse-KL distillation roughly doubles MATH-500 retention. — [arXiv 2609.26708](https://arxiv.org/html/2609.26708)
- KL-only distillation (QAD) beats CE-style QAT on chat and reasoning, works with teacher-generated or narrow-domain data, and transfers across domains. — [NVIDIA QAD report](https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf)
- Production hybrids keep attention layers at higher precision. — [NVIDIA QAD report](https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf). Late-layer attention-relation distillation helps ternary. — [BitDistill](https://arxiv.org/html/2510.13998)
- Recipes keep norms and SSM dynamics in FP and trainable, and SubLN is critical for ternary stability. — [arXiv 2606.18114](https://arxiv.org/html/2606.18114v1), [arXiv 2510.13998](https://arxiv.org/html/2510.13998)

### Inferences (ranked experiment list)
1. **Diagnostic ablation (cheap)**: same 65.5M budget, but keep the full-attention layers' q/k/v/o (optionally plus the first and last blocks) in BF16 or 4-bit. If copying jumps toward 99% and MMLU leaves chance, the attention path is the bottleneck. Then either budget precision there (mixed-precision ternary) or add attention-relation distillation (MiniLM loss on attention maps or value relations of the last few full-attention layers) and push further tokens.
2. **Train all non-projection tensors** (RMSNorm, q/k-norm, GDN gates, A_log, dt_bias, conv1d), as all recipes do. Add SubLN before o_proj and down_proj if the hybrid lacks a norm there (BitDistill's biggest ablation effect).
3. **Scale tokens to 2-10B**, with general continued-pretraining text (BitDistill stage 2) mixed with teacher-generated chat, math and MC responses. Use KL at T = 1 (QAD) or a forward/reverse blend (CAKLD/JSD). Track teacher-KL on held-out chat, not just CE.
4. **Add an on-policy phase**: student rollouts on chat and GSM8K prompts with reverse-KL to the teacher. This is the most direct fix for degenerate GSM8K generation. Optionally add verifier reward later (RL only after a KD cold start).
5. **Initialize from the team's best GPTQ/Hadamard PTQ solution**, not RTN. Use ternary-scale LR (~1e-4 at large batch) on latent weights with WD 0 and a cooldown. Treat deadzone trapping explicitly (Tequila-style reactivation, or a floating zero count during QAT).
6. **Expectations**: even native ternary models trained on trillions of tokens lose more on knowledge (MMLU) than on commonsense, and a 0.8B model is in the regime where low-bit degradation is worst (QiD; HF SmolLM). A realistic target is MMLU above chance and non-degenerate GSM8K, not FP parity.

### Gaps
- No published experiment pins how much of a ternary hybrid's copy failure comes from attention-layer precision vs token budget. Experiment 1 is designed to answer that locally.

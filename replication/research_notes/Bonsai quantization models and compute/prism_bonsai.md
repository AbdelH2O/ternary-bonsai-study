# Prism ML's Bonsai model family: published method, claims, lineage, and independent evidence (as of 2026-10-05)

Labels used throughout: **[PRISM]** = stated by Prism ML (whitepaper, model card, docs, press release, staff). **[3P]** = third-party measurement or report. **[PRESS]** = journalist reporting (may relay Prism claims). Inferences are kept in the "Inferences" subsections only.

Generations, oldest first: 1-bit Bonsai 1.7B/4B/8B (2026-03-31) -> Ternary Bonsai 1.7B/4B/8B (2026-04-16) -> Bonsai 27B, 1-bit and ternary, from Qwen3.6-27B (2026-07-14) -> Ternary Bonsai 2 27B, from Qwen3.8-27B (2026-09-17, current flagship). All four whitepapers are PDFs in the PrismML-Eng/Bonsai-demo repo; the local clone holds the same files at the repo root.

---

## 1. What has Prism published about training (papers, reports, blogs, talks, tweets)? QAT vs PTQ, distillation, teacher, tokens, compute, data, rotation choice, why 42 zeros?

### Takeaway
Prism has published **no arXiv paper and no training recipe**. Its only technical documents are four "technical report" whitepapers (Mar/Apr/Jul/Sep 2026), and they describe the *format*, kernels and benchmarks, not the optimization. They say nothing about data, token counts, loss, teacher, QAT vs PTQ, or the zero budget. What Prism does say about method: it "starts from an off-the-shelf pretrained model and moves it into a binary or ternary representation"; it rests on "proprietary Caltech intellectual property"; models were "trained using Google v4 TPUs" (8B) and "v5 TPUs" (27B, Bonsai 2); and Bonsai 2 uses a fixed blockwise Walsh-Hadamard rotation (block 1024, fixed ±1 signs) whose matching transform is applied to activations at runtime.

### Cited Findings
**Statements about the method [PRISM]**
- 1-bit Bonsai 8B whitepaper (dated March 31, 2026): "1-bit Bonsai 8B is built from Qwen3-8B… The architecture is unchanged: the novelty lies entirely in the deployment stack" — [1-bit Bonsai 8B whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/1-bit-bonsai-8b-whitepaper.pdf)
- Same document, §3: "This foundation comes from proprietary Caltech intellectual property that addresses a long-standing research challenge through rigorous mathematics rather than ad hoc heuristics." It also criticizes other near-1-bit approaches that "rely on curated calibration sets, auxiliary metadata, custom layer handling, or bespoke runtimes". No training details are given. — [1-bit Bonsai 8B whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/1-bit-bonsai-8b-whitepaper.pdf)
- Ternary Bonsai whitepaper (April 16, 2026): "The underlying architectures are unchanged: the novelty lies entirely in the weight representation." Base models are Qwen3-8B, -4B and -1.7B. — [Ternary Bonsai whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/ternary-bonsai-8b-whitepaper.pdf)
- Bonsai 27B whitepaper (July 2026) contrasts its approach with BitNet: BitNet "avoids the quality collapse only by pretraining the network from scratch… Bonsai takes the opposite path from BitNet: it starts from an off-the-shelf pretrained model and moves it into a binary or ternary representation". It describes the novelty as "the representation transformation that maps the pretrained 27B into binary or ternary weights while preserving its behavior". It also states that "no conventional post-training scheme is end-to-end", without saying which category Bonsai falls into. — [Bonsai 27B whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-27b-whitepaper.pdf)
- Bonsai 2 27B whitepaper (September 2026): it describes the weights as ternary "in a fixed rotated basis" with FP16 group scales, R = (1/√n)·H_n·S with n = 1024, H_n Walsh-Hadamard and S a fixed diagonal of ±1 signs, and f(x) = W(Rx). The rotation citation is SpinQuant (Liu et al., arXiv:2405.16406). It contains no training section. — [Bonsai 2 27B whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-2-27b-whitepaper.pdf)
- Bonsai 2 HF model card: "Each weight matrix undergoes blockwise orthogonal transformation (1024-block size, fixed ±1 signs) before ternary assignment. The matching transform applies to activations at runtime, folded into stored weights with no additional storage cost." — [HF Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
- Bonsai 2 newly keeps some tensors in full precision, "neither rotated nor quantized": the GDN recurrent-state path (in_proj_a, in_proj_b 5120×48 ×48 layers; conv1d 10240×4; A_log, dt_bias, norm.weight), all RMSNorms and q_norm/k_norm. That is 26,238,464 params (0.0976% of the LM, 52 MB at bf16), moving 1.71 → 1.72 bpw. "The previous release ternarized these alongside everything else." — [Bonsai 2 27B whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-2-27b-whitepaper.pdf)
- Compute: "Bonsai 8B is an 8-billion parameter large language model, trained using Google v4 TPUs" (relayed in WSJ, 2026-03-31) — [WSJ via Edge AI Foundation PDF](https://wiki.edgeaifoundation.org/wp-content/uploads/2026/04/Exclusive-_-Caltech-Researchers-Claim-Radical-Compression-of-High-Fidelity-AI-Models-WSJ.pdf). Press release: "Trained using Google v4 TPUs", plus compute grants from Google and Caltech — [PrismML launch news](https://prismml.com/news/prismml-launches-worlds-first-1-bit-ai-model). Bonsai 27B is "Trained using Google v5 TPUs" — [PrismML Bonsai 27B news](https://prismml.com/news/prismml-releases-bonsai-27b). Bonsai 2 27B is a "27.8-billion parameter… trained using Google v5 TPUs" — [PR Newswire 2026-09-17](https://www.prnewswire.com/news-releases/prismml-launches-bonsai-2-27b-its-most-capable-model-yet-302882228.html). No TPU-hours, chip counts or token counts were found anywhere.
- Hassibi (WSJ): "We spent years developing the mathematical theory required to compress a neural network without losing its reasoning capabilities"; "The mathematics are proprietary"; the IP "is owned by Caltech, and PrismML is the sole exclusive licensee"; the framework "can be applied to any" architecture (transformers, diffusion…). — [WSJ PDF](https://wiki.edgeaifoundation.org/wp-content/uploads/2026/04/Exclusive-_-Caltech-Researchers-Claim-Radical-Compression-of-High-Fidelity-AI-Models-WSJ.pdf)
- The only training recipe Prism *has* described is for the **DSpark drafter** (27B whitepaper §6). It uses a diffusion-flavored block-denoising objective with an interpolating discrete-diffusion schedule and noise-level conditioning. The "distillation loss is weighted by each position's estimated probability of surviving verification". The drafter is six layers, fed hidden states from five evenly spaced target layers, adds ~0.5 GB, and ships 4-bit. — [Bonsai 27B whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-27b-whitepaper.pdf)
- docs.prismml.com describes Bonsai 2 as "Post-Training Quantization (PTQ) with ternary weights" according to a summarizer fetch. I could not verify that wording verbatim, and the whitepaper itself never says PTQ or QAT. Treat it as unconfirmed. — [docs.prismml.com/bonsai-2-27b](https://docs.prismml.com/bonsai-2-27b)
- Hassibi on future plans (TechCrunch, 2026-09-17): "The next models that we will release, hopefully in the next couple of months, will be in the several-hundred-billion-parameter range, and I expect it will be easier to retain the intelligence there." — [TechCrunch](https://techcrunch.com/2026/09/17/prismml-hopes-its-tiny-llm-could-change-how-we-all-use-ai/)

**The 42-zero budget**
- No Prism document, card or post mentions a fixed zero count. The "42 zeros per 128" finding comes from the team's own GGUF scan: 12,507 of 12,864 sampled groups (97.2%) — [local analysis/bonsai2/README.md](file:///home/station/Documents/Bonsai-demo/analysis/bonsai2/README.md). A third party independently measured a zero share of 0.3276 (= 41.9/128), "uniform to three decimals across attention, MLP and the embedding". They *attribute* it to the rounding rule and a Gaussian shape ("Gaussian → ~0.31"), not to a budget — [HF discussion #44](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/discussions/44).

### Inferences
- Prism's public wording ("moves [a pretrained model] into a representation", Caltech "mathematical theory", TPU training, a 1.7B whose weights keep only ~89% sign agreement with the ancestor) fits **post-hoc QAT/distillation from the Qwen ancestor**. The ancestor would be the natural teacher. It does not fit pure PTQ, and Prism has never confirmed either.
- The Bonsai 2 rotation is a *fixed* random-sign Walsh-Hadamard (QuaRot-style) with block 1024. It is not a learned SpinQuant rotation, even though Prism cites SpinQuant. Block 1024 matches the team's H1024 arm.
- A third-party zero share of 0.3276 matches the team's exact-42 count. But exact 42 in 97% of groups is a per-group constraint (e.g., top-86 magnitudes). A distributional rounding rule does not produce that, so the #44 "emerges naturally" explanation is likely wrong on this point.
- Keeping the GDN gating/state params (in_proj_a/b, conv1d, A_log, dt_bias) in high precision is a Bonsai-2-only change. Prism implies it mattered for quality; replications of the 27B should mirror it.

### Gaps
- Not public: training data, token count, teacher identity, loss (KL vs CE), optimizer, schedule, STE variant, whether the rotation is applied inside the training loop, TPU count/hours, or why 42 zeros. I found no arXiv paper, talk, podcast or X thread with these details. Searches of the HN threads found PrismML staff posting only support replies.
- The Caltech patent (below, Q3) was not retrievable directly. Its content is known only second-hand.

---

## 2. What do the Hugging Face model cards and Prism docs state (ancestors, benchmarks, license, bpw, memory)?

### Takeaway
Every Bonsai is a re-representation of a Qwen model with the architecture unchanged: Qwen3-1.7B/4B/8B for the small models, Qwen3.6-27B for Bonsai 27B, and Qwen3.8-27B for Bonsai 2. All are Apache 2.0 and use 128-weight groups with FP16 scales. Claimed retention is ~89% for 1-bit 8B, ~95% for ternary 8B, 94.6%/89.5% for ternary/binary 27B, and 98.2% for Bonsai 2. The published numbers differ across documents, and the small-model gaps to FP are larger than the headline framing suggests.

### Cited Findings
**Formats and sizes [PRISM]**
- 1-bit (Q1_0_g128): w = s_g·(2b−1), 1.125 bpw ideal. Applied to embeddings, attention, MLP and LM head. 8B: 8.19B params, context 65,536. FP16 16.38 GB → GGUF 1,151,820,864 B (1.15 GB, 14.2×); MLX 1.28 GB (1.25 bpw because MLX stores scale+bias: s_mlx = 2s_g, b_mlx = −s_g). — [1-bit Bonsai 8B whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/1-bit-bonsai-8b-whitepaper.pdf)
- Ternary g128: w = s_g·t, t ∈ {−1,0,+1}, b_eff ≈ 1.585 + 16/128 = 1.71 bpw. Ternary footprints: 8B ~1.75 GB (deployed 2.16 GiB), 4B ~0.86 GB (1.05 GiB), 1.7B ~0.37 GB (0.45 GiB). MLX uses 2-bit kernels. — [Ternary Bonsai whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/ternary-bonsai-8b-whitepaper.pdf)
- Ternary-Bonsai-1.7B card: base Qwen3-1.7B, 1.72B params, 32,768 context. Ships Q2_0 g128 (436–463 MB) and g64 (490 MB) vs F16 3.45 GB. Q2_0 decodes as w = (q−1)·scale with q ∈ {0,1,2,3}. Apache 2.0. — [HF Ternary-Bonsai-1.7B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-gguf)
- Bonsai 27B (previous generation): "derived directly from Qwen3.6-27B; the architecture is unchanged". ~24.8B language (64 blocks) + 0.46B vision (27 blocks) + 2.5B embedding/LM head. Ternary 1.71 bpw / 5.9 GB; binary 1.125 bpw / 3.9 GB. Vision tower HQQ 4-bit. Ships a DSpark drafter. — [Bonsai 27B whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-27b-whitepaper.pdf)
- Bonsai 2 27B: 27.36B total (24.35B language, 64 blocks; 0.47B vision, 27 blocks; 2.54B embedding/LM head). Hybrid attention ~75% linear / ~25% full; 262K context. Formats: true ternary 1.72 bpw 5.80 GB; PTQ1_0 1.76 bpw 5.93 GB; PQ2_0 2.16 bpw 7.25 GB; mmproj HQQ 4-bit in a Q8_0 container, 0.63 GB (BF16 reference 0.93 GB). MLX 2-bit g128 is 36 B per 128 weights instead of 34 (a redundant FP16 bias), 2.25 bpw, 8.49 GB package. Reasoning efforts "xhigh or medium (Low does not reduce thinking)". — [Bonsai 2 27B whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-2-27b-whitepaper.pdf)
- The Bonsai 2 HF card gives slightly different numbers: PTQ1_0 1.75 bpw / 5.95 GB, PQ2_0 2.13 bpw / 7.21 GB. It also links the dev repo for the Q2_0 band. — [HF Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
- The press release says 27.8B params; the whitepaper says 27.36B. — [PR Newswire](https://www.prnewswire.com/news-releases/prismml-launches-bonsai-2-27b-its-most-capable-model-yet-302882228.html) vs [whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-2-27b-whitepaper.pdf)
- "Unpacked" BF16 repos (e.g., Ternary-Bonsai-27B-unpacked) are dequantized versions for stock HF tooling, not latent weights. — [HF Ternary-Bonsai-27B-unpacked](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-unpacked)

**Benchmark claims [PRISM]** (all EvalScope + vLLM on H100)
- 1-bit 8B: 6-benchmark average (MMLU-Redux, MuSR, GSM8K, HumanEval+, IFEval, BFCLv3) 70.5 vs Qwen3-8B 79.3. Ternary 8B: 75.5 (MMLU-R 72.6, MuSR 56.2, GSM8K 91, HE+ 77.4, IFEval 81.8, BFCLv3 73.9), described as "more than 95%". — [Ternary Bonsai whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/ternary-bonsai-8b-whitepaper.pdf)
- 10-benchmark averages: Ternary 4B 62.83 vs Qwen3-4B 68.31 (1-bit 4B 55.39). Ternary 1.7B 49.58 vs Qwen3-1.7B 58.24 (1-bit 1.7B 40.88). For the 1.7B: GSM8K 74.2 vs 83.1, MMLU-Redux 52.9 vs 66.8, MATH-500 54.4 vs 74.0, BFCLv3 51.0 vs 71.8. — [Ternary Bonsai whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/ternary-bonsai-8b-whitepaper.pdf)
- Ternary-Bonsai-1.7B card (6 benchmarks): 58.47 vs Qwen3-1.7B 66.57 (~88%). — [HF Ternary-Bonsai-1.7B-gguf README](https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-gguf/raw/main/README.md)
- Bonsai 27B (thinking, 15 benchmarks): Qwen3.6-27B FP16 85.07; Ternary 80.49 (94.6%); 1-bit 76.11 (89.5%); Q4_K_XL 84.99; IQ2_XXS 72.73 (stated true 2.8 bpw / 9.4 GB). Ternary 27B by category: knowledge 76.96 vs 83.15; IF 71.77 vs 78.47; agentic 74.01 vs 80.00; vision 65.19 vs 72.61. — [Bonsai 27B whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-27b-whitepaper.pdf)
- Bonsai 2 27B (thinking, xhigh, 20 benchmarks): 83.9 vs Qwen3.8-27B FP16 85.4 (98.2%), Qwen3.6-27B 83.6, IQ2_XXS 75.2 (7.3 GB). Selected rows: GPQA-D 85.76 vs 90.51; AIME26 95.83 vs 94.58; LiveCodeBench v6 90.07 vs 90.05; BigCodeBench 58.07 vs 61.49; IFBench 74.00 vs 71.00; OCRBench v2 56.88 vs 60.99. Terminal-Bench 2.1 is 52.8 vs 69.7 and SWE-bench Verified 60.8 vs 80.6 (~75%). At medium effort GPQA-D falls to 75.56 vs 84.34. — [Bonsai 2 27B whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-2-27b-whitepaper.pdf)
- The HF card for the same model gives different numbers: 14 benchmarks, FP16 86.32 vs Bonsai 84.78 (also "98.2%"), IQ2_XXS 72.59 (84.1%), UD-Q4_K_XL 85.18. Category scores also differ (e.g., coding 89.42 vs the whitepaper's 81.58; agentic 74.92 vs 77.57). — [HF Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf). Prism's blog lists yet another set (knowledge 83.95, vision 78.59) — [prismml.com/news/bonsai-2-27b](https://prismml.com/news/bonsai-2-27b)
- Prism also claims its ternary/binary 27B models induce "roughly 12–15× less output divergence" under a 4-bit KV cache than the FP16 model. — [Bonsai 27B whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-27b-whitepaper.pdf)

### Inferences
- On *benchmarks*, Prism's own numbers show sizeable small-model gaps: Ternary 1.7B retains ~85–88% of Qwen3-1.7B, with GSM8K −9 and BFCL −21. "Near FP quality" for the 1.7B holds for the team's held-out loss metric, but not on Prism's own task suite. Agree on a metric before claiming parity.
- Bonsai 2's 98.2% comes from short-answer suites. Prism's own long-horizon numbers (TB2.1, SWE-V) show ~75% retention.
- The card and the whitepaper are two different evaluation passes (14 vs 20 benchmarks). Cite the whitepaper's Table 10 as canonical, because it has per-benchmark rows.

### Gaps
- No perplexity/KL numbers vs the ancestor are published by Prism for any generation (only third-party, Q4).
- No release of the high-precision latent/shadow weights, and no published Bonsai 2 drafter.

---

## 3. Who is Prism ML (founders, funding, academic lineage)?

### Takeaway
PrismML is a Pasadena startup that came out of stealth on 2026-03-31. Its co-founders are Caltech professor Babak Hassibi (CEO), Sahin Lale, Omead Pooladzandi and Reza Sadri. It exclusively licenses Caltech-owned IP. Seed funding was $16.25M (March, Khosla, Cerberus, Caltech); a $22.25M seed total was reported in September. The most method-relevant lineage is the Hassibi group's **stochastic/regularizer mirror descent** work and a 2026 Hassibi–Lale patent application on training-concurrent 1-bit quantization. Hassibi is also co-author of the classic Optimal Brain Surgeon (Hassibi & Stork), the ancestor of OBQ/GPTQ.

### Cited Findings
- Team [PRISM]: Babak Hassibi (Co-Founder & CEO), Sahin Lale (Co-Founder, Co-Head of Research), Omead Pooladzandi (Co-Founder, Co-Head of Research), Reza Sadri (Co-Founder, VP Strategy), Shayan Ilbagian (CFO), Karim Mattar (VP Eng), Tushar Bansal (Dir. Product). Advisors: Ion Stoica, Bruno Pati, Julie Schoenfeld. — [prismml.com/about](https://prismml.com/about)
- Funding [PRESS]: "$16.25 million in a SAFE and seed round with investors Khosla Ventures, Cerberus Capital and Caltech" (WSJ, 2026-03-31). Amir Salek (Cerberus, ex-Google TPU/silicon) and Vinod Khosla called it a "mathematical breakthrough". — [WSJ PDF](https://wiki.edgeaifoundation.org/wp-content/uploads/2026/04/Exclusive-_-Caltech-Researchers-Claim-Radical-Compression-of-High-Fidelity-AI-Models-WSJ.pdf). TechCrunch (2026-09-17): seed round $22.25M (Khosla, Cerberus, Caltech). — [TechCrunch](https://techcrunch.com/2026/09/17/prismml-hopes-its-tiny-llm-could-change-how-we-all-use-ai/). Prism releases also cite support from Google and Samsung — [PrismML Bonsai 27B news](https://prismml.com/news/prismml-releases-bonsai-27b); [PR Newswire](https://www.prnewswire.com/news-releases/prismml-launches-bonsai-2-27b-its-most-capable-model-yet-302882228.html)
- Apple [PRESS]: in July 2026 CNBC reported Apple was in exploratory talks with PrismML; the structure was unclear. Hassibi said "things are progressing nicely" — [mlq.ai summarizing CNBC](https://mlq.ai/news/apple-in-talks-to-acquire-prismml-startup-that-shrinks-ai-models-to-run-on-iphone/) (the [CNBC original](https://www.cnbc.com/2026/07/14/apple-prismml-ai-compression-iphone.html) returned 403 to fetch). In September, Hassibi "declined to comment" on Apple — [TechCrunch](https://techcrunch.com/2026/09/17/prismml-hopes-its-tiny-llm-could-change-how-we-all-use-ai/)
- Download counts [PRESS]: the original model had 11M downloads plus 2.6M for smaller variants — [TechCrunch](https://techcrunch.com/2026/09/17/prismml-hopes-its-tiny-llm-could-change-how-we-all-use-ai/)
- Academic lineage:
  - Azizan, Lale, Hassibi, "Stochastic Mirror Descent on Overparameterized Nonlinear Models" (IEEE TNNLS 2021) — [Caltech Authors](https://authors.library.caltech.edu/records/6rrav-h4569)
  - "Explicit Regularization via Regularizer Mirror Descent" — [arXiv 2202.10788](https://arxiv.org/html/2202.10788); [Caltech Authors](https://authors.library.caltech.edu/records/6y975-1rh51)
  - Akhtiamov, Ghane, Hassibi, "One-Bit Quantization for Random Features Models" (arXiv 2510.16250, 2025-10-17): "quantizing weights of all layers except the last incurs no loss in generalization error" asymptotically — [arXiv](https://arxiv.org/abs/2510.16250)
  - Akhtiamov, Ghane, Pooladzandi, Hassibi, "Implicit Bias and Convergence of Matrix Stochastic Mirror Descent" (arXiv 2602.18997, Feb 2026) — [arXiv](https://arxiv.org/abs/2602.18997)
- Patent [3P, second-hand]: a commenter cites "US 20260220467, published July 30, 2026" (Hassibi and Lale). It reportedly describes "training-concurrent one-bit quantization and large-q SMD/RMD potentials that move the weight distribution toward two bipolar values before the final 1-bit projection", with potential ψ(w) = ‖w‖_q^q, q ≫ 1. — [HF discussion #44](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/discussions/44). The Google Patents URL I tried 404'd, so I could not verify the patent directly.
- Hassibi & Stork's approximate-Hessian work (Optimal Brain Surgeon, 1992/93) is the basis later cited by Hessian-aware PTQ papers such as arXiv 2504.05352 — [arXiv 2504.05352](https://arxiv.org/html/2504.05352v1)

### Inferences
- The "proprietary Caltech mathematics" most plausibly means **mirror-descent training with a quantization-friendly potential**. It would start from the pretrained FP model and move it along the near-flat solution manifold toward a ±a (or ternary) clustered point before the final projection. This fits "move a pretrained model into a representation", the TPU training, low sign agreement with the ancestor, and an exact-ternary final projection. It is unconfirmed by Prism.
- A commenter who tested mirror-descent variants found them ineffective in their hands ([#44](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/discussions/44)). So the patent is a lead, not a recipe.

### Gaps
- No Prism-authored paper ties the SMD/RMD or patent work to Bonsai. Patent text, claims and any ternary extension remain unverified.
- No valuation; the $16.25M vs $22.25M discrepancy (extension vs reporting error) is unresolved.

---

## 4. Independent evaluations, community benchmarks, HN/Reddit discussion, quality gaps

### Takeaway
Third parties consistently report the same pattern. Bonsai models are strong on short, bounded benchmarks but measurably diverge from the FP distribution (Bonsai 2 PQ2_0: mean KLD 0.34, PPL ×1.31 vs BF16). They degrade on long-horizon, agentic and multi-turn work: loops, excess reasoning tokens, tool-call failures. A detailed forensic thread concludes Bonsai 2 is rotation + ternary QAT, with ~92% of codes matching rotate-then-RTN.

### Cited Findings
**Hacker News [3P]**
- Thread sizes: Show HN 1-bit Bonsai (2026-03-31, 430 pts/153 comments); Ternary Bonsai (2026-04-18, 225/57); Bonsai Image 4B (2026-05-31, 464/201); Bonsai 27B (2026-07-14, 706/250); Bonsai 2 27B (2026-09-17, 589/200). — [HN Algolia](https://hn.algolia.com/api/v1/search?query=Bonsai&tags=story&hitsPerPage=30)
- 1-bit thread: one commenter (syntaxpr) characterized it as "Not training… Qwen-3 model" (unsourced speculation). Another (fxwin) noted the whitepaper "only compares these to full precision models" rather than to same-footprint quants. Testers reported hallucination and ~190 tok/s on an RTX 3090. — [HN 47593422](https://news.ycombinator.com/item?id=47593422)
- Bonsai 27B thread: verdverm's preliminary lm-eval run gave wikitext 16.75 vs 8.00 baseline, plus gsm8k failures. Others reported reasoning "doom loops", hallucinations, and weaker vision/tool use than Gemma 4 12B. PrismML staff posted only support replies. — [HN 48910545](https://news.ycombinator.com/item?id=48910545)
- Bonsai 2 thread: testers said it "fall[s] apart spectacularly" on longer tasks (loops); a 25-minute agentic run produced nothing useful. Ternary and FP8 versions failed Jabberwocky recitation where bf16 nearly passed. Reported speeds: RTX 3070 40.6 tok/s, RTX 3060 26.5 tok/s. — [HN 49746618](https://news.ycombinator.com/item?id=49746618)

**Hugging Face discussions on Ternary-Bonsai-2-27B-gguf [3P]**
- #54 (mrumel): wikitext-2, 50×2048 chunks vs BF16 (PPL 6.1345). Bonsai PQ2_0 mean KLD 0.340, median 0.176, same-top-token 77.8%, PPL ratio 1.311. For comparison, Q4_K_XL KLD 0.0087 and Q8_0 0.00064. — [HF #54](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/discussions/54)
- #44/#62 (SkyIsNotGreen), forensics:
  - ~92% of ternary codes match rotate-then-RTN (0.920 mean / 0.885 min agreement). The PQ2_0 codec is "plain absmean RTN at g128, no LS refinement".
  - Replacing the remaining ~8% with RTN choices makes PPL go 18.6 → 23,606.
  - Their best reproduction on a 1.7B dense model: rotation-in-the-loop STE-QAT + KD + managed decay, 16k steps, 1.137× PPL / 87.9% retention. An earlier 90.6% figure was retracted as contamination; plain STE+KD gave 2.09×.
  - Single runs, forensic quality. No Prism reply.
  - Sources: [HF #44](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/discussions/44); [HF #62](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/discussions/62)
- #74: the same author applied the recipe to a 35B-A3B MoE (PQ2_0 experts + rank-512 KD corrections, ~$7 of H100 time). They report routing drift as the main MoE loss. — [HF #74](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/discussions/74)
- #47: users dispute the "98.2%" framing. Reports: full 32,768-token reasoning without output, loops, 2–3× more reasoning tokens than Q4 quants, "NOT trustworthy for multi-turn agentic runs". Another user found it strong in a long chat. No Prism reply. — [HF #47](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/discussions/47)
- #33: Prism's intelligence-density table divides Bonsai by the ideal 5.80 GB but competitors by shipped size. With shipped PTQ1_0 (5.95 GB), density is 0.4565 and the "over 2.3×" claim fails. — [HF #33](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/discussions/33)
- Discussion index: many hardware-throughput reports, some tool-call failure reports (#40), LM Studio incompatibility (#37, #68), and one staff post, from bri-prism on GGUF metadata (#59). — [HF discussions](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/discussions)

**Reddit/aggregators**
- The search engine summarized r/LocalLLaMA reactions as "benchmaxxed": strong on the suite, weaker on daily coding, context handling and tool use. I could not reach a primary Reddit thread, so treat this as unverified; it matches the HN/HF reports. — [kie.ai summary](https://kie.ai/blog/what-is-bonsai-27b)

### Inferences
- Mean KLD 0.34 with same-top-token 77.8% means the shipped model is a *different function* from Qwen3.8 that scores similarly on task suites. That is consistent with end-to-end training that relocates weights, not near-lossless rounding. It also matches the team's 1.7B finding (sign agreement 89%, larger layer errors than PTQ, yet better final quality).
- The #62 forensics give 92% RTN agreement *in the rotated basis* for 27B. The team's 89% is sign agreement vs the unrotated 1.7B ancestor. These are different metrics, so don't equate them.

### Gaps
- No independent, systematic benchmark reproduction of Prism's suite (e.g., EvalScope with the same settings) was found. Community data is anecdotal or single-run.
- No primary r/LocalLLaMA thread was retrieved.

---

## 5. Upstreaming status (llama.cpp PQ2_0/PTQ1_0/Hadamard; MLX)

### Takeaway
1-bit Q1_0 is merged in mainline llama.cpp (CPU, April 2026) and 1-bit MLX is merged (July 2026). For Bonsai 2, only the FWHT plumbing is landing upstream, via PrismML's own small PRs. A third-party all-in-one PR for PQ2_0/PTQ1_0 plus Hadamard was closed, with maintainers asking Prism to submit it themselves. Stock llama.cpp cannot run Bonsai 2 as of early October 2026.

### Cited Findings
- ggml-org/llama.cpp#21273 "ggml: add Q1_0 1-bit quantization support (CPU)" by khosravipasha, merged 2026-04-06; follow-up #21636 optimized x86/generic dot. — [PR #21273](https://github.com/ggml-org/llama.cpp/pull/21273); [PR #21636](https://github.com/ggml-org/llama.cpp/pull/21636)
- ggml-org/llama.cpp#27779 "ggml-cpu: add F16 input to the FWHT" by bri-prism, opened 2026-08-27, **merged 2026-09-18**. It frames this as groundwork for Bonsai 2's rotated basis. Follow-ups: Metal #29094, CUDA #29096, Vulkan #29101. — [PR #27779](https://github.com/ggml-org/llama.cpp/pull/27779). (The local README still lists #27779 as "Open"; it is out of date.)
- ggml-org/llama.cpp#29077 (QuentinDanblon, third party): "add PQ2_0 and PTQ1_0 ternary types; apply Hadamard-folded weights".
  - PQ2_0 = type 142, g128, 34-byte block (fp16 scale + 2-bit slot per trit, Q2_0 codec). PTQ1_0 = type 143, g128, 28-byte block (fp16 scale + base-3 trits, 5/byte, remainder 4/byte, as TQ1_0).
  - Metadata prefix `prism.hadamard.*`; y = W'·(H·(s·P·x)).
  - Opened 2026-09-18, closed 2026-09-22. Code owner CISC: "Please leave this for PrismML to submit themselves."
  - Sources: [PR #29077](https://github.com/ggml-org/llama.cpp/pull/29077); feature request [issue #29058](https://github.com/ggml-org/llama.cpp/issues/29058)
- Community ports exist: ik_llama.cpp #2540/#2550, beellama.cpp #169. Ollama import fails on types 142/143 (#18521). — [ik_llama #2540](https://github.com/ikawrakow/ik_llama.cpp/pull/2540); [beellama #169](https://github.com/Anbeeld/beellama.cpp/pull/169); [ollama #18521](https://github.com/ollama/ollama/issues/18521)
- MLX: ml-explore/mlx#3161 "Add 1-bit affine quantization support (Metal)" by khosravipasha, opened 2026-02-24, **merged 2026-07-15**. A maintainer was initially skeptical; it was accepted once native 1-bit models existed. — [mlx #3161](https://github.com/ml-explore/mlx/pull/3161). (The local README still lists it as pending.) Bonsai 2's MLX pack runs on stock MLX (2-bit affine) — [Bonsai 2 whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-2-27b-whitepaper.pdf)
- Prism's fork has its own PRs, e.g., PrismML-Eng/llama.cpp#253 "add PQ1_0, the Q1_0 binary codec at group 64". — [PrismML-Eng/llama.cpp #253](https://github.com/PrismML-Eng/llama.cpp/pull/253)

### Inferences
- Stock llama.cpp will remain unable to run Bonsai 2 until Prism submits the type and `prism.hadamard` metadata PRs. The gibberish risk of the dev-repo Q2_0 band persists until then.

### Gaps
- No public timeline from Prism for submitting PQ2_0/PTQ1_0 upstream.

---

## 6. Related prior work they cite, or that clearly matches their method

### Takeaway
Prism cites BitNet / BitNet b1.58 as the from-scratch foil, SpinQuant for rotation, and HQQ for the vision tower. The method most consistent with the evidence combines (a) QuaRot/SpinQuant-style Hadamard incoherence processing, (b) BitNet-b1.58 absmean ternarization with group-128 FP16 scales, and (c) post-hoc QAT/distillation from the FP ancestor, possibly with mirror-descent geometry from the founders' own work.

### Cited Findings
- Cited by Prism: BitNet (arXiv 2310.11453) and "The Era of 1-bit LLMs" b1.58 (arXiv 2402.17764) as the from-scratch line that "has not scaled" beyond ~2B. HQQ (Badri & Shaji 2023) for the 4-bit vision tower. DSpark (DeepSeek-AI 2026) and DFlash (arXiv 2602.06036) for drafting. — [Bonsai 27B whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-27b-whitepaper.pdf). SpinQuant (arXiv 2405.16406) and the Fino–Algazi fast WHT (1976) for Bonsai 2's rotation — [Bonsai 2 27B whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-2-27b-whitepaper.pdf)
- The Ternary whitepaper also cites the BitNet papers ([6], [7]) — [Ternary Bonsai whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/ternary-bonsai-8b-whitepaper.pdf)
- Related ternary-storage work (not Prism): BITCOS (Georganas, Heinecke, Dubey; arXiv 2609.16338, 2026-09-14) reports zeros up to 51.5% across 29 ternary models and proposes a 2−z bpw bitmap+sign layout. It does not mention Bonsai. — [arXiv 2609.16338](https://arxiv.org/abs/2609.16338)
- Mirror-descent view of quantization (Ajanthan et al., arXiv 1910.08237) is an earlier, non-Caltech version of the same idea. — [arXiv 1910.08237](https://arxiv.org/abs/1910.08237)

### Inferences
- The embedding matching BitNet absmean in 99.83% of entries (team finding) together with "plain absmean RTN at g128" ([#62](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/discussions/62)) suggests Prism's final projection is the BitNet b1.58 absmean rule. If so, the secret is in the pre-projection weights, which training shapes, not in the quantizer.
- Prism's own framing "starts from an off-the-shelf pretrained model", set against BitNet's from-scratch pretraining, positions Bonsai as continued training / QAT on a pretrained model. That family includes the BitNet-distillation and low-bit QAT literature, though Prism cites none of it.

### Gaps
- Prism cites no QAT, distillation, GPTQ/OBQ or QuaRot papers, so which ones they actually used is unknown.

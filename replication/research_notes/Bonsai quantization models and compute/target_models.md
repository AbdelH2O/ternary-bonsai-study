# Target models for a ternary / ~1.75-2.13 bpw QAT-distillation pipeline (landscape as of 2026-10-05)

Method note: most per-model facts below were pulled on 2026-10-05 from the Hugging Face Hub API (`/api/models/<id>`: license, `safetensors.total` param count, `createdAt`; `<id>/resolve/main/config.json`: architecture, layer types, experts, context). HF's `downloads` field is a **trailing 30-day** count, not all-time. Model-page URLs are cited for each; the vendor listing endpoints used were `https://huggingface.co/api/models?author=<org>&sort=downloads&direction=-1`. Size estimates labelled "est." are my own arithmetic, scaled from Bonsai 2 27B's actual ratio (27.78B params -> 5.9 GB, i.e. ~1.70 effective bits/weight overall incl. embeddings/vision) and from the PQ2_0 band (2.13 bpw). Per-model embedding/vision overheads will move them by a few hundred MB.

## 1. What Prism already serves and where the gap is

### Takeaway
Prism has ternary/1-bit versions of exactly one modern architecture: the **dense Qwen3.5-style hybrid GDN 27B** (Qwen3.6-27B for gen 1, Qwen3.8-27B for Bonsai 2), plus older text-only 1.7B/4B/8B and a 4B image model. It has **no MoE, no Gemma, no Mamba-hybrid, and no 35B-A3B / 100B+ release**. It has said it is going to several-hundred-billion-parameter models next. That leaves the mid-size MoE tier (26-36B total, 3-4B active) as the clearest gap, and Prism looks set to move into the frontier tier itself.

### Cited Findings
- Bonsai 2 27B: base model `Qwen/Qwen3.8-27B`, Apache-2.0, created 2026-09-16, ~4.12M downloads/30d and 2,432 likes. The single most-downloaded Prism repo. — [HF prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
- Gen-1 Ternary-Bonsai-27B and Bonsai-27B (1-bit) both have base model `Qwen/Qwen3.6-27B` (2026-07-04). — [HF Ternary-Bonsai-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf), [HF Bonsai-27B-gguf](https://huggingface.co/prism-ml/Bonsai-27B-gguf)
- Full Prism catalogue: Bonsai 1.7B/4B/8B (1-bit, Mar 2026), Ternary-Bonsai 1.7B/4B/8B (Apr 2026), bonsai-image binary/ternary 4B (May 2026), Bonsai/Ternary-Bonsai 27B (Jul 2026), Ternary-Bonsai-2-27B (Sep 2026), plus AWQ-4bit and MLX variants. None are MoE. — [HF API prism-ml listing](https://huggingface.co/api/models?author=prism-ml&sort=createdAt&direction=-1&limit=60)
- Prism says Bonsai 2 27B keeps 98.2% of Qwen3.8 27B's score on its 20-benchmark suite (83.9 vs 85.4), reaches up to 143 tok/s on an RTX 5090, and was released 2026-09-17. — [PrismML news](https://prismml.com/news/prismml-launches-bonsai-2-27b)
- Roadmap: "PrismML says its next compressed models will be in the several-hundred-billion-parameter range within a couple of months." — [DataNorth](https://datanorth.ai/news/prismml-releases-ternary-bonsai-2-27b). A search snippet also mentioned "a Bonsai 27B variant tuned for agentic coding" as next on the roadmap. I could not confirm that in the PrismML press release or in [SiliconANGLE](https://siliconangle.com/2026/09/18/prismml-launches-bonsai-2-27b-a-high-intelligence-ai-model-so-small-it-fits-on-consumer-hardware/), so treat it as unverified.
- The PrismML press release mentions no MoE or larger models. — [PrismML news](https://prismml.com/news/prismml-launches-bonsai-2-27b)
- Community ternary activity is almost all on Bonsai 2 derivatives (abliterated, MTP, DFlash). Two independent ternary efforts stand out: `CodeMasterCody3D/taardis-27b-full-ternary` (~266k/30d, Sep 2026) and `sdkyuan/qwen3.8-27B-qat-q2_0-gguf` (~5k/30d, Aug 2026). — [HF search "ternary"](https://huggingface.co/api/models?search=ternary&sort=downloads&direction=-1&limit=20), [HF search "Q2_0"](https://huggingface.co/api/models?search=Q2_0&sort=downloads&direction=-1&limit=20)
- A small-scale replicator already exists: `benzeng/tritfold-0.6b-ptq1_0` and `tritfold-1.7b-*-ptq1_0` (Sep-Oct 2026). These are PTQ1_0 GGUFs at 0.6B/1.7B, which suggests a Qwen3-0.6B/1.7B base. — [HF search "PTQ1_0"](https://huggingface.co/api/models?search=PTQ1_0&sort=downloads&direction=-1&limit=20)

### Inferences
- If Prism goes to "several hundred billion" next, DeepSeek-V4-Flash (291B, MIT), GLM-5.3-Flash (321B, MIT) and Qwen3.5-397B-A17B (Apache) are the obvious candidates for it. A single-GPU team should not race Prism there.
- The uncontested, high-demand gap is the **~26-36B-total MoE tier** (Qwen3.6-35B-A3B, Gemma 4 26B-A4B, Nemotron 3.5 Lightning 30B-A3B, GLM-4.7-Flash). A second gap is a **Gemma 4** ternary of any size, since Prism has only done Qwen.

### Gaps
- No Prism statement on which architecture it will target next. "Several hundred billion" does not name a model.
- The base model of Prism's gen-1 1.7B/4B/8B is not stated on the GGUF cards (they point to `*-unpacked` repos). AGENTS.md describes them as text-only legacy.

## 2. The landscape: per-model facts (Oct 2026)

### Takeaway
The 2026 open-weight frontier is dominated by Qwen3.5/3.6/3.8 (Apache, hybrid GDN, native vision, 262K context), Gemma 4 (Apache, sliding-window + MoE/dense, vision), DeepSeek V4 (MIT, 291B-1.6T MoE), GLM-5.x (MIT/custom, 321B-753B), Kimi K3 (2.78T, custom license, ships MXFP4), and NVIDIA Nemotron 3/3.5 (Mamba-2 hybrid MoE). Llama 4, Phi-4 and Mistral's dense line have fallen well behind in downloads.

### Cited Findings: Qwen family (same architecture as Bonsai 2 = maximum pipeline reuse)
Every Qwen3.5/3.6/3.8 model below uses `qwen3_5` / `qwen3_5_moe`. `full_attention_interval=4` means 3 Gated-DeltaNet (linear_attention) layers per full-attention layer. Vocab is 248,320, context is 262,144, and a vision tower is included (`vision_config` present).

| Model | Params (total / active) | Layers (linear / full) | Notes | License / date | 30d DL | Source |
|---|---|---|---|---|---|---|
| Qwen3.8-27B | 27.78B dense | 64 (48/16), hidden 5120 | Bonsai 2 base | Apache-2.0 / 2026-08-05 | 6.76M (+FP8 4.80M, unsloth GGUF 6.55M) | [HF](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Qwen3.6-27B | 27.78B dense | 64 (48/16) | Bonsai gen-1 base | Apache-2.0 / 2026-04-21 | 2.38M | [HF](https://huggingface.co/Qwen/Qwen3.6-27B) |
| **Qwen3.6-35B-A3B** | 35.95B / ~3B (256 experts, top-8, expert FFN 512) | 40 (30/10), hidden 2048 | Same GDN hybrid + MoE + vision | Apache-2.0 / 2026-04-15 | 3.41M (+FP8 5.15M, NVIDIA NVFP4 5.89M, unsloth GGUF 1.30M) | [HF](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) |
| Qwen3.5-35B-A3B | 35.95B / ~3B | 40 (30/10) | Previous gen of the above | Apache-2.0 / 2026-02-24 | 1.54M | [HF](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) |
| Qwen3.5-122B-A10B | 125.1B / ~10B (256 experts, top-8) | 48 (36/12), hidden 3072 | | Apache-2.0 / 2026-02-24 | 0.56M (+NVFP4 1.12M) | [HF](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) |
| Qwen3.5-397B-A17B | 403.4B / ~17B (512 experts, top-10) | 60 (45/15) | | Apache-2.0 / 2026-02-16 | 0.45M | [HF](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) |
| Qwen3.5-9B | 9.65B dense | 32 (24/8), hidden 4096, untied emb | | Apache-2.0 / 2026-02-27 | 8.83M | [HF](https://huggingface.co/Qwen/Qwen3.5-9B) |
| Qwen3.5-4B | 4.66B dense | 32 (24/8), hidden 2560, tied emb | | Apache-2.0 / 2026-02-27 | 7.97M | [HF](https://huggingface.co/Qwen/Qwen3.5-4B) |
| Qwen3.5-2B | 2.27B dense | 24 (18/6), hidden 2048, tied emb | | Apache-2.0 / 2026-02-28 | 4.78M | [HF](https://huggingface.co/Qwen/Qwen3.5-2B) |
| Qwen3.5-0.8B | 0.87B dense | 24 (18/6), hidden 1024, tied emb | | Apache-2.0 / 2026-02-28 | 2.54M | [HF](https://huggingface.co/Qwen/Qwen3.5-0.8B) |
| Qwen3.8-Flash-Next | ~180B stored (125B main + 51B n-gram embedding + ~4B MTP) / 6B active; 512 experts top-10 | 48 (36/12) | Qwen4 architecture preview (`qwen4_exp`) | **Qwen Community License 1.0** / 2026-08-24 | 1.53M | [HF](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), [codersera](https://codersera.com/blog/qwen-3-8-model-lineup-2026) |

- Base (pretrained) checkpoints exist for Qwen3.5-0.8B-Base, 2B-Base, 4B-Base and 9B-Base. — [HF search Qwen3.5](https://huggingface.co/api/models?search=Qwen3.5&sort=downloads&direction=-1&limit=40)
- Per codersera, "There is no Qwen 3.8 at 0.8B, 3B, 9B, 14B, 35B, 70B or 122B." The open Qwen3.8 weights are 27B, Flash-Next, and a 2.4T-A95B model under the "Qwen3.8-Max License". — [codersera](https://codersera.com/blog/qwen-3-8-model-lineup-2026)
- Alibaba calls Flash-Next "an experimental preview of the architecture that will underpin Qwen4". — [orcarouter](https://www.orcarouter.ai/blog/qwen3-8-flash-next-release)
- llama.cpp added the `qwen4exp` architecture (PR #27742), but at that release "the QSA indexer and the PLE n-gram embedding were not wired up yet". Unsloth maintains a dedicated Flash-Next branch. — [newreleases b10660](https://newreleases.io/project/github/ggml-org/llama.cpp/release/b10660), [unsloth GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)

### Cited Findings: Google Gemma 4 (Apache-2.0, vision, tied 262,144-entry vocab, sliding window + periodic full attention; no linear attention)
| Model | Params | Arch | Context | Date | 30d DL | Source |
|---|---|---|---|---|---|---|
| **gemma-4-26B-A4B-it** | 25.81B total, 128 experts (expert FFN 704); ~3.8B active per [atomic.chat](https://atomic.chat/blog/guides/best-local-llm-16gb) | 30 layers (25 sliding-1024 / 5 full) | 262,144 | 2026-03-11 | **12.72M** (top of all LLMs surveyed) | [HF](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| gemma-4-31B-it | 31.27B dense | 60 layers (50/10) | 262,144 | 2026-03-11 | 9.87M | [HF](https://huggingface.co/google/gemma-4-31B-it) |
| gemma-4-12B-it | 11.96B dense ("unified" arch) | 48 layers (40/8) | 262,144 | 2026-05-23 | 1.84M | [HF](https://huggingface.co/google/gemma-4-12B-it) |
| gemma-4-E4B-it | 8.0B stored (per-layer-embedding design) | 42 layers (35 sliding-512 / 7) | 131,072 | 2026-03-02 | 4.44M | [HF](https://huggingface.co/google/gemma-4-E4B-it) |
| gemma-4-E2B-it | 5.12B stored | 35 layers (28/7) | 131,072 | 2026-03-02 | 3.03M | [HF](https://huggingface.co/google/gemma-4-E2B-it) |

### Cited Findings: other families
| Model | Params (total / active) | Arch | Modalities / context | License / date | 30d DL | Source |
|---|---|---|---|---|---|---|
| DeepSeek-V4-Flash | 290.9B, 256 routed experts, top-6 | `deepseek_v4`, sliding window 128, ships FP8 | text / 1,048,576 | MIT / 2026-04-22 | 1.01M (V4-Flash-0731: 4.54M) | [HF](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) |
| DeepSeek-V4.1-Flash | 763.2B, 384 experts, top-6 | `deepseek_v41`, vision | text+image / 1M | MIT / 2026-09-10 | 0.87M | [HF](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) |
| DeepSeek-V4-Pro | 1.6T, 384 experts | | text / 1M | MIT / 2026-04-22 | 0.42M | [HF](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) |
| Kimi-K3 | 2.78T, 896 experts | `kimi_linear` (linear-attention hybrid), vision; ships `mxfp4-pack-quantized` | 1M | custom "kimi-k3" / 2026-06-13 | 1.24M | [HF](https://huggingface.co/moonshotai/Kimi-K3) |
| Kimi-Linear-48B-A3B-Instruct | 49.1B / ~3B, 256 experts | `kimi_linear` (KDA linear attention hybrid) | text | MIT / 2025-10-30 | 0.21M | [HF](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct) |
| GLM-5.3-Flash | 321.3B, 288 experts, top-8 | `glm5_next`: 34 linear-attention + 11 DeepSeek-sparse-attention layers, vision | 1M | MIT / 2026-08-25 | 5.74M | [HF](https://huggingface.co/zai-org/GLM-5.3-Flash) |
| GLM-5.3 | 753.3B | `glm_moe_dsa` | text / 1M | custom "glm-5.3" / 2026-08-25 | 1.44M | [HF](https://huggingface.co/zai-org/GLM-5.3) |
| **GLM-4.7-Flash** | 31.2B, 64 experts, top-4 | `glm4_moe_lite` | text / 202,752 | MIT / 2026-01-19 | 1.68M | [HF](https://huggingface.co/zai-org/GLM-4.7-Flash) |
| gpt-oss-20b | 20.9B, 32 experts, top-4 | alternating sliding-128 / full, native MXFP4 experts | text / 131,072 | Apache-2.0 / 2025-08-04 | 6.32M | [HF](https://huggingface.co/openai/gpt-oss-20b) |
| gpt-oss-120b | 116.8B, 128 experts, top-4 | same, MXFP4 | text / 131,072 | Apache-2.0 / 2025-08-04 | 4.41M | [HF](https://huggingface.co/openai/gpt-oss-120b) |
| Nemotron-3-Nano-30B-A3B | 31.6B, 128 experts, top-6 | `nemotron_h`: Mamba-2 + MoE + few attention layers (pattern `MEMEM*E...`) | text / 262,144 | NVIDIA Nemotron Open Model License / 2025-12-04 | 0.90M (NVFP4 1.36M) | [HF](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) |
| **Nemotron-3.5-Lightning-30B-A3B** | 31.6B, same arch | `nemotron_h` | text / 262,144 | **OpenMDW-1.1** (permissive) / 2026-08-01 | 0.57M (NVFP4 0.81M) | [HF](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) |
| Nemotron-3-Nano-4B | 3.97B dense | Mamba-2 hybrid (`M-M-M-MM...*`) | text / 262,144 | NVIDIA Nemotron OML / 2026-03-07 | 2.75M | [HF](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16) |
| Nemotron-3-Super-120B-A12B | 123.6B, 512 experts, top-22 | Mamba-2 hybrid MoE | text / 262,144 | NVIDIA Nemotron OML / 2026-03-10 | 1.15M | [HF](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) |
| Mistral-Small-4-119B-2603 | 119.4B, 128 experts, top-4 | `mistral4`, vision | 1,048,576 | Apache-2.0 / 2026-01-23 | 0.07M | [HF](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603) |
| Mistral-Medium-3.5-128B | 127.7B dense | `ministral3`, vision | 262,144 | custom ("other") / 2026-03-31 | 0.10M | [HF](https://huggingface.co/mistralai/Mistral-Medium-3.5-128B) |
| Devstral-Small-2-24B-2512 | 24.0B dense | `ministral3`, vision | 393,216 | Apache-2.0 / 2025-11-28 | 0.32M | [HF](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512) |
| Ministral-3-14B-2512 | 13.9B dense | vision | 262,144 | Apache-2.0 / 2025-10-31 | 0.21M | [HF](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512) |
| granite-4.2-30b / 8b / 3b | 29.3B / 8.8B / 3.7B dense | config says `GraniteForCausalLM` (plain transformer, not the Granite-4.0 Mamba hybrid) | text / 131,072 | Apache-2.0 / 2026-08-07 | GGUF repos ~0.30M each | [HF 30b](https://huggingface.co/ibm-granite/granite-4.2-30b), [8b](https://huggingface.co/ibm-granite/granite-4.2-8b), [3b](https://huggingface.co/ibm-granite/granite-4.2-3b) |
| LFM2.5-8B-A1B | 8.47B / ~1B, 32 experts, top-4 | `lfm2_moe`: 18 short-conv + 6 attention layers | text / 128,000 | LFM Open License 1.0 / 2026-05-28 | 31k (GGUF 447k) | [HF](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B) |
| LFM2.5-2.6B | 2.70B dense | `lfm2`: 22 conv + 8 attention | text / 131,072 | LFM1.0 / 2026-07-28 | 103k (GGUF 1.04M) | [HF](https://huggingface.co/LiquidAI/LFM2.5-2.6B) |
| LFM2.5-1.2B-Instruct | 1.17B | 10 conv + 6 attention | text / 128,000 | LFM1.0 / 2026-01-06 | 117k (GGUF 351k) | [HF](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) |
| SmolLM3-3B | 3.08B dense | 36 full-attention layers | text / 65,536 | Apache-2.0 / 2025-07-08 | 629k | [HF](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) |
| Olmo-3-7B-Instruct / Olmo-3.1-32B-Instruct | 7.30B / 32.2B dense | sliding-4096 (3:1) + full | text / 65,536 | Apache-2.0 / 2025-11 & 2025-12 | 476k / 16k | [HF 7B](https://huggingface.co/allenai/Olmo-3-7B-Instruct), [32B](https://huggingface.co/allenai/Olmo-3.1-32B-Instruct) |
| Falcon-H1R-7B | 7.59B | `falcon_h1` (parallel Mamba+attention hybrid) | text / 262,144 | Falcon LLM License / 2025-10-29 | 1.5k | [HF](https://huggingface.co/tiiuae/Falcon-H1R-7B) |
| phi-4 | 14.66B dense | `phi3` | text / 16,384 | MIT / 2024-12-11 | 426k | [HF](https://huggingface.co/microsoft/phi-4) |
| Llama-4-Scout-17B-16E | 108.6B total | MoE (config gated, 401 to anonymous requests) | | Llama 4 Community License / 2025-04-02 | 172k | [HF](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) |

- Meta's most-downloaded models are still Llama 3.x (Llama-3.2-1B-Instruct 7.49M, Llama-3.1-8B-Instruct 6.10M). Llama-4-Scout is at 172k. — [HF API meta-llama](https://huggingface.co/api/models?author=meta-llama&sort=downloads&direction=-1&limit=25)
- The Mistral, Microsoft (Phi) and TII (Falcon) LLMs all sit below ~2.2M/30d. Mistral-7B-Instruct-v0.3 at 2.14M is the top one. — [HF API mistralai](https://huggingface.co/api/models?author=mistralai&sort=downloads&direction=-1&limit=25), [HF API microsoft](https://huggingface.co/api/models?author=microsoft&sort=downloads&direction=-1&limit=25), [HF API tiiuae](https://huggingface.co/api/models?author=tiiuae&sort=downloads&direction=-1&limit=25)

### Inferences
- **Architecturally closest to Bonsai 2** (the pipeline reuses as-is: GDN kernels, rotation scheme, mmproj, llama.cpp `qwen35`/`qwen35moe` paths): every Qwen3.5/3.6 model. Qwen3.6-35B-A3B differs only in its MoE FFN.
- **Linear-attention hybrids from other vendors** (GLM-5.3-Flash, Kimi-Linear/K3) are also GDN/KDA-style, but they are huge or text-only. Nemotron-H and Falcon-H1 use Mamba-2, LFM2 uses short convolutions, and Gemma 4 uses only sliding-window attention. All of these need new handling (conv/SSM weights, different sensitive tensors).

### Gaps
- No Phi-5 or other newer Microsoft LLM showed up in Microsoft's top-25 list. Phi-4 (Dec 2024) is the latest Phi seen.
- Llama 4 config could not be read anonymously, so its architecture is not verified here.
- LMArena / Artificial Analysis rankings were not fetched. The quality ranking between, e.g., Qwen3.6-35B-A3B and Gemma 4 26B-A4B is not established in these notes.
- Granite 4.2's config reports a plain dense transformer. I did not verify whether IBM dropped the Granite 4.0 Mamba hybrid or publishes it in separate `-h` repos.

## 3. Existing official and community low-bit releases (where the gap is)

### Takeaway
Official vendor low-bit releases stop at **~4 bits**: gpt-oss MXFP4, Kimi-K3 MXFP4, Gemma 4 QAT int4/Q4_0, NVIDIA NVFP4/FP8, Qwen FP8/GPTQ-Int4, and IBM mxfp4. Sub-2-bit models from vendors are research-scale only: BitNet b1.58 2B4T (Apr 2025) and Falcon3 1.58-bit (2024). Below 2.5 bpw, everything else is PTQ (Unsloth UD-IQ1_M/IQ2_XXS, EXL3 ~2.5-3 bpw) or Prism's Qwen-only QAT. So **a QAT ternary of any MoE or of any Gemma model would be new**.

### Cited Findings
- Gemma 4 QAT: on 2026-06-05 Google released QAT checkpoints for every Gemma 4 size. The 26B-A4B drops from ~17 GB at standard Q4 to ~15 GB, a "~72% VRAM cut versus BF16". — [runaihome](https://runaihome.com/blog/gemma-4-qat-local-ai-hardware-update-2026/), [dev.to](https://dev.to/jovan_chan_9500711396d4e6/gemma-4-qat-for-local-ai-in-2026-how-googles-june-5-checkpoints-put-the-26b-in-15gb-2ghl)
- Official QAT repos: `google/gemma-4-12B-it-qat-q4_0-gguf` (796k), `gemma-4-E4B-it-qat-q4_0-gguf` (736k), `gemma-4-E2B-it-qat-q4_0-gguf` (428k), `gemma-4-12B-it-qat-w4a16-ct` (831k). Unsloth ships `gemma-4-26B-A4B-it-qat-GGUF` (501k). — [HF search gemma-4](https://huggingface.co/api/models?search=gemma-4&sort=downloads&direction=-1&limit=40)
- gpt-oss ships MoE weights natively in MXFP4 (`quantization_config` leaves attention and router unquantized). — [HF gpt-oss-20b config](https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json)
- Kimi-K3 ships in `mxfp4-pack-quantized` format. — [HF Kimi-K3 config](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)
- NVIDIA publishes NVFP4 versions of other vendors' models: Qwen3.6-35B-A3B-NVFP4 (5.89M/30d, NVIDIA's most-downloaded repo), Gemma-4-31B-IT-NVFP4 (1.51M), Qwen3.5-122B-A10B-NVFP4 (1.12M), Gemma-4-26B-A4B-NVFP4 (1.07M), Qwen3.8-27B-NVFP4 (590k). — [HF API nvidia](https://huggingface.co/api/models?author=nvidia&sort=downloads&direction=-1&limit=25)
- IBM ships fp8/nvfp4/mxfp4 variants of Granite 4.2. — [HF search granite-4.2](https://huggingface.co/api/models?search=granite-4.2&sort=downloads&direction=-1&limit=40)
- BitNet: `microsoft/bitnet-b1.58-2B-4T` (Apr 2025, 19k/30d) and `-gguf` (16k). Microsoft has since applied BitNet to an ASR model (`VibeVoice-ASR-BitNet`, 66k, Jul 2026) and to embeddings (`bitnet-embedding-0.6b`, Jul 2026), but has released no new BitNet chat LLM. — [HF search bitnet](https://huggingface.co/api/models?search=bitnet&sort=downloads&direction=-1&limit=20)
- Falcon3 1.58-bit (1B-10B, Nov-Dec 2024) gets ~1k/30d or less. — [HF search 1.58bit](https://huggingface.co/api/models?search=1.58bit&sort=downloads&direction=-1&limit=20)
- Unsloth's Qwen3.6-35B-A3B GGUF repo includes UD-IQ1_M, UD-IQ2_XXS, UD-IQ2_M and UD-Q2_K_XL files (1.30M/30d, 1,660 likes). — [HF unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/tree/main)
- EXL3: `turboderp/Qwen3.8-27B-exl3` (73k). Sub-3 bpw community EXL3s exist for frontier MoEs, e.g. DeepSeek-V4.1-Flash 2.9 bpw and GLM-5.3-Flash 2.51 bpw. — [HF search EXL3](https://huggingface.co/api/models?search=EXL3&sort=downloads&direction=-1&limit=20)
- Explicit IQ1_M repos have tiny download counts (<=2k): e.g. `Manojb/Qwen_Qwen3.5-35B-A3B-IQ1_M.gguf`, `dxx117/Qwen3.6-35B-REAP-IQ1M`, `MarxistLeninist/Qwen3.8-27B-IQ1_M-GGUF`. — [HF search IQ1_M](https://huggingface.co/api/models?search=IQ1_M&sort=downloads&direction=-1&limit=20)
- A Gemma 4 31B "1-bit" community attempt exists (`arcticoneai/gemma4-31B-1bit`, ~0 downloads). — [HF search 1bit](https://huggingface.co/api/models?search=1bit&sort=downloads&direction=-1&limit=20)

### Inferences
- The ~4-bit tier is saturated by official vendor releases, so the pipeline should not stop at 4 bits. Its value lies in **1.6-2.2 bpw with QAT quality**, where the only competition is PTQ IQ1/IQ2 (known to degrade sharply) and Prism.
- Standalone IQ1 repos get few downloads, but the people downloading big Unsloth repos clearly want sub-4-bit MoEs. I could not get per-file download splits (see Gaps).

### Gaps
- HF does not expose per-file download counts, so I could not measure how often the IQ1/IQ2 files inside the Unsloth/bartowski repos are actually downloaded.
- I found no reliable data on quality loss of UD-IQ1_M/IQ2_XXS for Qwen3.6-35B-A3B or Gemma 4 26B-A4B.

## 4. Fit criteria and ranked shortlist

### Takeaway
**Best high-impact target: Qwen3.6-35B-A3B.** It is the same Apache-licensed hybrid-GDN + vision architecture as Bonsai 2, an MoE (est. ~7.6 GB at Bonsai-2 density, ~9.6 GB at PQ2_0) with 3B active parameters, very popular (top-downloaded NVFP4/FP8 builds), and Prism has not shipped it. **Runner-up: Gemma 4 26B-A4B**, the most-downloaded open LLM surveyed (Apache, vision), at est. ~5.5-6.9 GB, needing new architecture work. **Cheap experiments should climb the Qwen3.5 ladder (0.8B -> 2B -> 4B -> 9B)** because it is the identical architecture to the Bonsai 2 base.

### Cited Findings (supporting the fit reasoning)
- Qwen3.6-35B-A3B and Qwen3.8-27B use the same GDN block layout (`full_attention_interval` 4, vocab 248,320, vision config). The 35B-A3B adds 256 experts with top-8 routing and a 512-wide expert FFN. — [HF Qwen3.6-35B-A3B config](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json), [HF Qwen3.8-27B config](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)
- The Qwen3.5 small models share the `Qwen3_5ForConditionalGeneration` architecture with Qwen3.8-27B. The 0.8B/2B/4B tie embeddings and the 9B/27B do not. — [HF Qwen3.5-0.8B config](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json), [HF Qwen3.5-9B config](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json)
- At 16 GB, the PTQ comparison points are a 35B-A3B hybrid MoE in 16 GB VRAM at 13.7 GB using a mixed IQ3_XXS/IQ2_S quant, and Gemma 4 26B-A4B at 13.9 GB with IQ4_XS. — [atomic.chat](https://atomic.chat/blog/guides/best-local-llm-16gb)
- llama.cpp support is mature for qwen35/qwen35moe (the Bonsai demo runs on it) and for Gemma 4 (Unsloth GGUFs exist). Flash-Next's `qwen4exp` support is partial. — [unsloth Gemma 4 26B GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF), [newreleases b10660](https://newreleases.io/project/github/ggml-org/llama.cpp/release/b10660)
- Nemotron 3.5 Lightning moved to the OpenMDW-1.1 license, while Nemotron 3 Nano and Super use the NVIDIA Nemotron Open Model License. — [HF Lightning](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16), [HF Nano](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)

### Ranked shortlist (b): high-impact targets
Sizes are est. at ~1.70 bpw effective (Bonsai 2 density) / 2.13 bpw (PQ2_0). Teacher memory is BF16 weights only (2 bytes/param).

1. **Qwen3.6-35B-A3B** (Apache-2.0, 2026-04, vision, 262K, hybrid GDN MoE). Est. 7.6 / 9.6 GB, so it fits 16 GB laptops with room for long context and is borderline on 12 GB phones, with 3B-active decode speed. This gives the most pipeline reuse of any target: only expert/router handling is new. Demand is the strongest of any non-Prism target: 3.4M base + 5.1M FP8 + 5.9M NVIDIA NVFP4 + 1.3M Unsloth GGUF per 30 days. Risks:
   - Routing sensitivity. Keep the router and shared parts at higher precision, the way gpt-oss keeps its router unquantized.
   - The BF16 teacher is ~72 GB. On one consumer GPU, precompute top-k teacher logits offline or rent an 80 GB card.
   - Qwen3.8 has no 35B-A3B, so Qwen3.6 is the latest 35B-A3B; a Qwen4 successor could arrive.
2. **Gemma 4 26B-A4B-it** (Apache-2.0, 2026-03, vision, 262K, sliding-window + MoE). Est. 5.5 / 6.9 GB, versus ~15 GB for Google's own QAT int4. It has the highest demand of any open LLM surveyed (12.7M/30d). Fit is moderate:
   - There is no linear attention, so the GDN-specific parts of the pipeline do not apply.
   - The large tied 262K vocab (0.74B embedding params) should stay at higher precision.
   - It is a non-Qwen showcase, and no ternary exists.
3. **Qwen3.5-122B-A10B** (Apache-2.0, 2026-02, same arch). Est. 26.6 / 33 GB: that puts a 122B model on a 32 GB Mac or 2x16 GB and is comfortable at 48 GB. Same-arch reuse applies, but a ~250 GB BF16 teacher needs multi-GPU or offline logits. Prism's "several hundred billion" plan may land near here, so there is a collision risk.
4. **Gemma 4 31B-it dense** (Apache, 9.9M/30d). Est. 6.6 / 8.3 GB. It is valuable but overlaps Bonsai 2 27B's niche (a dense ~30B in ~6 GB). Do it as a follow-on to #2, which shares the architecture plumbing.
5. **Nemotron 3.5 Lightning 30B-A3B** (OpenMDW, 2026-08, text, Mamba-2 hybrid MoE, 262K). Est. 6.7 / 8.4 GB. It opens a Mamba-hybrid line Prism lacks, but the architecture differs from GDN (SSM in/out projections, conv1d), so reuse is lower. Demand is moderate (0.57M BF16 + 0.81M NVFP4).
6. **GLM-4.7-Flash** (MIT, 2026-01, 31.2B MoE, 64 experts top-4, text). Est. 6.6 / 8.3 GB. Popular (1.68M), but text-only and without linear attention.
7. **gpt-oss-20b** (Apache, 6.3M/30d). Est. ~4.4 GB versus ~13 GB in MXFP4, which makes a phone-class model with strong demand. The catch is that the "teacher" is itself MXFP4, and the native low-bit release makes the gain smaller (3x versus 9x). **gpt-oss-120b**: est. ~25 GB versus ~61 GB.
8. **Mistral Small 4 119B** (Apache, vision, 1M ctx, 128 experts). Est. ~25 GB. It is low on demand (67k), so impact is limited.
- **Not recommended for a single-GPU team:**
  - DeepSeek-V4-Flash/V4.1-Flash (MIT, 291B/763B) and GLM-5.3-Flash (MIT, 321B, which does have linear attention and would reuse the most): out of budget, and likely Prism's next territory.
  - Kimi-K3: custom license, 2.78T.
  - Qwen3.8-Flash-Next: Qwen Community License, the 51B n-gram table, and incomplete llama.cpp support.
  - Llama 4: gated, and low demand.

### Ranked shortlist (a): cheap experiments that transfer
1. **Qwen3.5-0.8B -> Qwen3.5-2B -> Qwen3.5-4B -> Qwen3.5-9B** (all Apache, Feb 2026, vision, 262K, the identical `qwen3_5` GDN hybrid as Qwen3.8-27B; Base checkpoints exist). Est. ternary sizes: 0.19 / 0.48 / 0.99 / 2.05 GB. A 9B in ~2 GB is itself a useful phone model, and Prism's gen-1 8B is text-only. Recommendation: replace the planned "1.7B" run with **Qwen3.5-2B** unless the 1.7B is meant to replicate Prism's gen-1 Qwen3-1.7B. Qwen3-1.7B is pure attention and does not exercise GDN. Mind the tied-vs-untied embedding switch between 4B and 9B.
2. **Gemma 4 E2B / E4B** (Apache, vision, 131K): cheap probes for Gemma-specific issues (sliding window, PLE embeddings, huge vocab) before committing to 26B-A4B. Official Q4_0 QAT baselines exist for comparison.
3. **LFM2.5-1.2B / 2.6B / 8B-A1B** (LFM Open License 1.0, which is non-Apache, so check the commercial terms). Short-conv hybrids with strong on-device GGUF demand (2.6B GGUF 1.04M/30d). The 8B-A1B is a cheap **MoE** testbed (est. ~1.8 GB ternary) for routing sensitivity before Qwen3.6-35B-A3B.
4. **Nemotron-3-Nano-4B** (Nemotron OML, 2.75M/30d): a cheap Mamba-2 hybrid probe if #5 in list (b) is pursued.
5. **SmolLM3-3B / Olmo-3-7B** (Apache, fully open data): controls with public pretraining data, for clean ablations. Neither is hybrid GDN, so transfer to Bonsai-style targets is weaker.

### Inferences
- **Sweet spots where ternary unlocks something new:**
  - 30-36B MoE into 6-10 GB: a 16 GB laptop or 12 GB phone gets frontier-ish quality at 3-4B-active speed.
  - 120B-class MoE into 25-33 GB: runs on 32 GB Macs or a single 32 GB GPU.
  - 9B into ~2 GB: any phone.
  - Dense 27-31B into ~6 GB is already shown by Prism.
- **There are no 70B-class dense targets worth pursuing.** Llama-3.3-70B (Nov 2024) is old, and the 2026 vendors moved to MoE at that scale. The 70B-into-16GB case is better served by a 122B-A10B MoE (~27 GB) or by 35B-A3B (~8 GB).
- MoE expert weights dominate memory in all MoE candidates: 256 experts x 40 layers for Qwen3.6-35B-A3B. So the 9x-class compression applies to nearly all parameters, while the router, attention and embeddings can stay at higher precision for a small size cost. This follows from the config structure; I did not measure it.

### Gaps
- No direct r/LocalLLaMA thread was found requesting ternary MoE or Gemma Bonsai variants; web searches for this returned nothing relevant. Demand evidence above is HF downloads only.
- No published ternary/QAT results on MoE routing sensitivity for Qwen3.6-35B-A3B or Gemma 4 26B-A4B were found.
- Exact active-parameter counts for Qwen3.6-35B-A3B and Gemma 4 26B-A4B come from naming and a secondary source, not from computing them from the configs.
- LFM Open License 1.0 and the NVIDIA Nemotron OML terms (revenue thresholds, redistribution of derivatives) were not read in full.

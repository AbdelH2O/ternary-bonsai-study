# Bounded quantization-aware distillation on Qwen3.5-0.8B — 2026-10-02

**Update 2026-10-03:** the frozen chat-aware data amendment is reported in [QAT08_CHAT.md](QAT08_CHAT.md).

**Decision (frozen rules):**
- **Training helps, strongly.** After 65.5M tokens of distillation, the free-zero-count ternary student is **−1.02 nat/token** better than the best one-shot arm (GPTQ) on twelve held-out books [−1.09, −0.94]. That closes **66%** of the GPTQ-to-FP gap.
- **Bonsai 2's 42-zero budget no longer costs after training.** The 42-zero student is slightly *better*, −0.025 [−0.035, −0.016] ("costs little"). The same budget cost +0.45 nat/token under one-shot reconstruction.
- **Raw-text capability returns.** Retrieval accuracy is 94.5% and 90.5% against 55% for GPTQ.
- **Chat-format capability does not return.** On MMLU-Redux both trained students answer "A" almost every time (22.5%, chance), and on GSM8K they loop or echo the prompt. The frozen capability criterion is not met.
- The frozen rental rule is met (training helps, closure ≥ 0.5). Recommendation: fix the chat gap at 0.8B first (see below), then rent for 1.7B.

This follows the [1.7B Phase 1 comparison](PHASE1_17B.md), which found Prism's quality likely comes from end-to-end training. The [protocol](results/qat08/protocol.json) (sha `99897718…`) fixed the recipe, arms, budget, evaluation sets and decision rules before training. The learning rate came from a pre-freeze probe on training-domain data only. Evidence tags: 🟢 measured; 🟡 inferred; 🟣 proposed.

## Recipe

| | |
|---|---|
| Model | Qwen/Qwen3.5-0.8B (`2fc06364`), hybrid: 18 Gated DeltaNet and 6 full-attention blocks; folded H512 basis |
| Trainable | The 150 decoder projections (497M weights) as FP32 latents; every forward uses their PQ2_0 ternary projection through a straight-through estimator, on rotated activations: exactly the runtime function |
| Frozen | Tied embedding (a fixed ternary projection under the arm's rule, as in Prism's 1.7B embedding); norms, gates, conv, A_log, dt_bias at FP (byte-identical to the GGUF template) |
| Arms | **Free:** absmean rule (BitNet b1.58; scale = mean \|w\| per 128-group). **42-zero:** top-86 magnitudes, scale = their mean (Bonsai 2's budget) |
| Teacher, loss | FP model in BF16; forward KL over the full 248K vocabulary at every position |
| Data | 65.5M tokens of FineWeb-Edu (one pinned shard), 1,024-token sequences, same order for both arms |
| Optimization | AdamW with BF16 moments, β = (0.9, 0.95), peak LR 1e-4 (probe: 2e-5 → 1.26, **1e-4 → 1.13**, 5e-4 → 2.80 monitor KL after 120 steps), 50 warmup steps, cosine to 10%, clip 1.0; 2,000 steps × 32,768 tokens |
| Hardware | RTX 5070 12 GB, about 3,000 tokens/s, 7.9 GiB peak; flash-linear-attention 0.5.2 kernels; about 5.5–6 h per arm |

🟢 The [self-test](qat_08b.py) confirmed:
- rotated-activation identity to 3×10⁻⁷ for every width;
- exact PQ2_0 pack/decode for both rules;
- exactly 42 zeros per group for the budget rule;
- an identity STE gradient.

Exports reuse the frozen quantizer, and every file passed the decode assertion.

## Results

🟢 **Held-out** ([results.json](results/qat08/results.json); 12 fresh Gutenberg books, 5,330 MMLU-Redux items, 1,319 GSM8K items, 200 registries; no evaluation case has majority 13-gram overlap with the training tokens):

| Arm | Books NLL | Retrieval acc. / margin | MMLU-Redux | GSM8K |
|---|---:|---:|---:|---:|
| FP | **3.284** | **100%** / 8.41 | **48.2%** | **55.8%** |
| GPTQ (best one-shot) | 4.815 | 55% / 0.28 | 22.5% | 0.1% |
| Absmean, untrained | 10.564 | 52% / 0.01 | 22.5% | — |
| Top-86 (42 zeros), untrained | 10.836 | 52% / 0.12 | 22.5% | — |
| **Absmean, trained** | 3.799 | 94.5% / 3.44 | 22.5% | 1.1% |
| **42-zero, trained** | **3.773** | 90.5% / 3.25 | 22.5% | 0.2% |

The trained arms are better than GPTQ on every one of the 12 books (by 0.8–1.2 nat/token) and remain +0.49 to +0.52 above FP (intervals [+0.44, +0.56]).

🟢 **Validation token curve** (three validation books; descriptive):

| Tokens | 0 | 8M | 16M | 33M | 65.5M |
|---|---:|---:|---:|---:|---:|
| Absmean | 10.52 | 4.46 | 4.30 | 4.09 | 3.905 |
| 42-zero | 10.82 | 4.44 | 4.30 | 4.05 | 3.890 |

For reference: FP 3.396 and GPTQ 4.812. Both arms pass GPTQ within the first 8M tokens and were still falling by 0.18 between 33M and 65.5M tokens. The training-domain monitor KL was 0.385 (absmean) and 0.365 (42-zero) at the end.

🟢 **Why MMLU and GSM8K fail.**
- **MMLU.** FP puts 97% of its next-token probability on the four letters and spreads its answers. Every quantized arm, trained or not, picks "A" for 99.5–99.9% of items. "A" is the correct answer for 22.5% of items, which explains the identical scores. The trained students put only 3–4% of their probability on any letter.
- **GSM8K.** All 1,319 outputs of each quantized arm hit the 1,024-token limit: repeated lines, echoed prompts, or code-switching into Chinese. The few "correct" items (14 for absmean, 3 for 42-zero) are last-number parsing coincidences in degenerate text. So the frozen "GSM8K > 0" condition is met only trivially, and the capability rule fails on MMLU regardless.
- 🟡 **Interpretation.** Distillation on plain web prose repaired the student's text-continuation behavior (books, raw-text retrieval) but never exercised the chat template, so chat-mode behavior wasn't recovered. Prism's 1.7B model chats well (GSM8K 74.7%), which suggests its training mix includes instruction- or chat-formatted data.

## Interpretation

- 🟢 **Training is the missing ingredient for loss.** One-shot reconstruction plateaued at +1.5 nat/token above FP. 65.5M tokens of quantization-aware distillation reach +0.5 and are still improving.
- 🟢 **The 42-zero format is not a handicap once trained.** This supports Bonsai 2's fixed budget being a free choice under training (efficiency, regular packing) rather than a quality sacrifice. At one shot it cost +0.45.
- 🟢 **Absmean (BitNet) rounding is a poor one-shot rule but a good training target.** Untrained it scores 10.6 nat/token, far worse than GPTQ; after training it is far better.
- 🟡 **Chat capability needs chat data.** This is the main gap between our student and a usable model, and the clearest recipe lesson so far.
- Remaining gap to FP (+0.5 on books) is larger than Prism's at 1.7B (+0.22), with a much smaller budget than Prism likely used and frozen non-projection tensors. The curve has not flattened.

## Next steps 🟣

1. **Chat-aware distillation amendment (0.8B, local, about 6 h):** mix teacher-generated chat responses (thinking off) to instruction prompts into the training stream, e.g. 25–50%. Re-test MMLU and GSM8K under the same frozen harness. If capability returns, the recipe is ready to scale.
2. **Then rent for 1.7B** with Prism's Ternary-Bonsai-1.7B as the direct yardstick: about $100–250 on one H100 for both rules at about 1B tokens. The frozen rental rule is met.
3. Optional ablations: unfreezing norms and gates, and a longer token budget.

**Limits:**
- one 0.8B model, one recipe and seed, 65.5M tokens;
- non-projection tensors frozen;
- one learning-rate probe on 120 steps;
- logit-scored MMLU and one GSM8K prompt;
- intervals treat books, items and registries as independent;
- nothing here identifies PrismML's recipe.

## Incidents

- **Probe launch** (pre-freeze): two `pkill` invocations matched their own command lines and killed their shells; nothing was lost. The protocol freeze was restarted once after replacing a slow `np.isin` contamination lookup with a sorted search; no protocol had been written.
- **Training crash:** the free arm's process tree was killed at step about 1,140, apparently when an earlier session ended despite `setsid nohup`. It resumed from the step-1,087 checkpoint as a systemd user service; replayed steps reproduced the logged losses to 4–5 significant figures.
- **Pause:** training paused overnight after the free arm finished, before the 42-zero arm started (no progress lost). The 42-zero arm ran in one session.
- **Environment:** `flash-linear-attention` 0.5.2 was installed into `.venv` (lockfile untouched).

## Reproduction

From `analysis/bonsai2/replication`, with the project venv and CUDA:

```bash
python3 prepare_qat08_data.py                     # FineWeb-Edu shard -> work/qat08/data
python3 qat_08b.py selftest
python3 qat_08b.py probe --rule absmean --lr 1e-4 --steps 120     # x3 rates, pre-freeze
python3 freeze_qat08.py                           # already run; refuses to overwrite
work/qat08/run_train.sh                           # both arms, resumable (ran as systemd user unit)
python3 score_qat08.py export && python3 score_qat08.py curve && python3 score_qat08.py heldout
python3 score_qat08.py gsm8k && python3 score_qat08.py analyze
```

Training logs are in `work/qat08/<arm>/train.jsonl`, latents in `work/qat08/<arm>/latent_*.pt`, exported GGUFs in `work/qat08/`. All are git-ignored.

# QAT17B design draft v0.2 — targeted quantization-aware distillation of Qwen3-1.7B on Modal — 2026-10-06

**Status: draft v0.1 for the user's review. Nothing here is frozen or approved.** v0.1 added the proposals from [PRISM_FORENSICS_17B.md](PRISM_FORENSICS_17B.md) and [CODE_DRIFT.md](CODE_DRIFT.md), marked *(v0.1)*. v0.2 makes matching Prism's results the goal, marked *(v0.2)*. See the revision history at the end. It follows the frozen recommendation `rent_1p7b_with_recipe` (arm M) from [QAT08_MCU.md](QAT08_MCU.md), which authorized no rental. The user chose Modal's free Starter credits on 2026-10-06. Before training, this draft becomes a frozen `design.json`, then a sealed `protocol.json` after data generation, and each needs its own approval, as in QAT08_MCU. Evidence tags: 🟢 measured; 🟡 inferred; 🟣 proposed.

## Goal and the one-run strategy

**Question.** Can quantization-aware distillation with targeted teacher data bring a ternary Qwen3-1.7B, in Prism's exact PQ2_0 format, close to Prism's Ternary-Bonsai-1.7B on the Phase 1 harness?

*(v0.2)* **The goal is to match Prism's results, not to reproduce Prism's model.** This is Question B of [REPLICATION_PLAN.md](REPLICATION_PLAN.md) §1. Question A, recovering Prism's process, can't be verified: Prism hasn't released its latent weights, recipe, data or token count, so forensics can rule methods out but never confirm one. The user chose this framing on 2026-10-06.

**Fixed, so the comparison is fair:**
- **Base model:** Qwen/Qwen3-1.7B@`70d244cc`, Prism's ancestor.
- **Format and size:** ternary weights in groups of 128 with FP16 scales (PQ2_0, 2.125 bits per weight packed), every projection and the tied embedding ternary, norms in F32. That's Prism's exact tensor policy ([PHASE1_17B.md](PHASE1_17B.md)), run by the same pinned runtime.
- **Evaluation:** both models scored by the same harness, with the same prompts and token IDs. Prism's published numbers are only a harness check.

**Free, chosen for results:** method, data, loss, rotation, learning rate and schedule. Prism's own choices matter only as hypotheses that might improve results: e.g., the weight-movement lever from [CODE_DRIFT.md](CODE_DRIFT.md) and the forensics. Ingredients Prism's 1.7B didn't use are allowed; the H1024 rotation is one.

**Scope:** Prism's ternary 1.7B (generation 1). Matching Bonsai 2 27B is far beyond this project's compute, and no claim about it follows from this run.

The Phase 1 gaps to close (🟢 [PHASE1_17B.md](PHASE1_17B.md)):

| | FP | Prism (yardstick) | Best one-shot (GPTQ-H1024) |
|---|---:|---:|---:|
| Book NLL | 3.067 | 3.285 | 4.591 |
| MMLU-Redux | 49.0% | 47.7% | 23.5% |
| GSM8K | 82.0% | 74.7% | 0.0% |

**Strategy for few iterations.** Compute is limited, so every failed run is expensive. The design therefore:
1. **Bets on one long run, not several arms.** Arms that differ in one factor answer questions, but we can't afford them. More tokens and capability-targeted data are the two levers the 0.8B work showed to help. One run combines both and saves checkpoints along the way, so the token curve comes from the same run.
2. **Makes the run extendable.** A warmup–stable–decay schedule saves the state before the decay. If the run is still improving, a follow-up run continues from that state rather than starting over.
3. **Moves risk to free local checks.** Before any credit is spent on training, local checks cover the trained function, the exporter, the data, resume and the evaluation harness. A $0.50 Modal dry run then checks the Modal side.
4. **Defines abort rules.** A run that is clearly failing stops early and keeps the remaining credit.

## What carries over from 0.8B, and what changes

🟢 Unchanged from the frozen 0.8B recipe (`qat_08b.py`, sha `c2142acc…`):
- the projections are trained as FP32 latents through a PQ2_0 quantizer with a straight-through estimator, on rotated activations, which is exactly the runtime function;
- the tied embedding is a fixed ternary projection; norms stay frozen at FP;
- the teacher is the FP model in BF16, with forward KL over the full vocabulary at every position;
- AdamW with BF16 moments, betas (0.9, 0.95), gradient clip 1.0; 32,768 tokens per step (32 × 1,024);
- the student starts from folded FP weights.

| Change | 0.8B (QAT08_MCU) | 1.7B (proposed) | Reason |
|---|---|---|---|
| Model | Qwen3.5-0.8B hybrid | Qwen/Qwen3-1.7B@`70d244cc`, dense | Prism's yardstick ancestor |
| Basis | H512 | **H1024** | 🟢 Phase 1 fold check; rotation mattered more at 1.7B (5.00 vs 11.88 validation NLL) |
| Quantizer | top86 (42 zeros per group) | **absmean, free zero count** for projections and embedding. *(v0.1, made optional in v0.2)* Rounding each scale through BF16 matches Prism's stored scales exactly, but it serves file-level fidelity, not results. Default: off (FP32 mean → FP16, as in the frozen 0.8B code); on only if the pre-flight shows no difference | 🟢 Prism's 1.7B is gen-1: about 38% zeros with varying per-group counts, and its embedding matches absmean in 99.83% of entries. 🟢 All 13.4M Prism ternary scales are exact BF16 values ([forensics](PRISM_FORENSICS_17B.md) finding 2). 🟢 At 0.8B, free vs 42-zero made no material difference after training (−0.025 nat) |
| Tokens | 65.5M (2,000 steps) | **131.1M (4,000 steps)**, extendable | 🟢 Book loss was still falling at 65.5M; continued training gained GSM8K (+3.9) |
| Schedule | cosine to 10% | **warmup–stable–decay**: 100 warmup steps, constant to step 3,200, linear decay to 10% by step 4,000; state saved at 3,200 | Allows extension without a rerun |
| Learning rate | 1e-4 (probed) | **probed on Modal**: *(v0.1)* {5e-5, 1e-4, 2e-4, 4e-4}, 300 steps, ties broken toward the larger rate | 🟢 The 1.7B start is further from the teacher: KL 12.8 vs 8.3 at step 0. 🟢 Prism's training moved weights far more than ours ([CODE_DRIFT.md](CODE_DRIFT.md), [forensics](PRISM_FORENSICS_17B.md) finding 4), and short probes penalize larger rates |
| Data | 68.75% FineWeb, 25% chat, 6.25% multiple choice | **adds a math slot and more teacher data** (below) | 🟢 GSM8K stayed at 16% at 0.8B; the data had no math-focused teacher responses |
| Hardware | local RTX 5070 | Modal H100 80 GB; micro-batch 8 × 4, no gradient checkpointing, cached ternary weights | 🟢 [profile](work/qat17b/profile_NVIDIA_H100_80GB_HBM3.json): 2.63 s/step, 53 GiB peak |

🟢 The weight cache (`qat_17b.py`) recomputes each ternary weight once per optimizer step instead of every forward pass. Forward value and gradient were bit-identical to the frozen path on the local GPU. On the H100 profile, the losses matched to every printed digit, with gradient norms agreeing to about 4 significant figures. It is 18% faster.

## Training data (🟣 proposed; counts are targets for the data phase)

Every step holds 32 sequences of 1,024 tokens in a fixed slot layout, as in arm M:

| Slot | Sequences per step | Share | Tokens over 4,000 steps | Unique target | Passes |
|---|---:|---:|---:|---:|---:|
| FineWeb-Edu prose | 14 | 43.75% | 57.3M | 57.3M (no repeats) | 1.0 |
| General chat (teacher answers) | 8 | 25.0% | 32.8M | ≥ 10M | ≤ 3.3 |
| Math (teacher answers, filtered for correct final answers) | 6 | 18.75% | 24.6M | ≥ 8M | ≤ 3.1 |
| Closed-book multiple choice (teacher answers) | 2 | 6.25% | 8.2M | ≥ 2.5M | ≤ 3.3 |
| In-context reading (answer in the passage) | 2 | 6.25% | 8.2M | ≥ 5M | ≤ 1.7 |

Why this mix:
- 🟢 **Prose keeps the book loss.** At 0.8B, each step away from prose cost little (M paid +0.031 nat for its multiple-choice slot). Prose stays the largest slot, and over 131M tokens it still gets 57M, nearly C's 49M.
- 🟢 **Formatted teacher data brings capability; prose alone does not.** Only the chat slot moved GSM8K (0.2% → 14.6%), and only the multiple-choice slot moved MMLU (+8.6 points).
- 🟡 **Math is the biggest gap and had no targeted data.** At 0.8B, math came only from 1,500 Daring-Anteater prompts. GSM8K is the metric furthest from Prism (−74.7 points one-shot), and Prism lost only 7.3 points to FP there.
- 🟢 **Binding failed (82% vs the 90% floor).** So the in-context slot doubles: 6.25% alone instead of sharing a 6.25% slot with ARC.

### Sources and licences

| Slot | Source | Licence | Use |
|---|---|---|---|
| Prose | `HuggingFaceFW/fineweb-edu`, new shards past those used before | ODC-By 1.0 | Qwen3-tokenized, documents joined with `<|endoftext|>` |
| Chat | `nvidia/Daring-Anteater@ae79f8ac`, synthetic subsets: the 10,000 prompts already screened, plus about 30,000 more from the 72,886-prompt `synthetic_conv` pool | CC BY 4.0 | Prompts only; original answers discarded |
| Math | `openai/gsm8k` train (7,473, minus a 500-item dev hold-out); MATH train (`EleutherAI/hendrycks_math`, 7,500); Daring-Anteater `synthetic_math` (2,999) | MIT; MIT; CC BY 4.0 | Prompts and gold answers only |
| Multiple choice | ARC-Easy/Challenge train (`allenai/ai2_arc@210d026f`, the 6,714 QAT08_MCU items in two option orders); CommonsenseQA train (`tau/commonsense_qa`, about 9,700) | CC BY-SA 4.0; MIT | Prompts; gold letter for balance checks only |
| In-context | FineWeb-Edu passages from a range not used for prose, built by the QAT08_MCU in-context item builder (`incontext_items`, a blanked passage sentence with four options), adapted to Qwen3 token IDs; about 15,000 items | ODC-By 1.0 | |

🟣 Licences and revisions are re-read from each pinned dataset card at prompt preparation and stored in the record, as in QAT08_CHAT. Any card that disagrees with this table stops preparation.

### Teacher answers

- **Teacher:** FP Qwen3-1.7B, served by the pinned runtime (`prism-b10735-842b188`) from a BF16 GGUF converted from the pinned checkpoint. It's rendered with the Qwen3 chat template, thinking off (`enable_thinking=False`), exactly as `gsm8k_17b.py` renders evaluation prompts. 🟡 BF16 matches the BF16 teacher used in training. A pre-flight check compares it to the F32 GGUF.
- **Chat, multiple choice, in-context:** greedy, up to 896 new tokens.
- **Math:** four samples per prompt at Qwen's recommended non-thinking settings (temperature 0.7, top-p 0.8, top-k 20), plus one greedy sample.
  - Keep a sample only if its final answer, parsed with the frozen boxed/last-number parser (GSM8K) or a boxed-answer match (MATH), equals the gold answer.
  - Deduplicate exact duplicates; keep at most 3 correct samples per prompt.
  - 🟡 Rejection sampling gives several correct reasoning paths per problem. KL to the teacher still trains the teacher's full distribution on those contexts, so filtering only chooses which contexts the student sees.
- **Filters for every response:** ends naturally; non-empty; no special or thinking tokens beyond the template's empty thinking block; full transcript ≤ 1,024 tokens; no 13-word span shared with any evaluation item. Token-ID equality between packed transcripts and the template is checked on all of them, not just a pilot.
- **Volume and cost:** about 29M generated tokens (chat about 16M, math about 11M, the rest about 2M). Generation runs on the local RTX 5070, free and on the pinned runtime. A 256-prompt pilot measures throughput first. 🟣 If the projection exceeds 8 hours, the alternative is the same model under vLLM on Modal for about $1–2; switching runtimes is a deviation that needs approval.
- **Stop rules:** data preparation stops before training if any slot misses its unique-token target, if a slot would exceed 4 passes, or if teacher accuracy on the GSM8K dev hold-out is under 75%. That last one is a teacher-harness check: 🟢 FP scores 82.0% on the test set.

### Evaluation hygiene

- **GSM8K dev:** 500 seeded GSM8K train items are held out of every stream. They track the GSM8K trajectory at checkpoints without touching the test set.
- **Prompt screening:** every prompt and response is screened against all evaluation texts (MMLU-Redux, MMLU-fresh, GSM8K test, binding, retrieval, books): exact match, any shared normalized 13-word span, and TF-IDF cosine cut at 0.6.
- **Seal audit:** before the seal, every evaluation case gets the rolling 13-token audit against the exact final stream. A case with majority overlap blocks the seal (the v2 lesson in QAT08_MCU).
- **No retrieval-format data:** the retrieval registries ("Name: code" lists) never appear in training data, so retrieval stays a clean diagnostic.

## Pre-flight: free local checks, then one cheap Modal dry run

Each check must pass before the next stage. The local GPU checks run outside the bb sandbox (🟢 the sandbox hides the GPU), started by the user or as systemd user units.

| # | Check | Where | Pass condition |
|---|---|---|---|
| 1 | Self-test: rotated identity at widths 2,048 and 6,144; absmean pack/decode; STE gradient; weight-cache identity | local GPU | as in `qat_08b.py selftest`; 🟢 cache identity already passed |
| 2 | **Function equivalence.** Export the folded-init latents with the frozen exporter into the Phase 1 H1024 template. Compare the pinned runtime's NLL on the 6 validation windows with the PyTorch student's NLL on the same token IDs | local GPU | mean difference ≤ 0.03 nat/token. The runtime's quantized matmul path may round activations, so the exact tolerance is fixed from this init measurement before freezing; a larger gap is investigated, not waived. This guards the whole budget: training a function the runtime doesn't compute would waste every credit |
| 3 | Teacher equivalence: HF BF16 teacher vs F32 GGUF on the same windows | local GPU | ≤ 0.01 nat |
| 4 | Training-loop smoke test on a 2-layer truncation of the 1.7B: data mixer slot layout, monitors, WSD schedule, checkpoint files, export of the truncated model | local GPU | runs end to end |
| 5 | **Resume determinism:** 6 steps uninterrupted vs 3 steps, a kill, a resume, then 3 steps | local GPU (2-layer) | bit-identical latents and optimizer state |
| 6 | Data gates (above), the 13-token seal audit, and the slot layout read back from the packed stream | local CPU | all pass |
| 7 | **Modal dry run:** the real training entrypoint for 30 steps with absmean, a forced container kill at step 15, automatic retry and resume. Then export the step-30 latents locally and repeat check 2 on them | Modal H100, about $0.50 | step continuity; throughput within 10% of the profile; check 2 passes on trained latents |

## Modal execution

Provider choice and marketplace rules follow the rental guideline in [AGENTS.md](AGENTS.md#renting-gpu-compute-guideline-agreed-with-the-user-2026-10-06).

- **Storage:** inputs (base model, Hadamard manifest, names, streams of about 0.55 GB) live on the `bonsai-qat17b` volume, with checksums verified in Modal as the profile already did.
- **Resilience:** 🟢 Modal can preempt GPU functions at any time and restarts them on the same input ([Modal docs](https://modal.com/docs/guide/preemption)). So the training function:
  - saves an atomic resume state every 20 minutes (latents 5.6 GB plus moments 5.6 GB) and at steps 1,000 / 2,000 / 3,000 / 3,200 / 4,000;
  - commits the volume after each save;
  - runs with `retries=3`. A preemption loses at most about 20 minutes (about $1.40).
- **Checkpoints kept:** latents only at 16M / 33M / 65.5M / 98M / 131M tokens. The full resume state is kept at step 3,200 for extension.
- **Monitoring:** monitor KL every 200 steps, separately for prose, chat, math, multiple choice and in-context (16 held-out sequences each). It's logged to the volume and readable while the run is going.
- *(v0.1)* **Weight-movement diagnostics** every 500 steps, computed on the GPU from the latents (cheap, no export):
  - zero fraction;
  - median group scale relative to the ancestor's group mean |w| in the same basis;
  - nonzero sign agreement with the ancestor;
  - flip rate by ancestor-size bin;
  - codes differing from absmean rounding of the ancestor ([CODE_DRIFT.md](CODE_DRIFT.md)'s metric).

  🟢 Prism's values for reference: 39.9% zeros, scale ≈ 2×, 89.4% sign agreement, 39% codes changed. These are descriptive, not gates: movement is a correlate of Prism's quality, not a proven cause.
- **Abort rules** (stop and save state, so no further compute is spent):
  1. any non-finite loss or gradient norm;
  2. at step 400, prose monitor KL above the selected learning-rate probe's own value at step 150;
  3. any domain's monitor KL at step 1,000 higher than at step 400.

## Learning-rate probe (Modal, about $4)

- **Runs:** *(v0.1)* four probes at {5e-5, 1e-4, 2e-4, 4e-4}, **300 steps** each, from the folded init with absmean. Linear warmup over 20 steps, then constant. Data comes from the mixed stream at an offset past the training range. Each also logs the weight-movement diagnostics at steps 150 and 300.
- **Rule:** pick the lowest mean monitor KL across the five domains at step 300. *(v0.1)* Among rates whose KL is within 2% of the best, take the largest.
- 🟡 **Why less conservative than 0.8B:** the 0.8B probe ran 120 steps and picked 1e-4 (2e-5 → 1.26, 1e-4 → 1.13, 5e-4 → 2.80). [CODE_DRIFT.md](CODE_DRIFT.md) notes that short probes penalize larger rates, while Prism's weights moved far more than ours. A longer probe and a tie-break toward movement is the cheapest test of that lever; 4e-4 may still lose, as 5e-4 did at 0.8B.

## Evaluation (local GPU, pinned runtime)

- **Harness:** the Phase 1 harness with unchanged case hashes:
  - 12 held-out and 3 validation books;
  - MMLU-Redux 2.0 (5,330 items, letter logits, thinking off);
  - GSM8K test (1,319 items, greedy, 1,024 tokens, boxed or last-number parsing);
  - retrieval (200 registries).
  
  🟢 FP and Prism outputs from Phase 1 are reused after a hash check.
- **Added sets:** MMLU-fresh (2,998 items) and binding (400 items) from QAT08_MCU, re-rendered with the Qwen3 template. FP, Prism (no YaRN) and the student are all scored on them.
- *(v0.2)* **Breadth sets, so "matched" isn't a four-metric claim:** two of Prism's own published benchmarks, scored for FP, Prism and the student through the same pinned runtime, thinking off, greedy.
  - **IFEval** (541 prompts; Google's instruction-following checker, Apache 2.0): prompt-level strict accuracy as the headline, the other three IFEval accuracies reported.
  - **HumanEval+** (164 problems, EvalPlus, Apache 2.0): pass@1, greedy. 🟣 Model-written code is executed only inside a network-less, resource-limited container, never directly on the host.
  - Prism publishes 70.1 / 51.8 against Qwen3-1.7B's 70.3 / 57.3 (EvalScope + vLLM). Those values only check our harness; every comparison uses our own scores.
- **Decision checkpoint:** the step-4,000 checkpoint, fixed in advance; validation can't pick another. Earlier checkpoints get the validation books and the GSM8K dev set for the curve only.

### Decision rules (proposed for the user to adjust before freezing)

*(v0.2)* The top reading is named **matched**, not "replicated": it claims equal results under the fixed conditions above, not a reproduction of Prism's model.

Comparisons are against `prism_noyarn`: books by paired t over 12 books; items by paired normal intervals. All intervals are 95%.

| Reading | Condition |
|---|---|
| **matched** | Book minus Prism upper bound ≤ +0.10 AND MMLU-Redux minus Prism lower bound ≥ −3 points AND GSM8K minus Prism lower bound ≥ −3 points (the Phase 1 non-inferiority margins) |
| **strong_partial** | Not matched, AND GSM8K Wilson lower > 50% AND MMLU-Redux Wilson lower > 40% AND book minus Prism upper ≤ +0.30 |
| **partial** | Neither, AND the QAT08 recovered gate: MMLU-Redux Wilson lower > 30, GSM8K Wilson lower > 10, also with outputs that don't end naturally counted wrong |
| **not_recovered** | None of the above |
| **costs_ok** (reported with every reading) | Retrieval accuracy ≥ 90% AND binding uncalibrated accuracy ≥ 85% |

*(v0.2)* **Breadth is reported next to every reading and never omitted.** It is not a gate: 164 HumanEval+ problems can't support a 3-point non-inferiority margin. Each report gives the student-minus-Prism IFEval and HumanEval+ differences with paired 95% intervals. A claim worded "matches Prism" must also state any breadth metric whose interval lies wholly below −5 points.

**Extension rule** (for a follow-up run, not this run). If the reading isn't `matched`, and from 65.5M to 98M tokens (both in the stable phase) either validation book NLL falls by ≥ 0.02 or GSM8K dev rises by ≥ 3 points, recommend continuing from the step-3,200 state: +2,000 stable steps, then a new 800-step decay, about $9. Otherwise recommend a data change, not more tokens. *(v0.1)* The report also places the weight-movement trajectory against Prism's values, so the recommendation says whether the run is plausibly short of Prism's training regime.

## Cost estimate (Modal; local GPU work is free)

| Item | Estimate |
|---|---:|
| Profile, done | about $1 (🟡 check the Modal dashboard) |
| Learning-rate probe, 4 runs × 300 steps *(v0.1)* | about $4 |
| Dry run with forced preemption | $0.50 |
| Main run: 4,000 × 2.63 s = 2.92 h, plus monitors and checkpoints, about 3.1 h, at $3.95/h H100 plus memory | about $13 |
| Reserve for one or two preemptions | $3 |
| **Total** | **about $22** |

🟣 Throughput was profiled with top86; absmean skips the sort, so it should be no slower. The dry run measures it.

## Risks, stated in advance

- 🟡 **The token budget is small for ternary training.** Published ternary training work uses far more tokens; Prism's budget is unknown. Full non-inferiority to Prism in 131M tokens is unlikely. The design aims for the largest capability recovery per token, and the extension rule decides whether to keep going.
- 🟡 **Math data is the main untested ingredient.** If GSM8K moves and MMLU doesn't, that splits the remaining work cleanly. If neither moves, the most likely explanation is that the token budget is too small, not the data.
- 🟡 **Prose share fell from 68.75% to 43.75%.** The absolute prose token count is about the same as at 0.8B, but the book result may lag. Prism's own +0.22 nat book gap to FP leaves room.
- 🟡 **One run, one seed.** Readings are about this run, not the recipe's variance.
- 🟡 **The teacher's mistakes are distilled.** Only math is filtered for correctness.

## Open choices for the user

1. **Token budget:** 4,000 steps (about $13, leaves room for a retry) vs 6,000 steps (about $20, less room for a retry).
2. **Thresholds:** the decision-rule thresholds, especially `strong_partial` (GSM8K > 50%, MMLU > 40%).
3. **Rejection sampling:** whether rejection-sampled math is acceptable, or math should be greedy only (fewer unique tokens, about 1.5× more repetition).
4. **Teacher generation runtime:** local pinned runtime (default, free, slower) vs vLLM on Modal (fast, about $1–2, different runtime).
5. *(v0.1)* **Teacher:** keep Qwen3-1.7B. [Forensics](PRISM_FORENSICS_17B.md) finding 7 found Prism answers like neither Qwen3-4B nor 8B, so a larger teacher isn't indicated. Prism's targets likely included non-Qwen-style responses, perhaps hard labels. Adding a hard-label slot would be a new, untested ingredient; the default is not to. *(v0.2)* Under the matching goal it's allowed if a cheap 0.8B test shows it raises scores; matching Prism's style is not a reason to add it.
6. *(v0.1)* **Multiple-choice share:** the running 0.8B MC12 experiment ([QAT08_MC12_DESIGN.md](QAT08_MC12_DESIGN.md)) tests 12.5% multiple-choice data. Its result should set the 1.7B multiple-choice slot before this design freezes.

## Next steps after approval of this draft

1. Write `prepare_qat17b.py` (prompts, cases, teacher generation, packing, audits) and `qat_17b.py train` (mixer, WSD, abort rules, resume, retries).
2. Run pre-flight checks 1–6 locally.
   *(v0.2)* Add the IFEval and HumanEval+ harness (HumanEval+ in a network-less container), and score FP and Prism on it, before any student exists.
3. Freeze `design.json`, then generate data, pack, audit, and seal `protocol.json`; each stage needs the user's approval.
4. Run the learning-rate probe and the dry run on Modal.
5. Train, export, evaluate locally, and write `QAT17B.md` with the frozen decision first.

## Revision history

| Version | Date | Change | Basis |
|---|---|---|---|
| v0 | 2026-10-06 | First draft: one extendable 4,000-step run, math slot, absmean, H1024, WSD, pre-flight checks, Modal cost estimate | [H100 profile](work/qat17b/profile_NVIDIA_H100_80GB_HBM3.json); [QAT08_MCU.md](QAT08_MCU.md) |
| v0.1 | 2026-10-06 | Scales rounded through BF16; weight-movement diagnostics; longer, less conservative learning-rate probe with 4e-4; teacher stays Qwen3-1.7B; multiple-choice share waits for MC12 | [PRISM_FORENSICS_17B.md](PRISM_FORENSICS_17B.md); [CODE_DRIFT.md](CODE_DRIFT.md) |
| v0.2 | 2026-10-06 | Goal set to matching Prism's results (Question B), not reproducing its model; base model, format and evaluation fixed, method free; BF16 scale rounding made optional; IFEval and HumanEval+ added as reported breadth metrics; top reading renamed `matched`; scope limited to the ternary 1.7B | The user's decision ("Update the design along these lines"); [RESEARCH_LOG.md](RESEARCH_LOG.md), 2026-10-06 goal question |

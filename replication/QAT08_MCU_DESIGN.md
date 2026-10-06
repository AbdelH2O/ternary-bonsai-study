# QAT08-MCU design v2 — multiple-choice read-out data, trained norms/gates and continued training at 0.8B — 2026-10-04

**Status (2026-10-06): complete. Results are in [QAT08_MCU.md](QAT08_MCU.md). The frozen recommendation is `rent_1p7b_with_recipe` (arm M); it is not a rental authorization.** The design below is the historical frozen record (design v2 `8247ef1e…`, protocol `b99f4bdd…`).

- Frozen [design v2](results/qat08_mcu/design.json), sha `8247ef1ee2e87b1b0d55a857225587def0790f9a6f68f2e5e50b2fab295a44a9`, verified by `freeze_qat08_mcu.py verify-design`.
- It supersedes v1 (`2963624c…`), which is archived byte-for-byte in `results/qat08_mcu/versions/v1` ([MANIFEST.json](results/qat08_mcu/versions/v1/MANIFEST.json), sha `df6d4ced…`). The v1 protocol seal was blocked by the 13-gram gate (see "Revision v2" below).
- Arms: **M, U and B**. The optional MU arm is dropped by the predeclared budget rule.
- 🟢 Sealed [protocol](results/qat08_mcu/protocol.json) `b99f4bdd…`: only the protocol stage reran (118.6 s), with generation and packing reused unchanged. No evaluation case has majority 13-gram overlap with any arm's exact stream: C/U max 0.146 (MMLU-Redux), M max 0.400 (wrapper overlap on short items), B max 0.249. GPU-h: 1.87 measured (prep 1.48, data 0.39), 22.3 projected of 25.
- (Historical) The next gate was the user's approval of the sealed protocol (`protocol_approval.json`, given as "Proceed" after the seal). Training ran only after that, as the systemd unit `qat08-mcu-run`: `python qat08_mcu_control.py start --phase run`, then `pause` (finishes a step and saves; safe before a reboot), `resume` and `status`.

Spec: [review](reviews/REVIEW_QAT08_CHAT_opus55.md) sections 6–7; [plan](../../../docs/superpowers/plans/2026-10-03-qat08-mcu-experiment.md); [diagnostics](DIAG_READOUT.md). Independent prep review: [REVIEW_MCU_PREP_parent.md](reviews/REVIEW_MCU_PREP_parent.md) (pass within its scope). Evidence tags: 🟢 measured; 🟡 inferred; 🟣 proposed or projected.

## Why these arms

🟢 The read-out diagnostics ([DIAG_READOUT.md](DIAG_READOUT.md)) found the chat student C cannot copy an answer stated in its prompt: 27.2% calibrated, against FP's 99.25%. They also found no knowledge signal without letters.

The frozen rule's `harness_invalid` stop came from a calibration artifact in the FP floor. A disclosed post-results amendment ([decision_amendment.json](results/diag_readout/decision_amendment.json), sha `44dd024b…`) gives **`readout_broken` → M, U, B; optional MU**. B enters only on a weak letter trajectory.

## Hypotheses

Each arm is compared with C, the QAT08 chat student: same recipe, 65.5M tokens, top-86 rule.

- **H_M (read-out data):** training on teacher answers to multiple-choice prompts teaches the student to put its probability on the right letter.
- **H_U (trained norms and gates):** freezing the non-projection FP tensors (norms, gates, conv, A_log, dt_bias) at FP blocks recovery; training them helps.
- **H_B (more tokens):** C was still improving, so 2,000 more steps on unseen data help.

## Arms (🟢 from `design.json`)

| Arm | Stream | Trained FP tensors | Initialization | Schedule | Steps |
|---|---|---|---|---|---:|
| M | `work/qat08_mcu/data/train_M.u32`: C's mix with a multiple-choice slot | no | folded | cosine, peak 1e-4 | 2,000 |
| U | C's exact stream `work/qat08_chat/data/train.u32` | **yes**, from exact F32 checkpoint values, FP peak LR **1e-3** | folded | cosine, peak 1e-4 | 2,000 |
| B | `work/qat08_mcu/data/train_B.u32`: next unseen FineWeb and chat windows | no | C's `resume.pt` at step 2,000, with its optimizer state | constant 1e-5 | 2,000 |

Unchanged from C:
- the quantizer, rule top86, H512 basis and fixed ternary embedding;
- the teacher and all-position forward KL;
- AdamW with BF16 moments, betas (0.9, 0.95), clip 1.0;
- 32,768 tokens per step, checkpoints at 8M/16M/33M/65.5M, and the monitor.

Comparators (hashed): FP `604cb165…`, plain-text student qat_42 `06f90c31…`, C `80870d93…`.

## Preparation results (GPU, 1.48 GPU-h of the 2.0 ceiling)

🟢 **Self-test:**
- rotated-activation identity within 2.7×10⁻⁷ at every width;
- exact PQ2_0 pack/decode;
- top86 gives exactly 42 zeros per group;
- the 169 FP tensors at init are byte-identical to the GGUF template.

🟢 **Teacher pilot** (first 64 shuffled prompts; FP F32 GGUF, greedy, thinking off, 512-token limit):
- all 64 ended naturally; 2,919 tokens in 8.8 s, longest 431 tokens;
- 61 answers parsed, and 46 of 64 were correct.

  | Source | Prompt type | Correct |
  |---|---|---:|
  | ARC | letter only | 19/29 |
  | ARC | reason first | 4/5 |
  | In-context | letter only | 17/24 |
  | In-context | reason first | 6/6 |

- The 3 unparsed answers state a letter and then explain without "Answer: X".
- 🟡 The teacher's own answers are imperfect, about 72% here. The arms distill the teacher's distribution, as in C, not gold labels.

🟢 **FP-tensor learning-rate probe.** Each probe ran 120 steps from the folded init on training-domain data (offset 300,000 sequences), with projections at 1e-4. Each took 22.1 min.

| FP-tensor LR | 0 (frozen) | 1e-5 | 1e-4 | 1e-3 |
|---|---:|---:|---:|---:|
| Final monitor KL | **1.0086** | 1.0290 | 1.0279 | 1.0124 |

- 🟢 `unfreeze_helps_in_probe`: **false**. The frozen rule picks the best unfrozen rate, **1e-3**, for U.
- 🟡 The differences are 0.004–0.02 nats from one run each, 120 steps from an untrained start (KL 8.27 at step 0). That says little about U at 2,000 steps, and U stays in because the diagnostics require it.

## Training data (🟢 counts measured; 🟣 stream sizes are design targets for the data phase)

- **Multiple-choice prompts:** 12,714 items.
  - 6,714 are ARC train questions (CC BY-SA 4.0, two option orders each) and 6,000 are in-context items, whose answer is in a FineWeb-Edu passage (sequences 100,000–290,000).
  - 2,452 ask for short reasoning first; the rest ask for the letter only, in 5 paraphrases.
  - Gold letters are balanced: A 3,218, B 3,116, C 3,193, D 3,187.
- **Screen against all evaluation question text** (11,557 texts: MMLU-Redux, MMLU-fresh, binding, the diagnostics' dev letters, GSM8K):
  - 2 exact duplicates were dropped;
  - the highest TF-IDF cosine is 0.50 (cut 0.6), and the 99th percentile is 0.25.
- 🟣 **Data phase:**
  - teacher answers to all prompts;
  - responses are kept only if they end naturally, are non-empty, contain no special or thinking tokens, share no 13-word span with evaluation content, and fit 1,024 tokens;
  - the phase stops unless at least 5,000 are kept, with at least 1M unique tokens and at most 4 passes;
  - stream M holds 4.10M multiple-choice, 45.06M FineWeb and 16.38M chat tokens.
- 🟣 Before training, a 13-token rolling audit checks every evaluation case against each arm's exact stream; any case with majority overlap blocks the protocol.

## Fresh evaluation sets (🟢)

- **Books:** 12 held-out Gutenberg books (24 cases) and 3 validation books (6 cases). None appears by ID, file name or title anywhere in earlier experiments; the incident below explains why that check was rebuilt.
  - Validation: Nicholas Nickleby, North and South, The Woman in White.
  - Held out: Daniel Deronda, Barchester Towers, The Way We Live Now, Pendennis, The Pickwick Papers, Little Dorrit, Dombey and Son, The Old Curiosity Shop, Our Mutual Friend, Barnaby Rudge, Anna Karenina, Tom Jones.
  - An early check found no 13-token overlap between any of the 30 book cases and the FineWeb or chat training streams.
- **MMLU-fresh:** 2,998 MMLU test items never scored in this project: the frozen 3,000-item sample minus 2 contaminated items (see "Revision v2").
  - They are drawn from 7,099 eligible items.
  - Every MMLU-Redux 2.0 item is excluded, flagged ones included, by question, option tuple or 13-word span.
  - Gold letters: A 688, B 779, C 763, D 768.
- **Unchanged continuity sets:**
  - MMLU-Redux: 5,330 items;
  - GSM8K: 1,319 items;
  - retrieval: 200 registries;
  - binding (answer in the prompt): 400 new items.

## Decision rules (verbatim from `design.json`)

- **H_M:** M (or MU) minus C MMLU-Redux calibrated pairwise >= +4 points with paired lower bound > 0, AND arm MMLU-fresh pairwise lower bound > 50, AND binding calibrated accuracy >= 90
- **H_U:** U (or MU) minus C book NLL mean <= -0.05 with upper bound < 0, OR MMLU-Redux pairwise effect as above
- **H_B:** B minus C book NLL mean <= -0.05 with upper bound < 0, OR MMLU-Redux pairwise effect as above (budget effect)
- **recovered:** MMLU-Redux Wilson lower > 30 AND GSM8K Wilson lower > 10 AND GSM8K natural-EOS Wilson lower > 10 (unchanged QAT08_CHAT gate)
- **fresh_confirmed:** MMLU-fresh Wilson lower > 30
- **costs_ok:** book minus C upper bound <= +0.10 AND GSM8K minus C lower bound >= -3 points AND retrieval accuracy >= 90
- **format_only:** letter mass >= 0.5 and MMLU-Redux pairwise < 52: report as format learning, never as recovery
- **recommendation:** rent_1p7b_with_recipe if any arm is recovered AND fresh_confirmed AND costs_ok; else iterate_0p8b_mc_data if M or MU shows the pairwise effect; else rent_1p7b_token_ladder if the diagnostics branch was knowledge_lost AND B shows a knowledge signal (MMLU-Redux pairwise effect, or MMLU-fresh pairwise lower bound > 52) AND U/MU show no pairwise effect (a book gain alone never triggers rental); else iterate_0p8b if any arm shows an effect or book gain; else stop_data_iteration_0p8b

🟡 The diagnostics branch is `readout_broken`, not `knowledge_lost`, so the token-ladder rental cannot trigger in this experiment.

🟡 Binding calibrated accuracy in H_M uses the same mean-log-prob calibration that misfired on FP in the diagnostics. A near-perfect arm could therefore score a few points low. H_M needs 90, far below FP's 94.0 calibrated, so this matters only for an arm near that line; the report will also show uncalibrated binding.

## Budget (25.0 GPU-h ceiling, enforced by the supervisor)

| Phase | GPU-h |
|---|---:|
| Prep (🟢 used) | 1.48 |
| Data: generation, packing, protocol (🟣 ceiling) | 1.00 |
| Training, 3 arms × 6.5 (🟣 per-arm ceiling) | 19.50 |
| Evaluation, 3 arms × 0.15 + fixed 0.5 (🟣) | 0.95 |
| **Projected total** | **22.93** |

Adding MU would project to 29.58 h, so the rule dropped it. Training is expected to take about 5.5–6 h per arm, judging by C.

## Deviations (from `design.json`)

1. The diagnostics decision is read through the post-results amendment. The FP binding floor now uses uncalibrated accuracy. The original `decision.json` (`harness_invalid`) is kept and hash-bound.
2. The OpenBookQA licence is unknown, so ARC train supplies the closed-book multiple-choice items. Fresh knowledge is measured on MMLU-fresh instead of ARC test.
3. The protocol needs its own explicit user approval, which the controller never writes.
4. B continues C's saved optimizer state at a constant 1e-5 for 2,000 steps on the next unseen data.
5. transformers loads the 18 GDN norms in BF16, so C, M and B train on BF16-rounded norms while their GGUFs keep the exact F32 values. U starts from, and exports, the exact F32 values.
6. U clips gradients jointly over projections and FP tensors; C, M and B clip projections only.
7. U exports its 36 trained `in_proj_a`/`in_proj_b` gates in their BF16 template storage, a small rounding.
8. The 25 GPU-h ceiling covers prep, data and evaluation as well as training.
9. In-context items appear in a single option order; ARC items appear in two.
10. The item mix is about 48% in-context and 52% ARC, against the review's proposed 60/40.
11. The teacher limit is 512 tokens, against the review's 1,536.

## Incidents before the freeze

- 🟢 **Reused books.** The first book list reused 8 titles (Emma, Pride and Prejudice, Wuthering Heights, Middlemarch, Dracula, Frankenstein, Little Women, Monte Cristo) and one reserve (War of the Worlds).
  - Cause: the freshness check read only lists named `GUTENBERG`.
  - Found by an independent audit before any GPU stage or approval.
  - Fix: the check now scans every script, saved JSON, report and downloaded book under `analysis/` by ID, file name and title, with regression tests.
  - The first artifacts are archived under `results/qat08_mcu/rejected/` and `work/qat08_mcu/rejected/`. The record is `work/qat08_mcu/pre_freeze_incidents.json`, carried in `design.json`.
- 🟢 **Loader fix.** The independent prep review found that the diagnostics loader didn't hash the current plan. It now checks decision, results and plan provenance on every path; the fix was retested by the reviewer.
- 🟢 **Self-test accounting.** Before launch, the GPU self-test became the first prep stage, so its time counts toward the 2.0 h prep ceiling.

## Revision v2 (after the blocked v1 protocol seal)

- 🟢 **What happened.** The v1 data phase finished generation and packing, with every packing gate passing. Then the protocol seal's 13-gram gate blocked one MMLU-fresh item, `miscellaneous/7642/D`:
  - 0.517 overlap with C's stream (arms C and U) and 0.569 with arm M's stream; under 0.07 with B's.
  - Its sibling `miscellaneous/7932/C` quotes the same Zora Neale Hurston passage, at 0.376 and 0.414.
  - The passage is verbatim in C's stream at sequence 47,864, token 31, which is FineWeb sequence 35,898. That is inside M's FineWeb prefix and before B's range.
  - The cause was mine: MMLU-fresh was not checked against the existing streams before the v1 freeze.
- 🟢 **User basis.** The user said "Proceed with option 1". That approved, before any v2 hash existed, removing exactly these two items (3,000 → 2,998, no replacement), re-freezing, reusing the generated data after hash checks, and re-running only the protocol seal. It authorizes no training. The user has not seen or approved the v2 sha itself.
- 🟢 **Changes from v1.** Nothing else changed: no arm, data rule, gate, threshold, model, recipe, learning rate or budget.
  - MMLU-fresh case record: 2,998 items (`f450812b…`), the v1 lines minus the two IDs, order kept.
  - A new design-time 13-gram audit of every evaluation set against every existing training stream (qat_42's 65.5M-token prefix, C/U, M, B). A majority-overlap case blocks the freeze. Result: **0 majority cases in every set and stream**. In M, 4 MMLU-Redux and 1 MMLU-fresh items exceed 0.3, the highest 0.40. That is the shared instruction and chat-template wrapper on short items, not content.
  - The protocol seal now projects from all measured GPU time, including the blocked seal, and checks the reused data against their recorded hashes.
- 🟢 **Reused data.** The teacher responses, kept set, multiple-choice corpus and both streams are reused unchanged; nothing was regenerated.
  - Packing was re-run in an isolated temporary directory and reproduced all five outputs byte-identically (`mc_unique`, `kept_mc`, `train_M`, `train_B`, `data_record`).
  - The re-run multiple-choice screen is **not** byte-identical to the original prompt file. Only the per-row annotation `max_eval_cosine` differs (max |Δ| 0.0013), because the TF-IDF is fitted with 11,555 instead of 11,557 evaluation texts.
  - Every teacher input (IDs, order, `prompt`, `prompt_ids`) and every other field is identical. The screen's decisions are identical, and no row reaches the 0.6 cut in either version.
  - The byte-identity check (`revision_v2_reuse_check.json`, accepted=false) and the per-row diff are kept as recorded. A separate acceptance record (`revision_v2_reuse_acceptance.json`) accepts reuse on the actual training inputs and transcripts and is bound to both. The parent thread made this as an implementation ruling within option 1.
- 🟢 **GPU time so far:** 1.84 h (prep 1.48, data 0.36: generation 0.32, packing 0.005, blocked seal 0.03).

## Reproduction

From `analysis/bonsai2/replication`, with the project venv:

```bash
python prepare_qat08_mcu.py cases && python prepare_qat08_mcu.py mc        # CPU + network; sealed
python qat08_mcu_control.py start --phase prep                              # GPU: selftest, pilot, 4 FP-LR probes (done)
python freeze_qat08_mcu.py design && python freeze_qat08_mcu.py verify-design   # CPU (done)
python qat08_mcu_control.py start --phase data      # GPU <= 1.0 h, only after design_approval.json
python qat08_mcu_control.py start --phase run       # GPU, only after protocol_approval.json
python qat08_mcu_control.py pause | resume | status # pause finishes a training step and saves; resume after reboot
```

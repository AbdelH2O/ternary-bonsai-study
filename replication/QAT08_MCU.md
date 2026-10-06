# QAT08-MCU: multiple-choice read-out data, trained norms/gates and continued training on Qwen3.5-0.8B — results, 2026-10-06

**Decision (frozen rules): `rent_1p7b_with_recipe`, recipe arm M.** This is the predeclared *recommendation* under the frozen rule. It authorizes no rental: renting and the 1.7B run need their own user decision, budget and protocol.

- 🟢 **Arm M (multiple-choice teacher data) meets the frozen recovery gate.** It reaches 35.9% on MMLU-Redux (Wilson [34.6, 37.2]) and 35.2% on the never-scored MMLU-fresh set [33.5, 36.9], against 23.9% and 23.7% for the chat student C. GSM8K is 16.3% [14.4, 18.4], with 16.2% ending naturally.
- 🟢 **The recovery is partial.** M closes about half of C's gap to FP on MMLU-Redux (FP 48.2%) and about 56% on MMLU-fresh (FP 44.3%). On GSM8K it stays far from FP (55.8%); its gain over C, +1.7 points [−0.5, +4.0], is not significant. The frozen "recovered" gate is a low bar: an MMLU lower bound above 30 and GSM8K lower bounds above 10.
- 🟢 **H_M itself is false.** M meets its pairwise effect (+8.6 points vs C [+7.3, +9.8]) and fresh-set conditions (fresh pairwise lower bound 58.0 > 50). It misses the binding floor: 82.0% calibrated, 79.5% uncalibrated, against ≥ 90. So M fails H_M on either definition; this is not the calibration artifact that affected FP.
- 🟢 **Trained norms/gates (U) and continued training (B) show no MMLU effect** (pairwise −0.2 and −0.4 vs C) and no book gain. Both are flagged format-only: the letter mass is high but pairwise stays near chance. B is the only arm with a significant GSM8K gain over C, +3.9 points [+1.8, +5.9]. GSM8K is not part of H_B, though, and 18.4% is still far from FP.
- 🟢 **Costs are within the frozen allowances for every arm.** M pays a small but measurable book cost, +0.031 nat/token vs C [+0.018, +0.044], against the +0.10 allowance. Retrieval is 93.0% for M (C 94.5%).


Frozen per-arm flags (`decide_mcu`, reproduced exactly from `results.json`):

- 🟢 **M**: effect=yes, book_gain=no, costs_ok=yes, recovered=yes, fresh_confirmed=yes, format_only=no, H_M=no
- 🟢 **U**: effect=no, book_gain=no, costs_ok=yes, recovered=no, fresh_confirmed=no, format_only=yes, H_U=no
- 🟢 **B**: effect=no, book_gain=no, costs_ok=yes, recovered=no, fresh_confirmed=no, format_only=yes, H_B=no

Diagnostics branch: `readout_broken`. [Design](results/qat08_mcu/design.json), [protocol](results/qat08_mcu/protocol.json), [results](results/qat08_mcu/results.json). Evidence tags: 🟢 measured; 🟡 inferred; 🟣 proposed.

## 🟢 Recipe (from the frozen design v2 `8247ef1e…`)

Every arm starts from Qwen/Qwen3.5-0.8B (hybrid: 18 Gated DeltaNet and 6 full-attention blocks).
- Shared recipe: folded H512 basis; the 150 projections are trained as FP32 latents through the PQ2_0 top-86 (42-zero) quantizer.
- The teacher is the FP model with all-position forward KL.
- AdamW with BF16 moments, betas (0.9, 0.95), clip 1.0.
- 2,000 steps × 32,768 tokens = 65.536M tokens per arm.

| Arm | Training stream (per 16 sequences) | Projection LR | Norms/gates/conv/A_log/dt_bias | Start |
|---|---|---|---|---|
| C (comparator) | 75% FineWeb-Edu, 25% chat | cosine, peak 1e-4 | frozen | folded |
| **M** | **68.75% FineWeb-Edu (11/16), 25% C's chat rows (4/16), 6.25% multiple-choice (1/16)**: 4.096M multiple-choice tokens, 1.54 passes over 12,599 kept items | cosine, peak 1e-4 | frozen | folded |
| U | C's exact stream | cosine, peak 1e-4 | **trained**, cosine peak 1e-3, from exact F32 values | folded |
| B | C's layout continued on the next unseen FineWeb and chat windows | **constant 1e-5** | frozen | **C's final state and optimizer** |

Data sources:
- **Multiple-choice items:** ARC-Easy/Challenge train (`allenai/ai2_arc@210d026f`, CC BY-SA 4.0), plus in-context items built on FineWeb-Edu passages (sequences 100,000–290,000).
- **Corpora:** FineWeb-Edu (ODC-By 1.0); chat prompts from `nvidia/Daring-Anteater` (CC BY 4.0), preserved from the chat run.
- **Teacher answers:** the FP F32 GGUF, greedy, through the pinned Qwen3.5 chat template with thinking disabled, 512-token limit. Only naturally ended responses with no 13-word overlap with evaluation content were kept.

## 🟢 Held-out results (C = chat arm, comparator)

| Arm | Books NLL (vs C) | MMLU-Redux acc | Calibrated pairwise (vs C) | Letter mass | MMLU-fresh acc | Fresh pairwise | Binding cal. | GSM8K | GSM8K strict | Retrieval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp | 3.176 (-0.474 [-0.502, -0.446]) | 48.2 [46.9, 49.5] | 68.6 (+18.3 [+17.0, +19.6]) | 0.97 | 44.3 [42.6, 46.1] | 66.9 | 95.0 | 55.8 [53.1, 58.5] | 55.0 [52.3, 57.6] | 100.0% |
| qat_42 | 3.663 (+0.013 [-0.004, +0.030]) | 22.5 [21.4, 23.6] | 51.0 (+0.6 [-0.6, +1.9]) | 0.04 | 22.9 [21.4, 24.5] | 50.9 | 25.0 | 0.2 [0.1, 0.7] | 0.0 [0.0, 0.3] | 90.5% |
| C | 3.650 (+0.000 [+0.000, +0.000]) | 23.9 [22.7, 25.0] | 50.4 (+0.0 [+0.0, +0.0]) | 0.78 | 23.7 [22.2, 25.3] | 50.6 | 25.8 | 14.6 [12.8, 16.6] | 14.4 [12.6, 16.4] | 94.5% |
| M | 3.681 (+0.031 [+0.018, +0.044]) | 35.9 [34.6, 37.2] | 58.9 (+8.6 [+7.3, +9.8]) | 0.95 | 35.2 [33.5, 36.9] | 59.4 | 82.0 | 16.3 [14.4, 18.4] | 16.2 [14.3, 18.3] | 93.0% |
| U | 3.654 (+0.005 [-0.004, +0.014]) | 22.7 [21.6, 23.9] | 50.1 (-0.2 [-1.3, +0.8]) | 0.85 | 23.7 [22.2, 25.3] | 50.2 | 27.8 | 15.3 [13.5, 17.4] | 15.2 [13.4, 17.3] | 98.0% |
| B | 3.640 (-0.009 [-0.020, +0.001]) | 22.7 [21.6, 23.8] | 49.9 (-0.4 [-1.3, +0.4]) | 0.77 | 23.4 [21.9, 24.9] | 51.0 | 24.8 | 18.4 [16.4, 20.6] | 18.4 [16.4, 20.6] | 97.5% |

## 🟢 Validation curve (descriptive; final checkpoints fixed)

| Model | NLL |
|---|---:|
| fp | 3.2153 |
| qat_42 | 3.7094 |
| C | 3.7023 |
| M-init | 10.9690 |
| M-t8m | 4.2373 |
| M-t16m | 4.1407 |
| M-t33m | 3.9400 |
| M-final | 3.7251 |
| U-init | 10.9690 |
| U-t8m | 4.3841 |
| U-t16m | 4.1198 |
| U-t33m | 3.9799 |
| U-final | 3.7262 |
| B-t8m | 3.6864 |
| B-t16m | 3.7195 |
| B-t33m | 3.7070 |
| B-final | 3.6963 |

## 🟢 Data, contamination and compute

- MC corpus: 12599 kept items, 2,666,056 unique tokens, 1.54 passes; by source {'arc': 6649, 'incontext': 5950}; teacher letter accuracy {'arc_wrong_or_unparsed': 1607, 'incontext_correct': 4825, 'arc_correct': 5042, 'incontext_wrong_or_unparsed': 1125}.
- MC screen: {'exact_question': 2}; eval-cosine quantiles {'0.5': 0.08241865591978928, '0.9': 0.15827382385153305, '0.99': 0.25471370856720915, '1.0': 0.5005022998163452}.
- FP LR probe: {'selected_lr': 0.001, 'final_monitor_kl': {'0.0': 1.0085647329688072, '1e-05': 1.0290452614426613, '0.0001': 1.0278529785573483, '0.001': 1.0124183781445026}, 'baseline_frozen_kl': 1.0085647329688072, 'unfreeze_helps_in_probe': False, 'rule': 'lowest final monitor KL after 120 probe steps among 1e-5, 1e-4, 1e-3; frozen-FP baseline recorded'}.
- **13-token audit of every evaluation case against each arm's exact training stream.** **Zero cases have majority (> 50%) overlap in any set or stream**, which is the frozen gate. The counts below are cases with *any* shared 13-token run, which is not a violation:
  - C/U stream (`work/qat08_chat/data/train.u32`): MMLU-Redux 36/5,330 (max 0.146), MMLU-fresh 20/2,998 (max 0.068), GSM8K 4/1,319; books, binding and retrieval 0.
  - B stream: MMLU-Redux 36/5,330 (max 0.249), MMLU-fresh 18/2,998 (max 0.065), GSM8K 4/1,319; books, binding and retrieval 0.
  - M stream: every MMLU-Redux (max 0.367), MMLU-fresh (max 0.400) and binding (max 0.064) case shares some 13-gram. That is the shared instruction sentence and chat template: M's first paraphrase is word for word the evaluation instruction. It is not leaked content. GSM8K 4/1,319; books and retrieval 0.
- 🟢 **Compute:** the supervisor ledger records **20.709 GPU-h** of the 25.0 ceiling (the design projected 22.29). Breakdown:

  | Phase | GPU-h |
  |---|---:|
  | Prep | 1.478 |
  | Data (generation, packing, blocked v1 seal, v2 seal) | 0.393 |
  | Training: M 6.048, U 6.201, B 6.051 | 18.30 |
  | Exports and evaluation (GSM8K 0.281, held-out 0.150, curve 0.027) | 0.538 |

  Every arm stayed under its 6.5 h ceiling. The generated line "20.67" is the workers' own stage-time sum (20.674); the 0.035 h difference is process start-up and shutdown, which only the supervisor's clock counts. Paused wall time is not counted.

## Interpretation

- 🟡 **Multiple-choice-formatted teacher data restores part of the multiple-choice performance.**
  - Binding (answer stated in the prompt) rose from 25.8% (C) to 82.0%.
  - MMLU-Redux pairwise rose from 50.4 to 58.9, with a similar gain on MMLU-fresh.
  - Item-level disjointness is supported: M's items come from ARC train and FineWeb passages, and every evaluation case passed the 13-gram gate. General content learning is still possible.
  - M mixes closed-book ARC questions with in-context items, so this experiment cannot tell repaired read-out from new or relearned knowledge. The result is consistent with the diagnostics' `readout_broken` branch but does not prove that knowledge was merely blocked.
- 🟡 **The repair is partial.** Calibrated binding is 82% against FP's 95%; uncalibrated binding is 79.5% against FP's 99.75%. MMLU closes about half the gap to FP. The cause could be too little multiple-choice data (6.25% of the mix), lost knowledge, or both; this experiment cannot separate them. M's training prompts also used the evaluation's exact instruction wording, so other prompt formats are untested.
- 🟡 **Trained norms and gates (U) did not help.** This matches the pre-freeze probe, where freezing them was best. The extra trainable tensors gave no read-out or book benefit at this budget.
- 🟡 **Continued training (B) improves GSM8K, not multiple choice.** Its +3.9-point gain [+1.8, +5.9] is a real math gain and the only significant GSM8K change. It is not a joint recovery, because MMLU is unchanged. B's validation curve is flat (3.686 → 3.696 nat/token): 65.5M more tokens at a constant 1e-5 did not lower book loss. M's data combined with B's longer training is untested.
- 🟣 **Proposed for a 1.7B protocol** (needs its own decision and approval):
  - The frozen rule recommends renting with M's recipe.
  - Phase 1's 1.7B yardstick is a different model: Prism's Ternary-Bonsai-1.7B on the dense Qwen3-1.7B. It scored 74.7% on GSM8K in our Phase 1 harness (FP 82.0%). That is a size- and model-specific benchmark, not a target derived from this 0.8B hybrid result.
  - Our 0.8B students reach 16–18% against an FP of 55.8%, so reasoning is the larger remaining gap.
  - A 1.7B protocol should keep a multiple-choice slot, measure GSM8K, and treat the token budget and data mix as open variables.

## Limits

- One seed per arm, one 0.8B hybrid model and one recipe.
- M's multiple-choice data combines closed-book ARC and in-context items, so read-out repair and content learning are confounded.
- MMLU is scored by letter logits. Items, books and registries are treated as independent in the intervals.
- The teacher is imperfect: 75.8% correct on ARC and 81.1% on in-context items. The arms distill its distribution, errors included.
- Binding uses the same mean-log-prob calibration as the diagnostics. Uncalibrated scores are reported too, and they do not change H_M.
- The fresh set excludes two items that quote a passage present in the training data (see below); its 2,998 items are otherwise the frozen sample.
- Nothing here identifies PrismML's recipe.

## Incidents and provenance

- **Diagnostics amendment.** The frozen read-out decision stopped as `harness_invalid`, because the FP binding floor used calibrated accuracy (FP 94.0% calibrated, 99.25% uncalibrated).
  - With the user's approval, a disclosed post-results amendment ([decision_amendment.json](results/diag_readout/decision_amendment.json)) moved the same 95% floor to uncalibrated accuracy. That gave `readout_broken` → M, U, B.
  - The original decision is kept and hash-bound.
- **Reused books (before freeze).** The first book list reused 8 earlier titles. An independent audit caught it; the book check was rebuilt and the books replaced (`work/qat08_mcu/pre_freeze_incidents.json`).
- **Loader provenance fix.** The diagnostics loader now hashes the current plan, results and decision on every path. An independent parent review found the gap and retested the fix.
- **Self-test accounting.** Before launch, the GPU self-test became a prep stage so it counts toward the prep ceiling.
- **Blocked v1 seal → design v2.**
  - The v1 protocol seal was blocked by the 13-gram gate. MMLU-fresh `miscellaneous/7642/D` overlapped C's stream at 0.517 and M's at 0.569. Its sibling `miscellaneous/7932/C` quotes the same Zora Neale Hurston passage. The passage sits at C sequence 47,864, token 31, which is FineWeb sequence 35,898; the coordinates were corrected after the parent's check.
  - On the user's "Proceed with option 1", both items were removed (3,000 → 2,998, no replacement). The design was re-frozen as v2 (`8247ef1e…`); v1 is archived byte-for-byte in [versions/v1](results/qat08_mcu/versions/v1/MANIFEST.json).
  - Generation and packing were reused after an isolated CPU reproduction: packing was byte-identical. The multiple-choice prompts differed only in a screen annotation (`max_eval_cosine`, max |Δ| 0.0013). That was accepted by a separate validator record; the failed byte-identity check is kept.
  - An independent parent review ([REVIEW_MCU_V2_parent.md](reviews/REVIEW_MCU_V2_parent.md)) passed v2 before the reseal.
- **Pauses and resumes.** All were user-requested, used the controller's graceful pause, and resumed from `resume.pt` with no gaps or duplicate steps.
  - M was paused at step 1,551 and resumed 2026-10-05.
  - B was paused at step 1,036 and at step 1,719, and resumed 2026-10-05 and 2026-10-06.
- **Runtime stall.** On the 2026-10-06 resume, the executing agent's runtime did not start within 120 s. The parent thread stopped it and ran the same resume itself (`work/qat08_mcu/resume_2026-10-06_parent.json`). No frozen file or limit changed.
- **Approvals**, each in the user's exact words:
  - prep: "Proceed";
  - design v1 data phase: "Proceed";
  - v2 revision: "Proceed with option 1", given before the v2 hash existed;
  - training: "Proceed", after the seal.
- **Operational watchers** (`work/qat08_mcu/watch_run.sh`, `watch_milestones.py`) sit outside the frozen implementation.

## Verification

🟢 Checked on CPU after completion, independently by the executor and the parent thread:
- all 47 raw score outputs and all 6 GSM8K outputs match their recorded hashes;
- all 6 model files match;
- `freeze_qat08_mcu.py verify-design` passes;
- each arm's log has steps 1–2,000 consecutively;
- the frozen `decide_mcu` reproduces the stored decision exactly;
- M's MMLU-Redux accuracy (35.872%) and pairwise score (58.94) recompute from the raw rows.

The parent thread additionally re-parsed every GSM8K answer and recomputed the book intervals and retrieval scores: [REVIEW_MCU_RESULTS_parent.md](reviews/REVIEW_MCU_RESULTS_parent.md).

## Reproduction

From `analysis/bonsai2/replication`. Read-only verification of the finished run (CPU):

```bash
PY=../../../.venv/bin/python
CUDA_VISIBLE_DEVICES= $PY freeze_qat08_mcu.py verify-design
CUDA_VISIBLE_DEVICES= $PY diag_cases.py verify
CUDA_VISIBLE_DEVICES= $PY qat08_mcu_control.py status
CUDA_VISIBLE_DEVICES= $PY test_qat08_mcu.py
```

Historical entry points only: the commands below are not a complete, ordered bootstrap script. They already ran in this checkout and some write exclusive files. A new run requires an isolated checkout, the preparation and approval steps in [Plan B, Tasks 8–10](../../../docs/superpowers/plans/2026-10-03-qat08-mcu-experiment.md), and approval records bound to its own hashes. The v2 revision commands additionally require the archived v1 manifest and artifacts described above; they cannot bootstrap that archive. Do not re-run the revision commands against this completed run.

```bash
$PY diag_cases.py cases
$PY diag_cases.py freeze
bash run_diag_readout.sh                       # read-out diagnostics (systemd unit diag-readout)
$PY diag_amendment.py "Let's go with option 1 then"
$PY prepare_qat08_mcu.py cases
$PY prepare_qat08_mcu.py mc
$PY qat08_mcu_control.py start --phase prep    # self-test, pilot, FP-LR probes
$PY revise_qat08_mcu_v2.py retire              # design v2 revision (run once, after the blocked v1 seal)
$PY revise_qat08_mcu_v2.py cases
$PY revise_qat08_mcu_v2.py reuse-check
$PY revise_qat08_mcu_v2.py accept
$PY revise_qat08_mcu_v2.py record
$PY freeze_qat08_mcu.py design
$PY qat08_mcu_control.py start --phase data    # generation, packing, protocol seal
$PY qat08_mcu_control.py start --phase run     # training M/U/B, exports, evaluation, analysis, report
```

Controller operations, one command each: `$PY qat08_mcu_control.py pause`, `$PY qat08_mcu_control.py resume`, `$PY qat08_mcu_control.py status`.

Running the `report` stage again rewrites this file's generated parts; the hand-written sections would need restoring. Checkpoints, GGUFs and scores are in `work/qat08_mcu/` (git-ignored).

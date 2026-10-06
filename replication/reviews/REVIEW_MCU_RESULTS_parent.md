# Independent check of QAT08-MCU results — 2026-10-06

**Verdict: the measured scores and frozen decision reproduce.** The frozen recommendation is `rent_1p7b_with_recipe`, using arm M. This is a recommendation for a future experiment, not an instruction to rent or run anything.

The parent Codex agent checked the existing artifacts on CPU. It ran no training or model evaluation, changed no scientific inputs, and did not re-render or overwrite the results.

## Checks

- `freeze_qat08_mcu.py verify-design` passes. The design is `8247ef1ee2e87b1b0d55a857225587def0790f9a6f68f2e5e50b2fab295a44a9`; the protocol is `b99f4bddfd2177c1f52cb83a747eb1173ea030944a02b1e32d145f89b82a783e`.
- Every raw-score, final-model and GSM8K hash recorded in `results.json` matches its file. Score case IDs and order match the frozen case files; GSM8K IDs and gold answers match its frozen cases.
- Independent calculations from the raw arrays reproduce raw MMLU accuracy, calibrated pairwise scores, binding accuracy, per-book means, book confidence intervals, paired MMLU intervals and retrieval accuracy for all six models.
- Re-parsing every saved GSM8K response with the pinned parser reproduces its prediction and correctness. Ordinary and natural-EOS accuracy, Wilson intervals and paired GSM8K intervals reproduce.
- All three training logs contain exactly steps 1–2,000, with no gaps or duplicates; all three DONE files exist. The run state is complete, with no budget stop.
- Applying `decide_mcu` to the verified summary and the sealed rules reproduces the stored decision exactly. The supervisor ledger records 20.709459 GPU-hours, within the 25-hour ceiling.

## Findings to preserve in the report

| Measure | Chat comparator C | MC-data arm M |
|---|---:|---:|
| MMLU-Redux accuracy | 23.86% | 35.87% |
| Fresh MMLU accuracy | 23.72% | 35.16% |
| Calibrated pairwise MMLU | 50.36% | 58.94% |
| Binding, calibrated / raw | 25.75% / 24.75% | 82.00% / 79.50% |
| GSM8K | 14.56% | 16.30% |
| Held-out book NLL | 3.6496 | 3.6806 |

M passes the joint recovery, fresh confirmation and cost rules. Its paired MMLU gain is +8.57 points [7.30, 9.85]; its book cost is +0.03097 nat/token [0.01768, 0.04425], below the allowed upper bound of +0.10. Recovery means crossing the declared floors: it remains below FP's 48.20% MMLU and 55.80% GSM8K.

The stricter `H_M` hypothesis fails because binding is 82%, below its 90% floor. Its raw binding score also fails, so this failure cannot be attributed solely to the calibration artifact seen in diagnostics.

M's GSM8K improvement over C is +1.74 points [-0.47, 3.96], which does not establish an improvement. B's improvement is +3.87 [1.81, 5.92], but B remains at chance on MMLU and does not pass the joint recovery rule. U also fails MMLU recovery. These negative results concern the particular learning rates, data, initialization and budgets tested.

The M intervention combines closed-book ARC questions and synthetic questions whose answers appear in the prompt. It cannot isolate letter-format learning from knowledge learned from those teacher transcripts. One seed, correlated questions, the shared instruction wrapper and the omission of the combined MU arm limit generalization.

The report must distinguish any 13-gram overlap from majority-overlap gate violations, retain both dataset amendments and their provenance, and disclose the pauses and operational agent stall. No new GPU experiment or rental is authorized by this review.

## Final report review

**Pass.** The completed report includes the recipe, licenses, thresholds, partial-recovery limits, incidents and provenance. Its causal interpretation now keeps format learning and content learning separate as unresolved explanations. The reproduction section distinguishes read-only verification from historical entry points and the prerequisites for a new run. Binding comparisons use the same metric on both models. All local report links resolve, the plan/chat/design status pointers are updated, and the frozen implementation hashes remain unchanged.

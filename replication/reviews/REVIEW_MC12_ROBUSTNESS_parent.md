# Independent MC12 wording-check review — 2026-10-06

**PASS: M meets the frozen `wording_robust` criterion on these development cases.**

The parent checked the sealed protocol and every one of the 24 raw-score hashes, verified IDs and ordering against each frozen case file, recomputed all letter and binding metrics, and reproduced the complete saved verdict and table from raw arrays. No GPU work was run by the reviewer.

| Instruction | M minus C, calibrated pairwise points [95%] | M raw letter accuracy | M raw binding |
|---|---:|---:|---:|
| original | 7.56 [5.15, 9.97] | 36.74% | 80.25% |
| give_letter | 8.60 [6.27, 10.92] | 34.55% | 75.00% |
| pick_choice | 8.73 [6.36, 11.10] | 34.68% | 80.75% |
| output_single | 7.03 [4.60, 9.46] | 34.81% | 82.75% |

All four instructions pass FP's raw-binding and letter-pairwise sanity floors. Each novel wording has a positive paired lower bound and retains more than half the original gain. No variant was excluded. Original FP and C log-probabilities reproduce the earlier diagnostics exactly.

This supports generalization across these three instruction wordings: M's advantage does not require the exact instruction sentence used in training. It does not establish robustness to arbitrary formats, reorderings or free-form tasks. Binding remains 75–83%, below FP's 97.5–99.75% across these wordings; recovery is partial. These are development questions, not a new held-out capability verdict.

The result is descriptive and changes no mix, recipe, held-out set or decision threshold. The already-authorized MC12 phase can proceed under the existing protocol approval. A requested pause or a budget stop still takes precedence.

# Independent MC12 preflight review — 2026-10-06

**PASS: the frozen design implements the authorized wording check and one 12.5% multiple-choice arm.** This review releases that scoped local GPU work under the user's earlier “Proceed with the recommendation” (`phist_f8bxskdn7s`); it does not claim the user reviewed these hashes or authorize a rental, extra arms, budget changes or commits.

- Design SHA256: `336237b71523994cf58bbb5558629eb477910baa78033e00e94804bff17889b7`.
- Protocol SHA256: `de416a69e5ab549f64fe2cd22fc93b0bc398242703d68324d63ee33970761335`.
- Reviewer: parent Codex agent, independent of the Opus implementation author. Review and tests were on CPU; the only GPU action was an idle-status query.

## Independently checked

The parent read all seven new scripts and ran the full MC12 suite: **25/25 passed**. Tests cover the actual stream, screening, decision boundaries, frozen-file mutation rejection, namespace isolation, conservative budget accounting, process-group shutdown, durable publication and a toy crash/restart. The resumed toy reproduces the uninterrupted losses and final weights, with no duplicate steps.

Every one of the 64,000 stream rows was independently compared with its source: 40,000 ordered FineWeb rows, 16,000 unchanged C chat rows and 8,000 contiguous MC chunks. The MC corpus is reused unchanged for 3.0727 passes. The folded start and training block match M; the new explicit seed is recorded and historical seed parity is not claimed.

All 15 downloaded Gutenberg title headers match, and a separately captured historical inventory of 1,977 files, including MCU, finds no clashes. Fresh2 has 3,000 items and no overlapping IDs, questions, 13-word passages or option tuples with all 3,000 archived MCU-fresh items, including both Hurston exclusions. Reused evaluation files are byte-identical.

The parent independently reran all seven evaluation sets against the exact MC12, M and C streams. All **21 audit cells** agree with the frozen audit, including top-case fractions; none exceeds the majority-overlap gate. MC wrapper overlap is disclosed. Content-only screening checks both retained prompts and teacher answers.

The sealed manifest verifies: reviewed new-code hashes match exactly; old numeric helpers match their historical chat hashes; the 29 implementation entries, 53 inputs, 32 runtime libraries, comparator models and reused outputs are checked. All decision rules and budgets match the reviewed declarations. No GPU scoring occurred before the freeze.

## Findings resolved before freeze

Review fixes added teacher-response screening; actual evaluation-file and source provenance checks; historical diagnostic-score binding; concurrent-phase refusal; shared training/scoring pause signals; hard shutdown of descendants; durable outputs before completion markers; truncated-journal recovery; and rollback of crash logs to the saved optimizer step, including crashes before the first checkpoint. Thresholds were not relaxed. A TF-IDF test fixture below the intended cutoff was corrected rather than changing the cutoff.

The original prompt-only screen and corrected rebuild are archived and disclosed. Fresh2 membership and actual training inputs stayed identical.

## Non-blocking historical test limitation

The unchanged MCU `test_new_books_are_unused` scans the present tree. It now sees MCU book names in the later MC12 historical-inventory provenance, so it fails that timeless “unused” assertion. This is a later reference to prior work, not book reuse in MC12 or a changed MCU artifact. The MCU protocol, code and training data remain unchanged; the author reports the other MCU tests pass. The frozen historical test has not been edited.

## Limits and operation

One training run cannot measure seed-to-seed variability. Doubling MC exposure also reduces FineWeb exposure at equal total tokens, and the closed-book/in-context mixture cannot isolate format repair from content learning. Robustness is descriptive, uses only development cases, reports all variants and does not retune the training mix or held-out selection. Original-wording FP sanity failure stops the run.

The reviewed ceiling is 8.0 GPU-hours, split 0.75 robustness / 6.5 training / 0.75 evaluation, with shutdown grace inside those limits. Checkpoints retain optimizer and RNG state. A requested pause stays paused until the user resumes.

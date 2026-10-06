# Independent review of MCU preparation — 2026-10-04

Reviewer: parent Codex agent, independent of the Opus implementation author. Scope: the decision amendment, its loader, and the corrected book inventory. Read-only CPU checks; no GPU execution, approval, or changes to frozen artifacts.

**Decision: pass within this review's scope. The amendment calculation and corrected book selection pass; the loader finding below is fixed and independently retested. GPU preparation still needs the user's approval.**

## Measured checks

- 🟢 `diag_amendment.amend` applied to the real saved results and original decision exactly reproduces the saved amended branch, arm lists, stop flag and reasons. Only the FP binding sanity floor moves to raw accuracy; its 95% threshold and the other branch rules stay fixed.
- 🟢 The actual `plan.json` and `diag_amendment.py` match the hashes recorded in the amendment. The original stop decision remains separate, and the amendment explicitly records that it was chosen after seeing results.
- 🟢 All three amendment tests pass, including low raw FP accuracy and low FP letter signal stopping the rule.
- 🟢 Both book-inventory tests pass: historical filename/split/protocol forms are detected, the eight reused titles and reused reserve are rejected, and the new book choices have no clash in the broadened inventory.
- 🟡 Title matching scans arbitrary text, including tokenizer vocabulary, so it can reject unused books. This is conservative and affects availability rather than making a used book pass.

## Loader finding — resolved

🟢 The reviewed initial `qat08_mcu_control.diagnostics` compared the amendment's recorded plan hash with the original decision's recorded hash, but did not compare either with the current `plan.json`. Reproduction: copy the real four diagnostic files to a temporary directory, modify only the copied `plan.json`, and call `diagnostics` on that directory. It still returned `readout_broken`.

This is a loader defect, not evidence that the real frozen plan changed: its current hash passes. Before preparation, compare the original decision's hashes with the actual plan and results on both amended and unamended paths; retain the amendment-to-decision checks. Add mutation tests and rerun the relevant suites. Requested from the Opus author through bb.

🟢 **Resolution independently verified:** the updated loader hashes the current decision, results and plan. Both decision paths check the actual results and plan, and the amended path additionally checks the actual decision. I reran `test_diagnostics_rejects_mutated_provenance` and `test_diagnostics_real_files_and_plan_mutation` on CPU; both pass, including my original changed-plan reproduction. Both book-inventory tests also finished and passed. No unresolved finding remains within this limited review.

## Scientific limits retained

🟡 The amended branch is a disclosed post-results decision. A matching hash does not make it predeclared. The continued-training arm follows a weak checkpoint trajectory; it is not strong evidence that longer training alone fixes the failure. These checks do not review every training/export/runtime path in Plan B or authorize GPU work.

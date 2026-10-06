# QAT08-MCU v2 contamination amendment — parent review

**Decision: PASS for the scoped revision and protocol reseal. Training is not approved.**

Reviewed on CPU by the parent Codex agent, independently of the Opus author, on 2026-10-04. The user authorized “Proceed with option 1”: the two specified exclusions, preservation of v1, data reuse after checks, and protocol resealing. This instruction preceded the v2 hash; it is not represented as the user having reviewed that hash.

- v2 design SHA256: `8247ef1ee2e87b1b0d55a857225587def0790f9a6f68f2e5e50b2fab295a44a9`.
- Historical v1 design SHA256: `2963624c41e8b7fe09739f96c83b4eace922b5352b84b470f75fb1e5bd363a10`.
- All 30 archived files match the read-only v1 manifest. The current MMLU-fresh TSV is exactly the original lines minus `miscellaneous/7642/D` and `miscellaneous/7932/C`, with no replacements and order preserved: 3,000 → 2,998. Every other case file matches its original hash.
- Compared both designs: arms, training, decision rules, budgets, data recipe, splits, learning-rate probes, comparators, input hashes, environment, runtime hashes and measured prep time are identical. Changes are restricted to cases, amendment/provenance, implementation hashes, an added deviation and pre-freeze audits.
- All ten reused-file hashes and all three reuse-evidence records verify. Temporary packing reproduced the unique MC transcript tokens, kept-response record, M and B training streams, and data record byte for byte.
- Regenerated MC prompt files are **not** byte-identical: only `max_eval_cosine` annotations differ, by at most 0.0013268, because TF-IDF includes the two fewer evaluation texts. Actual teacher inputs, IDs/order and all other row fields match. Screening retains the same prompts, with no threshold crossing. Original prompt files, response files and timing records remain unchanged. The failed byte-identity check is retained alongside the explicit annotation-only acceptance record.
- Read the revised code and independently ran the three relevant tests: exact real-case exclusions, majority-overlap rejection, and reuse validation. All pass. The reuse test rejects changed prompt tokens, order, membership, non-annotation fields, threshold crossings, exclusions, packed digests and archive hashes.
- The new pre-freeze audit covers seven sets, totaling 10,477 cases, against each of four 65,536,000-token streams: the actual QAT08 training prefix, C/U, M and B. Every stream has zero cases over the frozen 50% gate. This does not claim zero shared text; M has intended wrapper overlap and its maximum remaining case overlap is 40%.
- `freeze_qat08_mcu.py verify-design` exits successfully. The protocol must still perform and seal its own complete per-stream audit.

The original contamination incident remains recorded, with an appended coordinate correction: mixed C sequence 47,864 corresponds to raw FineWeb sequence 35,898. The exclusion decision is unchanged.

Approval for training remains a separate, pending user decision after the sealed protocol is available. No training or GPU research was performed by this reviewer, and nothing was committed.

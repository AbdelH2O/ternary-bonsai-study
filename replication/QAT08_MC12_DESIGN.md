# QAT08-MC12 design: prompt robustness, then 12.5% multiple-choice data at M's recipe — 2026-10-06

**Status: frozen; robustness scoring running, then training. No rental is authorized by any outcome.**

- Frozen [design](results/qat08_mc12/design.json) `336237b71523994cf58bbb5558629eb477910baa78033e00e94804bff17889b7`, [protocol](results/qat08_mc12/protocol.json) `de416a69e5ab549f64fe2cd22fc93b0bc398242703d68324d63ee33970761335`.
- Approval: the user's "Proceed with the recommendation" authorized this scoped plan before the hashes existed; the user has not reviewed the hashes. The parent thread's independent preflight passed: [REVIEW_MC12_PREFLIGHT_parent.md](reviews/REVIEW_MC12_PREFLIGHT_parent.md).

Follows [QAT08_MCU.md](QAT08_MCU.md), where arm M (6.25% multiple-choice data) met the frozen recovery gate but only partly. Evidence tags: 🟢 measured; 🟡 inferred; 🟣 proposed.

## What is tested

1. **Prompt robustness of M** (descriptive; development items only). FP, C and M are scored on the diagnostics' 1,508 MMLU-validation items and 400 synthetic binding items, with the case and letter order fixed. The wordings are the original instruction plus three that appear in no multiple-choice training prompt:
   - "Answer with only the letter (A, B, C, or D) of the correct option." (original)
   - "Give the letter (A, B, C or D) corresponding to the right answer, and nothing else."
   - "Pick the correct choice. Your reply should be only its letter."
   - "Output only the single letter of the option that is correct."

   **FP sanity per wording:** binding raw (uncalibrated) ≥ 95% and letter pairwise ≥ 60.
   - If the original wording fails, the harness is invalid: stop and report an incident.

   **M's wording-robustness verdict:**
   - **wording-robust** needs all three:
     - the original-wording M−C gain has a paired lower bound > 0;
     - at least 2 of the 3 new wordings are FP-sane;
     - on every FP-sane new wording, M−C has a lower bound > 0 and a mean ≥ 50% of the original gain.
   - **not robust:** the first two conditions hold, but some sane new wording fails its gain test.
   - **inconclusive:** otherwise.

   Every wording is reported. The result never changes the MC12 mix, the evaluation wording or any held-out selection.
2. **MC12**: one new arm, identical to M except the share of multiple-choice data.

## 🟢 Recipe

| | M (control, existing) | **MC12 (new)** |
|---|---|---|
| Per 16 sequences | 11 FineWeb, 4 chat, 1 multiple-choice | **10 FineWeb, 4 chat, 2 multiple-choice** (slots 10 and 14) |
| Share | 68.75% / 25% / 6.25% | **62.5% / 25% / 12.5%** |
| FineWeb rows | sequences 0–43,999 | sequences 0–39,999 (the same prefix) |
| Chat rows | C's rows | C's rows, byte-identical |
| Multiple-choice tokens | 4.096M (1.54 passes) | **8.192M (3.07 passes)** over the same 2,666,056 unique tokens, as consecutive whole 1,024-token chunks |

The rest of the recipe is the same as M:
- Qwen3.5-0.8B with a folded H512 start;
- the PQ2_0 top-86 quantizer;
- FP tensors frozen, with the same BF16 load policy;
- AdamW with BF16 moments, betas (0.9, 0.95), clip 1.0;
- a cosine schedule, peak 1e-4, 50 warmup steps;
- 2,000 steps × 32,768 tokens = 65.536M tokens;
- the same teacher and all-position forward KL.

MC12's seed is pinned at 20261007. M's start-up random state was never recorded, so seed parity with M is not claimed.

The data is reused unchanged after hash checks against the sealed MCU protocol: the multiple-choice prompts, the teacher responses, and the kept and packed corpus. Nothing is regenerated.

Sources and licences:
- **Multiple-choice items:** ARC train (CC BY-SA 4.0), plus in-context items built on FineWeb-Edu passages (ODC-By 1.0).
- **Chat:** Daring-Anteater prompts (CC BY 4.0).

## 🟢 Evaluation

Fresh sets:
- **Books:** 15 new Gutenberg books (3 validation, 12 held out). They were checked against every earlier experiment, MCU included, and every title was verified.
  - Validation: The Warden, Agnes Grey, Jude the Obscure.
  - Held out: The Woodlanders, The Egoist, Felix Holt, Romola, Rob Roy, The Antiquary, Sybil, Coningsby, The Last Chronicle of Barset, Humphry Clinker, The Mysteries of Udolpho, Evelina.
- **MMLU-fresh2:** 3,000 never-scored MMLU test items (`47efa1a7…`), from 3,964 candidates left after every exclusion:
  - MCU's Redux, dev and content filters;
  - every archived v1 MCU-fresh item, the two Hurston items included, by index, question, 13-word span and option tuple;
  - content screening against every retained multiple-choice prompt and teacher response (exact question, 13-word span, TF-IDF ≥ 0.6);
  - majority 13-token overlap with the MC12, M and C streams.

Reused sets (labelled reused, not fresh): MMLU-Redux (5,330 items), GSM8K (1,319), retrieval (200 registries), binding (400). FP, C and M scores on these are copied from the sealed MCU run after hash checks.

**Pre-freeze 13-token audit**, on complete rendered cases: zero cases with majority overlap in all 7 sets and 3 streams.
- In the MC12 and M streams every MMLU and binding case shares some 13-gram, because M's instruction sentence and chat template are shared. The maxima are 0.367 (Redux) and 0.393 (fresh2).

## 🟢 Decision rules (frozen; MC12 compared with M)

- **material_mc_effect:** MMLU-Redux calibrated pairwise, MC12 − M, mean ≥ +2.0 points with paired 95% lower bound > 0.
- **fresh_corroborated:** MMLU-fresh2 pairwise, MC12 − M, paired lower bound > 0.
- **costs_ok:** all four:
  - book NLL vs M, upper bound ≤ +0.05;
  - book NLL vs C, upper bound ≤ +0.10;
  - GSM8K vs M, paired lower bound ≥ −3 points;
  - retrieval ≥ 90%.
- **recovery_floor** (inherited from MCU):
  - MMLU-Redux and MMLU-fresh2 Wilson lower bounds > 30;
  - GSM8K and strict GSM8K Wilson lower bounds > 10.
- **Binding:** MC12's raw accuracy and its paired difference from M are reported. The target of ≥ 90% is reported as met or unmet.
- **Recommendation:**
  - `increase_mc_to_12.5` if all four conditions above hold;
  - else `keep_m_6.25_mc12_worse` if the Redux MC12 − M upper bound is < 0;
  - else `keep_m_6.25_inconclusive`.

  No rental follows automatically.

The +2.0 threshold is about a quarter of M's +8.6-point gain over C, and above the ~1.3-point half-width of M−C's paired interval. It was set before any MC12 scoring.

## 🟢 Budget and operations

- **GPU-hour ceilings:** 8.0 in total, split as robustness ≤ 0.75, training ≤ 6.5 and evaluation ≤ 0.75.
- **Supervisor accounting** counts all wall time, including failed and paused sessions:
  - an unfinished stage is charged through its last heartbeat plus 10 minutes;
  - the 300-second pause grace is reserved inside each ceiling;
  - there is no raise; a breach stops the run and is reported.
- **Phases:** two systemd user units, `qat08-mc12-robust` then `qat08-mc12-run`. Training cannot start before robustness completes, and only one phase can be active at a time. Every launch, resume and stage re-verifies the protocol.
- **Durable outputs:**
  - checkpoints and logs are fsynced;
  - stage markers are written durably and only after their outputs;
  - a crash restart drops log records past the checkpoint, archiving the original first;
  - the budget hard-stop kills the whole worker process group.
- **Pause and resume:** `python mc12_control.py pause` finishes a step and saves; then `resume` and `status`. A paused, failed or budget-stopped phase is never restarted automatically.

## Reproduction

From `analysis/bonsai2/replication`, with `PY=../../../.venv/bin/python`:

```bash
CUDA_VISIBLE_DEVICES= $PY test_qat08_mc12.py            # 25 CPU tests
CUDA_VISIBLE_DEVICES= $PY freeze_qat08_mc12.py verify   # read-only check of every frozen hash
```

Record only (already run here; they write exclusive files):

```bash
$PY prepare_qat08_mc12.py stream
$PY prepare_qat08_mc12.py robust
$PY prepare_qat08_mc12.py cases
$PY freeze_qat08_mc12.py design
```

GPU phases:

```bash
$PY mc12_control.py start --phase robust
$PY mc12_control.py start --phase run
```

## Pre-freeze incidents

The fresh2 content screen first used only prompt contents for the exact-question and TF-IDF checks. It was fixed to use prompts and responses, found by the parent's review. The rebuild was identical: 0 dropped, the same 3,000 items. Both builds are recorded in `work/qat08_mc12/pre_freeze_incidents.json`, and the first is archived under `results/qat08_mc12/rejected/`.

The frozen MCU test `test_new_books_are_unused` now fails only because MC12's records store the historical book inventory, MCU titles included, as provenance. It is disclosed, not edited.

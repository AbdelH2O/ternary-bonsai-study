# Read-out diagnostics for the QAT08 chat student — results

**Decision (predeclared rule): `harness_invalid`; next experiment arms [].**

- FP failed the harness sanity floor; investigate before any experiment

**Amendment (2026-10-04, written after the results were seen; [decision_amendment.json](results/diag_readout/decision_amendment.json)): `readout_broken`; next experiment arms M, U, B (optional MU).**

- The frozen FP binding floor used calibrated accuracy. FP scored 94.0% calibrated but 99.25% uncalibrated, with pairwise 98.0 [97.2, 98.8] and balanced predictions (99/101/101/99).
- Mean-log-prob calibration subtracts per-letter offsets (A −3.50 to D −4.70 nats). These measure how strongly a confident model rejects wrong letters, not a letter prior. They flip 22 gold-A items whose margin is under 1.2 nats.
- The amendment applies the same 95% floor to uncalibrated accuracy and changes nothing else. The original `decision.json` is kept and hash-bound. The branch is the same under any FP floor FP passes, because the chat student's 27.2% is far below the 70% cut.

Spec: [review](reviews/REVIEW_QAT08_CHAT_opus55.md) sections 4-6; frozen [plan](results/diag_readout/plan.json); [results](results/diag_readout/results.json); [decision](results/diag_readout/decision.json). Development data only (MMLU validation, synthetic registries). Evidence tags: 🟢 measured; 🟡 inferred; 🟣 proposed.

## 🟢 Per-arm signal (pairwise gold-vs-wrong %, 50 = no information)

| Arm | Letter raw acc | Letter calibrated acc | Letter pairwise [95%] | Letter mass | Cloze chat pairwise | Cloze raw pairwise | Binding calibrated acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp | 43.5 | 44.8 | 65.7 [63.8, 67.6] | 0.968 | 56.9 [55.0, 58.8] | 57.6 [55.7, 59.5] | 94.0 |
| gptq | 25.1 | 23.7 | 49.5 [47.6, 51.4] | 0.247 | 51.0 [49.1, 52.9] | 50.6 [48.7, 52.5] | 29.2 |
| qat_42-t8m | 25.8 | 24.3 | 49.4 [47.6, 51.3] | 0.049 | 50.8 [48.9, 52.7] | 52.1 [50.2, 53.9] | 24.5 |
| qat_42-t16m | 25.2 | 26.5 | 49.3 [47.4, 51.2] | 0.017 | 50.4 [48.5, 52.2] | 52.3 [50.5, 54.1] | 21.5 |
| qat_42-t33m | 25.0 | 23.9 | 50.2 [48.3, 52.1] | 0.036 | 52.3 [50.4, 54.2] | 52.7 [50.8, 54.5] | 26.8 |
| qat_42-final | 25.1 | 25.3 | 50.1 [48.3, 52.0] | 0.044 | 52.3 [50.4, 54.2] | 53.2 [51.3, 55.0] | 24.8 |
| qat_chat_42-init | 25.0 | 26.2 | 51.2 [49.3, 53.1] | 0.007 | 50.9 [49.1, 52.7] | 51.9 [50.1, 53.8] | 23.5 |
| qat_chat_42-t8m | 24.5 | 23.9 | 48.8 [46.9, 50.6] | 0.324 | 51.8 [49.9, 53.6] | 52.5 [50.6, 54.4] | 29.2 |
| qat_chat_42-t16m | 25.3 | 23.2 | 48.1 [46.2, 50.0] | 0.468 | 51.1 [49.3, 53.0] | 52.9 [51.0, 54.7] | 24.2 |
| qat_chat_42-t33m | 23.9 | 24.3 | 50.3 [48.5, 52.2] | 0.627 | 51.7 [49.9, 53.5] | 52.5 [50.6, 54.3] | 23.5 |
| qat_chat_42-final | 24.9 | 26.7 | 51.3 [49.4, 53.2] | 0.788 | 52.1 [50.3, 54.0] | 53.1 [51.3, 55.0] | 27.2 |

## 🟢 Trajectory (final minus t8m, pairwise points, paired item interval)

| Run | Letter | Cloze chat | Cloze raw |
|---|---:|---:|---:|
| qat_42 | +0.71 [-1.45, +2.87] | +1.53 [+0.04, +3.01] | +1.11 [-0.29, +2.50] |
| qat_chat_42 | +2.50 [+0.25, +4.75] | +0.35 [-0.97, +1.68] | +0.64 [-0.66, +1.94] |

## 🟢 Generative multiple choice and positional KL

| Arm | Generative acc [95%] | Parsed |
|---|---:|---:|
| fp | 46.7 [41.1, 52.3] | 245/300 |
| qat_42-final | 0.0 [0.0, 1.3] | 6/300 |
| qat_chat_42-final | 16.3 [12.6, 20.9] | 207/300 |

| Arm | KL question | KL template | KL answer position | 4-letter KL |
|---|---:|---:|---:|---:|
| qat_42-final | 1.196 | 4.691 | 3.910 | 0.928 |
| qat_chat_42-final | 0.500 | 0.116 | 0.794 | 0.566 |

## Interpretation

🟡 (Written by the executing agent from the tables above; inferred, not measured.)

- 🟡 **The read-out is broken.** The chat student cannot copy an answer stated in the prompt: binding calibrated accuracy 27.2% (pairwise 51.9 [48.3, 55.6]), with "A" predicted for 288 of 400 items, against FP's 99.25%. The plain-text student is also at chance (24.8%) despite 90.5% on raw-text retrieval in QAT08, so what fails is answering in the chat-and-letter format, not in-context copying as such.
- 🟡 **No knowledge signal survives without letters.** The best student option-likelihood lower bound is 51.3 (chat-final, raw format), below the 52 needed. FP's own signal there is modest (57.6 [55.7, 59.5]), so this test can only detect a large share of FP's knowledge; "absent" means "below about half of FP's signal", not zero.
- 🟡 **The trajectory is weak.** The chat student's letter pairwise rises +2.50 [+0.25, +4.75] from 8M tokens to final, which meets the rule's B trigger. But the 8M and 16M checkpoints sit below 50 (48.8, 48.1) and the final is 51.3, so most of the "rise" is recovery from a dip. Neither option-likelihood format rises.
- 🟡 **Chat training taught the format, not the answer.** Letter mass rose from 0.007 (init) to 0.788 (final) while pairwise stayed near 50. Against the plain student, KL to the teacher fell on the template (4.69 → 0.12) and at the answer position (3.91 → 0.79), but the student still doesn't put the probability on the right letter (4-letter KL 0.57).
- 🟡 **Reasoning first does not help.** The chat student reaches 16.3% (49 of 300), which is 23.7% of its 207 parsed answers, about chance; FP reaches 46.7%. The plain student almost never produces a parsable answer (6 of 300).
- 🟡 **Selected branch (amended): `readout_broken` → M, U, B; optional MU.** The rule: "b < 70 -> readout_broken (M, U; optional MU); … add B if letter or raw-cloze final minus t8m has mean >= 2 points and lower bound > 0". B enters on the weak letter trajectory above.


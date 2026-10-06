# Review of the chat-aware 0.8B amendment (QAT08_CHAT) — reviewer: Opus 5.5, 2026-10-03

**Scope.** This is an independent review of [QAT08_CHAT.md](../QAT08_CHAT.md), its frozen [design](../results/qat08_chat/design.json), [protocol](../results/qat08_chat/protocol.json), [execution amendment](../results/qat08_chat/resume_amendment_v1.json) and [results](../results/qat08_chat/results.json). I checked them against the raw per-item scores in `work/qat08_chat/scores/`, the raw GSM8K generations in `results/qat08_chat/gsm8k/`, the teacher data in `work/qat08_chat/data/`, the training log and the current scripts. Everything I did was read-only and CPU-only: no training, no model evaluation, and no edits to frozen files or existing reports. Context: [QAT08.md](../QAT08.md), [QAT08_CHAT_DESIGN.md](../QAT08_CHAT_DESIGN.md), [REPLICATION_PLAN.md](../REPLICATION_PLAN.md) and [PHASE1_17B.md](../PHASE1_17B.md). The handoff file `/tmp/claude-1000/bonsai-replication-handoff.md` does not exist on this machine, so I worked from the reports directly.

Evidence tags: 🟢 measured (I recomputed it or read it from an artifact); 🟡 inferred; 🟣 proposed. Nothing proposed here has been run or authorized.

---

## 1. Bottom line

**The result can be trusted as a measurement, and the frozen verdict is correct: chat recovery fails on MMLU.**

- The run is complete and the pipeline was not altered.
- All 40-plus frozen code, input, data and runtime hashes still match.
- The training log is continuous across the one pause.
- Every headline number reproduces from the raw files.
- No threshold, checkpoint or mixture was changed after the scores were seen.

The "pending" wording the user saw is the stale header of the pre-run design document (`QAT08_CHAT_DESIGN.md:3`, "Decision pending … await the user's go-ahead"). It is not the state of the run: `work/qat08_chat/run_state.json` is `complete`, and every stage has a `stage_finished … complete` event in `execution.jsonl`.

**What the run establishes** (🟢 unless tagged otherwise):

1. **Adding 25% teacher chat transcripts at a fixed 65.5M-token budget costs nothing on prose.**
   - Books: +0.005 nat/token [−0.012, +0.021] against the plain-text 42-zero arm.
   - Retrieval margin improves: +0.97 nat [+0.66, +1.27].
2. **It restores chat-mode generation.**
   - 95% of GSM8K outputs now stop naturally, against 0% for every earlier quantized arm.
   - The outputs are coherent step-by-step solutions.
   - GSM8K accuracy is 14.6% (192/1,319). If outputs were randomly reassigned to other items' gold answers, about 11.5 would be correct, so the gain is real. That is about 26% of FP's 55.8%.
3. **It does not restore MMLU in any form.** This is the key new finding, and the report misses it.
   - The chat student's letter ranking has no information about the answer. Gold beats a wrong option in 50.3% of pairwise comparisons [49.6, 51.0], against 68.6% for FP.
   - Removing each arm's letter bias moves accuracy only to 25.9%, essentially the same as the plain-text arm (25.8%).
   - On the 799 items where FP is more than 80% confident and right, the student gets 27.7%.
   - So the A-bias is a symptom, not the cause. The reported +1.39-point MMLU gain over the 42-zero arm comes entirely from shifting answers away from "A" toward the more common gold letters.

**What it does not establish:**
- The GSM8K gain is domain-matched (see V2). It is evidence of recovering *GSM8K-style math chat*, not general chat capability.
- Nothing here separates the two leading explanations for MMLU: lost parametric knowledge versus a missing multiple-choice answering skill (§4). Cheap diagnostics can (§5).

**Recommendation in one line:** run about 1–1.5 GPU-hours of read-out diagnostics on the existing exports first. Then run one local 0.8B factorial: multiple-choice read-out data × unfrozen non-projection tensors, with fresh held-out data. Do **not** rent 1.7B hardware yet (§7).

---

## 2. Completeness and integrity checks

| Check | Result | Evidence |
|---|---|---|
| Pipeline state | 🟢 complete; all 17 stages finished | `work/qat08_chat/run_state.json`; `execution.jsonl` |
| Frozen hashes | 🟢 0 mismatches across implementation, input and data hashes in `protocol.json`; design `185631c7…` and protocol `0a610877…` reproduce; the 4 amendment files match | recomputed SHA-256 |
| Training integrity | 🟢 2,001 log records for steps 0–2,000, no duplicate or missing steps; LR schedule identical to QAT08 at every step; KL continuous across the step-315 pause (0.743 → 0.812 → 0.769, within normal step noise) | `qat_chat_42/train.jsonl` vs `work/qat08/qat_42/train.jsonl` |
| Resume | 🟢 one graceful pause at step 315 with a fresh checkpoint; one resume; `DONE` written at step 2,000; 6.08 h of training against the 6.5 h ceiling | `execution.jsonl`, `DONE` |
| Exports | 🟢 all 5 GGUFs match their export records; one exporter hash; same byte size; zero fraction 0.328 (42/128) | `export_qat_chat_42-*.json` |
| Comparator reuse | 🟢 the 12 reused MMLU, retrieval and GSM8K outputs are byte-identical to the QAT08 originals and to the protocol hashes | `comparator_output_sha256` |
| Case counts | 🟢 MMLU 5,330 (gold A 1,200 / B 1,296 / C 1,365 / D 1,469); GSM8K 1,319 unique IDs per arm; 24 held-out and 6 validation book slices; 200 registries | `cases/`, score files |
| Headline statistics | 🟢 reproduced: MMLU 23.86% [22.74, 25.03]; GSM8K 14.56% [12.76, 16.56]; strict 14.40% [12.61, 16.40]; books chat − 42-zero +0.0047 [−0.0116, +0.0210] (paired t, 11 df; chat better on 4 of 12 books) | my recomputation |
| GSM8K parsing | 🟢 re-parsing with the frozen `gsm8k_17b.parse` gives 0 mismatches | `gsm8k_17b.py:95-97` |
| Data sizes | 🟢 `train.u32` = 65,536,000 tokens; `chat_unique.u32` = 2,098,230 tokens; 4,722 chats; 7.81 passes | file sizes, `data_record.json` |
| Teacher, template, thinking | 🟢 F32 FP GGUF (sha `604cb165…`), greedy, thinking off. Every kept chat was asserted equal to `apply_chat_template(..., enable_thinking=False)`, so training transcripts, MMLU prompts and GSM8K prompts all use the identical `<|im_start|>assistant\n<think>\n\n</think>\n\n` prefix | `prepare_qat08_chat.py:338-342`; decoded case files |
| Mixture layout | 🟢 every 4-sequence microbatch is [FineWeb, FineWeb, FineWeb, chat]. The trainer reads `step*32 + k*4`, so the 25% share holds in every microbatch, not just on average | `prepare_qat08_chat.py:350-358`; `qat_08b.py:212-216, 333-335` |
| Contamination | 🟢 rolling 13-token audit over the exact stream: no case with majority overlap; 36 MMLU and 4 GSM8K cases have a small overlap (maximum 14.6%) | `protocol.json` |

**Governance note (🟢, low scientific impact).** The design says training needs a user approval bound to *both* the design and protocol hashes. In practice the controller builds `training_approval.json` itself: it copies the design approval (recorded 22:20 UTC, before the protocol existed) and adds the protocol hash once the automated seal has run (`qat08_chat_control.py:138-141`). The user approved the design, and the protocol only adds data hashes and an automated contamination gate, so no scientific choice escaped review. Still, "explicit approval of the protocol" did not happen as a separate act. Future amendments should either ask the user explicitly or say plainly that the protocol seal is automatic.

---

## 3. Validity findings, ranked

### V1 — MMLU is a real failure to tell answers apart, not a letter-bias or harness artifact (🟢, high importance)

`work/qat08_chat/scores/*-mmlu.jsonl` stores the log-probabilities of A, B, C and D at the first assistant position. I renormalized over the four letters, then removed each arm's average letter preference ("contextual calibration"; also done leave-one-subject-out, with the same result).

| Arm | Raw acc. | Calibrated acc. | Gold-vs-wrong pairwise | Letter mass |
|---|---:|---:|---:|---:|
| FP | 48.20 | 48.31 | **68.6%** | 0.966 |
| GPTQ | 22.51 | 24.43 | 50.0% | 0.247 |
| QAT08 free | 22.46 | 25.48 | 50.7% | 0.027 |
| QAT08 42-zero | 22.48 | 25.80 | 51.0% | 0.043 |
| **Chat 42-zero** | 23.86 | 25.97 | **50.3% [49.6, 51.0]** | 0.779 |

- No subject reaches 60% pairwise for the chat arm. On marketing, sociology and psychology, where FP is above 84%, the chat arm stays between 49% and 55%.
- 🟢 The harness and position are right. The letter is read straight after `</think>\n\n`, which is exactly where every training answer begins. FP scores 48% here. In [PHASE1_17B.md](../PHASE1_17B.md), Prism's ternary 1.7B was within −1.4 points [−2.9, +0.2] of FP under this same letter harness, so the harness can register a ternary model's recovery.
- 🟢 The frozen ternary embedding does not merge the letters. I recomputed the top-86 and absmean projections of the folded FP embedding on CPU:
  - the letter rows keep cosine 0.88–0.90 to FP, the same as typical tokens (0.89);
  - the A/B/C/D difference directions keep cosine 0.79–0.81;
  - after ternarization the letters are slightly *more* distinct from one another (pairwise cosine 0.42–0.50 against FP's 0.51–0.60).
- 🟢 The chat data did teach the *format*: letter mass rose from 4% to 78%. It taught no MMLU *discrimination*.

### V2 — The GSM8K pass is real but domain-matched, and partly copies the training format (🟢, high importance for interpretation)

- **Kept-chat composition** (`kept_chats.json`; not in the report). Math is 1,388 chats and **28.7% of chat tokens**; general conversation 54.9%; precise instruction-following 13.8%; JSON 2.4%. Complex instructions are 0.2%: 5 of 1,000 survived, the other 995 hit the 768-token limit.
- **The math prompts are GSM8K-style word problems.** All 1,388 end with "Please show the calculation steps and lastly the final answer in format {{answer number}}". Some are templated near-twins of GSM8K problems. The closest pair (TF-IDF cosine 0.72) is the GSM8K test item "Kimberly bought 8 packages of cat food…" and the training prompt "Chad bought 6 packages of cat food…".
- **Similarity to GSM8K test items.** The median nearest-neighbour cosine is only 0.13, and only 26 test items reach 0.3 or more. The two most similar items were answered *wrong*, so there is no sign of memorized test answers. There is a modest near-domain gradient: the chat arm gets 21.2% in the most similar quartile against 11.5% in the least similar (×1.84), while FP goes from 63% to 50% (×1.26).
- **The student follows the training convention instead of the instruction.**
  - The GSM8K prompt asks for `\boxed{}`. FP complies in 1,173 of 1,319 outputs.
  - The chat student writes the training corpus's `{{N}}` in **760** outputs and `\boxed` in only 33.
  - The answer was therefore scored by the frozen last-number fallback in 1,286 of 1,319 outputs. This is permitted and harmless for the score: random-gold expectation is 0.87%.
  - It is direct evidence that 7.8 passes over a small corpus taught its surface conventions more strongly than instruction following.
- 🟢 **Report mislabel.** "0 parse fallbacks" (`QAT08_CHAT.md`, behaviour diagnostics) counts *server streaming* fallbacks (`gsm8k_17b.py:81`; `score_qat08_chat.py:189`), not answer-parser fallbacks. The sentence implies boxed compliance that did not happen.
- 🟡 **The errors are reasoning errors, not degeneration.** In item 0 the student adds eaten eggs to eggs laid. The chat arm solves 163 of FP's 736 correct items, plus 29 that FP misses.

### V3 — The 768-token generation limit strongly reshaped the chat corpus (🟢 measured, 🟡 effect)

52.8% of teacher responses (5,278 of 10,000) hit the 768-token limit and were dropped:
- conversation: 61% dropped;
- complex instruction: 99.5% dropped;
- math: 7% dropped.

The filter was predeclared and disclosed as a count. Its effect on content was not: the corpus became short-answer and math-heavy. 🟡 This probably helped GSM8K and gave nothing to knowledge-style multiple choice. Only **1** of the 4,722 kept prompts contains "multiple choice".

### V4 — The book comparison is equal in total tokens, not in prose tokens (🟢 fact, 🟡 meaning)

The chat arm's FineWeb is the **first 49.152M tokens** of QAT08's 65.536M stream (`prepare_qat08_chat.py:349-358`). Two consequences:
- "No book cost" means: at a fixed total of 65.5M tokens, replacing the last 25% of prose with chat costs nothing.
- 🟡 With 25% fewer prose tokens, the chat arm still matches the plain-text arm on books. The QAT08 curve was still falling by 0.18 nat between 33M and 65.5M tokens. Books therefore seem to depend more on total optimizer progress than on prose specifically. That supports cheap mixture changes, and it argues against "just more prose".

The FineWeb monitor KL moved accordingly: 0.388 for chat against 0.365 for QAT08 42-zero.

### V5 — The repeated chat data probably memorized its conventions (🟡)

The final training KL is lower for the chat arm (0.348 against 0.380), even though its FineWeb monitor KL is higher. From those two numbers, I estimate KL on chat sequences of roughly 0.2–0.25 nat/token, about 40% below FineWeb. This is the expected pattern for repeated, teacher-greedy, low-entropy text. Together with the `{{N}}` habit, it means more chat share or more passes over this corpus would mostly add memorization.

### V6 — Smaller points (🟢)

- Intervals treat items as independent, and there is one seed. For MMLU this does not matter (the failure is at chance). For GSM8K the lower bound is 2.8 points above the gate.
- The paired MMLU difference against the 42-zero arm, +1.39 [+0.73, +2.04], is "significant" but is a letter-bias shift (V1). It should not be presented as partial recovery.
- `QAT08_CHAT_DESIGN.md` still says "Decision pending". A one-line status pointer there would prevent the confusion the user hit. I did not edit it.

---

## 4. Ranked explanations for the residual gap

The gaps are: MMLU at chance (FP 48%); GSM8K at 26% of FP; books +0.49 nat/token above FP.

| Rank | Explanation | Status | For | Against / unknown |
|---|---|---|---|---|
| 1 | **Lost parametric knowledge at this budget.** The ternary student recovered fluency and in-context abilities (retrieval, step-by-step arithmetic on numbers in the prompt) but not the stored facts MMLU needs | 🟡 leading | Zero signal even on FP-confident items; book gap still +0.49; in-context tasks recover while closed-book ones do not | No direct knowledge probe yet (D1, D4) |
| 1 (tie) | **Missing multiple-choice read-out skill.** Mapping option content to a letter label was never trained: 1 multiple-choice prompt in 4,722 | 🟡 leading | GSM8K recovered exactly where the data matched; the format (letter mass) transferred while the skill may not | FP does this without multiple-choice training data. Untested: an in-context multiple-choice test (D2) separates this from rank 1 |
| 3 | **Training budget** (overlaps with 1) | 🟡 | Curve not flat; Prism's budget is probably orders of magnitude larger | A pure budget shortfall would usually show a weak signal, not none. Checkpoint scoring (D3) can measure the trend at no training cost |
| 4 | **Frozen non-projection tensors** (norms, GDN gates, A_log, dt_bias, conv) | 🟡 plausible | Prism changed every sampled tensor class at 27B (Phase 0 T1); without them the student cannot re-tune gains to ternary statistics | Prism 1.7B (Qwen3, not hybrid) keeps MMLU, but we do not know whether its norms were trained |
| 5 | **0.8B ternary capacity** | 🟡 unknown | FP 0.8B has a thin margin (68.6% pairwise), so modest damage reaches chance | Cannot be tested at 0.8B alone; this decides whether 0.8B is the right testbed (§7) |
| 6 | **Frozen ternary embedding** | 🟢 unlikely for letters | Row fidelity 0.89 for every token adds noise everywhere | Letters stay distinct (V1); Prism 1.7B keeps MMLU with a data-free absmean embedding |
| 7 | **Harness, template or position artifact** | 🟢 ruled out | — | Calibration does not help; the position matches the training format; FP and Prism are scored normally by the same harness |

The decisive open question is the **rank 1 tie**. If the student cannot answer multiple choice *even when the answer is in the prompt*, adding read-out data is the fix. If it can, the problem is knowledge or capacity, and read-out data would only teach a format.

---

## 5. Cheap diagnostics first (🟣; existing exports; no training)

All of these use the GGUFs already exported for `qat08_chat` (init, t8m, t16m, t33m, final) and for both QAT08 arms. None should use the MMLU-Redux or GSM8K held-out items for anything that feeds a decision. Use development data: the MMLU auxiliary/dev questions not in Redux, OpenBookQA or ARC *validation*, and fresh registries.

| # | Diagnostic | Cost | What each outcome means |
|---|---|---|---|
| D0 | ✅ Done here (CPU): letter calibration and pairwise signal; letter-embedding geometry; GSM8K format and near-domain analysis; kept-chat composition | 0 | See V1–V3 |
| D1 | **Cloze scoring of options.** Same prompts without letter labels; score each option's mean token log-probability given the question, in raw text and in chat format, on about 1,000 dev items; FP, 42-zero, chat | ~10 min GPU | Signal in cloze but not with letters → read-out problem (favours arm M). None in either → knowledge loss (favours U or budget) |
| D2 | **In-context multiple-choice binding.** The answer is stated in a short passage (for example a fresh registry: "What is Ann Lee's code? A. 4821 B. …"), with letter positions balanced; 400 items | ~5 min GPU | FP should be about 100%. Student under 70% → the read-out skill is broken; student at 90% or above → read-out intact, so the failure is knowledge |
| D3 | **Checkpoint trajectory.** D1/D2 plus letter-pairwise on dev multiple choice for t8m→final in both the plain and chat runs (10 exports) | ~20 min GPU | A rising pairwise signal (for example 50% → 54%+) means budget-limited, and longer training becomes a defensible arm. Flat at 50% means budget alone will not fix it |
| D4 | **Closed-book fact recall.** About 1,000 short raw-text factual prompts, scored by gold next-token rank (LAMA-style, from a licensed source, audited for overlap) | ~10 min GPU | Separates "facts lost" from "multiple-choice format" without any letter mechanics |
| D5 | **Teacher–student KL on multiple-choice prompts, split by position.** Question tokens versus the letter position, using the PyTorch training path on about 200 dev items | ~15 min GPU | Low KL on the question with high KL at the letter → read-out. High KL throughout → general representation damage |
| D6 | **Generative multiple choice with reasoning.** Let the chat student reason step by step on 300 dev items, then parse the letter | ~30 min GPU | If reasoning lifts it well above chance, the knowledge is partly present but the one-token answer cannot reach it |

Total is about 1–1.5 GPU-hours on the local RTX 5070. D1, D2 and D3 matter most; they determine the arms in §6.

---

## 6. Recommended next frozen experiment (🟣)

### QAT08-MCU: multiple-choice read-out data × unfrozen FP tensors, 0.8B, local

**Hypotheses.**
- **H_M:** MMLU stays at chance because the student never practised the multiple-choice read-out. Teacher-distilled multiple-choice transcripts restore a measurable letter signal at no prose cost.
- **H_U:** the frozen FP non-projection tensors limit how far ternary projections can recover. Training them, a tiny fraction of the parameters, improves books and capability.

**Arms.** All arms use: 42-zero (top-86) rule, H512, the same folded initialization, 65.536M tokens and 2,000 steps, the same LR, schedule, optimizer and all-position forward KL, and the same chat corpus slot.

| Arm | Stream (per 16 sequences) | Trainable | Notes |
|---|---|---|---|
| C | 12 FineWeb + 4 chat | projections | **Reuse the existing `qat_chat_42`.** Identical recipe; single seed noted as a limit |
| M | 11 FineWeb + 4 chat + **1 multiple-choice** (6.25% ≈ 4.1M tokens) | projections | Multiple-choice tokens come out of FineWeb, so chat exposure equals C |
| U | 12 FineWeb + 4 chat | projections + **non-projection FP tensors** (norms, GDN norm, A_log, dt_bias, conv1d, FP gates) | Their LR comes from a pre-freeze 120-step probe on the training-domain monitor only, as QAT08 did for the projections. Export keeps them F32; relax the byte-identity assertion only for those tensors and assert shape and dtype |
| M+U | as M | as U | Interaction arm. Drop it if the budget is tight; M and U alone still answer the main question |

**Multiple-choice data (arm M).**
1. **In-context binding items**, about 60%. FineWeb-Edu passages from *outside* the 65.5M training offsets (the pinned shard has 400M tokens). Each passage gets a question answerable from the passage, 4 options drawn from the passage or a neighbouring passage, and **letter position uniformly balanced**. Each item appears in 2 option orders, to teach position invariance.
2. **Closed-book items**, about 40%, from licensed training splits with no MMLU or ARC lineage. OpenBookQA train (Apache-2.0) is the obvious candidate; check licence and lineage before freezing. Do *not* use ARC, because ARC-Challenge is proposed below as a fresh evaluation set.
3. **Prompts.** 5 or more paraphrases of "answer with only the letter", some with a short explanation allowed. FP F32 teacher, thinking off, greedy, the same template-identity assertion as now. **What matters is the teacher's full soft distribution at the letter position** under all-position KL. This is not response-only training and needs no hard labels.
4. **Repetition cap:** at most 4 passes over the multiple-choice corpus, so it is not repeated as heavily as the chat corpus (7.8 passes).

**Fresh held-out data and contamination requirements.**
- 12 new Gutenberg books, title-checked and absent from every earlier freeze inventory, plus 3 validation books.
- MMLU-Redux (continuity; same 5,330 items) **plus a fresh knowledge set never used in this project**, such as ARC-Challenge test (1,172) and ARC-Easy test (2,376), scored with the same letter harness. Record the FP reference and the gold-letter distribution.
- The D2 in-context test, rebuilt from a new seed as a held-out read-out check.
- GSM8K unchanged (continuity), with its parser and termination rule.
- Audit **all** new training text (multiple-choice and FineWeb) against every evaluation set with the rolling 13-token check, keeping the existing majority-overlap block. Add a near-duplicate screen: reject any training multiple-choice item with TF-IDF cosine ≥ 0.6 to any evaluation question. Report the similarity distribution as was done for GSM8K in V2.
- Report kept-data composition by category and the share of generations dropped at the token limit (V3). Raise the generation limit to 1,536 for the multiple-choice explanations, or predeclare letter-only answers.

**Primary metrics.**
- MMLU-Redux and ARC accuracy, plus **calibrated gold-vs-wrong pairwise rate**. This is the bias-proof signal V1 showed is needed; raw accuracy alone can move with letter bias.
- D2 accuracy, book NLL, GSM8K both ways, and retrieval.

**Decision thresholds** (predeclared; paired against C; 95% intervals as now):

| Claim | Requirement |
|---|---|
| H_M supported | M − C MMLU-Redux pairwise ≥ +4 points with lower bound > 0, **and** ARC-Challenge pairwise lower bound > 50%, **and** D2 ≥ 90% |
| H_U supported | U − C on books ≤ −0.05 nat/token with upper bound < 0, **or** U − C MMLU pairwise ≥ +4 points with lower bound > 0 |
| Recovery (unchanged gate) | MMLU Wilson lower > 30% and both GSM8K lower bounds > 10% |
| Fresh confirmation | ARC-Challenge Wilson lower > 30% for any arm claiming recovery |
| Allowed costs | Books − C ≤ +0.10 upper bound; GSM8K − C ≥ −3 points lower bound; retrieval accuracy ≥ 90% |
| "Format only" flag | Letter mass up but pairwise < 52%: report as format learning, not recovery |

**Compute estimate** (from measured 3,030 tokens/s and 6.08 h per 65.5M-token run):
- Multiple-choice teacher generation: about 20k short responses at ~800 tokens/s, ≤ 0.5 h.
- M and U training: 2 × 6.1 h. M+U adds 6.1 h. LR probe for U: about 0.3 h.
- Export and evaluation: about 0.3 h per arm, including the ARC sets.
- **Total: about 13.5 GPU-h for 2 arms, about 20 GPU-h with the interaction arm.** Local, no rental. Set a 22 h ceiling.

**Stop/go before training** (from §5):
- **D2 at 90% or above and D1 shows no cloze signal** (read-out intact, knowledge lost): drop arm M. Run U plus a **budget arm B** instead: continue C's final latents for +65.5M tokens on the same stream at a predeclared constant LR, ≈ 6.1 h. Only in this branch is "longer training" the informative test.
- **D3 shows a rising trajectory:** add arm B regardless.
- **D2 under 70%:** run M first. U is lower priority.

**Stop/go after the experiment:**
- **Any arm passes recovery and fresh confirmation** → freeze that recipe for 1.7B (§7).
- **M shows signal (≥ +4 pairwise) but misses the gate** → one more local iteration on the multiple-choice share and data before renting.
- **No arm moves pairwise above 52% on either MMLU or ARC** → stop iterating data at 0.8B. The evidence then points to capacity or budget (ranks 3 and 5), which only a larger budget or model can test (§7).

### Why this beats "longer", "larger" or "more chat"

- **Longer training at 0.8B.** The MMLU signal is *absent* (50.3% [49.6, 51.0]), not weak, and identical between the plain and chat arms. GSM8K appeared within the *same* budget once matching data was present, which points at coverage rather than budget. Another 6 h run without D3 cannot tell budget from coverage; D3 measures the trend from 10 existing exports in about 20 minutes. Longer training is kept as a conditional arm (B) for the outcome where it is informative.
- **More chat share.** The current corpus has 1 multiple-choice prompt in 4,722, so more share mostly adds repetition. The corpus already shows convention copying at 7.8 passes (`{{N}}`, V2), and books did not need the prose (V4). More share would likely raise GSM8K-style behaviour and memorization without touching the MMLU mechanism.
- **Larger model (1.7B rental).** It costs about $100–250 plus engineering time and changes model, architecture (Qwen3 is not hybrid) and budget at once. If it failed we could not say why. The factorial isolates read-out, frozen tensors and (conditionally) budget at no rental cost, and its winning recipe is what a 1.7B run should use.

---

## 7. Should we rent for 1.7B now?

**🟡 Not yet.**
- QAT08's frozen rental rule ("training helps, ≥ 50% closure") is technically met. But QAT08's own recommendation was to "fix the chat gap at 0.8B first, then rent". That gap is half-closed: generation and GSM8K-style math are back; multiple-choice knowledge read-out is at chance.
- A 1.7B run with today's recipe would most likely repeat the same unexplained MMLU failure at higher cost. That would teach little, because we still could not separate read-out from knowledge from frozen tensors.

**Evidence that would justify renting** (either is sufficient):
1. **A working recipe.** A 0.8B arm passes the recovery gate and fresh ARC confirmation with no book cost. Rent to compare that recipe against Prism's Ternary-Bonsai-1.7B, with predeclared gates relative to Prism in this harness (Prism versus FP: MMLU −1.4, GSM8K −7.3 points).
2. **A demonstrated capacity or budget limit.** Diagnostics and the factorial show read-out is intact (D2 ≥ 90%), M and U do not help, and the knowledge signal rises with tokens (D3 or arm B). That is the case 0.8B on 12 GB cannot resolve, and the 1.7B run becomes the informative experiment. It should include the U change if U helped books, and use a token ladder (for example 250M / 500M / 1B) so the trend is measured rather than assumed.

The 1.7B design should keep MMLU-Redux for continuity, add the fresh ARC sets and the D2 read-out test, and report calibrated pairwise as a primary signal.

---

## 8. Suggested corrections to QAT08_CHAT.md (not applied; for the author)

1. State that MMLU shows **no discriminative signal** (calibrated pairwise 50.3%), and that the +1.39-point paired MMLU difference is a letter-bias shift.
2. Replace "0 parse fallbacks" with the answer-format counts: `{{N}}` 760, `\boxed` 33, neither 526. Note that the student copies the training math convention.
3. Add the kept-chat category composition and the share of each category dropped at the token limit.
4. Note that the math prompts are GSM8K-style word problems (near-domain), with the similarity analysis.
5. Note that the chat arm's FineWeb is the first 49.2M tokens of QAT08's stream (equal total tokens, not equal prose).
6. Add a status pointer to the stale "Decision pending" header in `QAT08_CHAT_DESIGN.md`.
7. Describe the training approval as auto-bound by the controller after the automated protocol seal.

---

## Appendix — how my numbers were produced (read-only, CPU)

- **MMLU calibration and pairwise signal.** For each arm in `work/qat08_chat/scores/<arm>-mmlu.jsonl`: renormalize the four letter log-probabilities, subtract the arm's average per letter (also leave-one-subject-out), take the argmax, and compute pairwise gold-greater-than-wrong per item. Bootstrap with 200 resamples for the chat arm's interval.
- **Embedding geometry.** `make_gptq_pilot.Hadamard` (CPU) folds `embed_tokens` rows 32–35 and 4,000 reference rows; `qat_08b.quantize` applies top-86 and absmean; unfold and compare cosines.
- **GSM8K.** Re-parse with `gsm8k_17b.parse` (0 mismatches). Random-gold baseline: 50 permutations. Format counts are regex counts on `text`. Near-domain analysis: TF-IDF (word 1–2-grams) cosine between decoded GSM8K questions and the 1,500 `synthetic_math` prompts.
- **Data.** `kept_chats.json`, `teacher.jsonl` and `prompts.jsonl` joined by ID; token shares by category.
- **Statistics.** Paired t over 12 books (11 df); Wilson intervals at z = 1.96.

---

## Addendum (2026-10-03, found while writing the execution plans)

🟢 **A small train/runtime mismatch in the frozen pipeline.**

- **What happens.** `transformers` loads the 18 GDN norm weights (`linear_attn.norm.weight`, class `Qwen3_5RMSNormGated`) in **BF16**, even with `dtype=torch.float32`. The checkpoint stores them as F32.
- **Where it shows up.** `qat_08b.build_student` calls `.float()` after loading, so the QAT08 and chat students trained with BF16-rounded copies. The exported GGUFs (copied from the converter's template) carry the exact F32 values.
- **What was measured.** Every one of the 169 FP non-projection payloads is byte-identical to the template *except these 18*. Each loaded value equals `exact.to(bfloat16)`.
- **Size.** The rounding is at most about 0.4% relative on these gains. 🟡 It is very unlikely to explain any of the MMLU gap, but it means the runtime function differs slightly from the trained function.
- **Plan B handling.** Arms that train FP tensors start from and export the exact values, and a test pins this behaviour. See `test_hf_load_rounds_only_gdn_norms` in [the experiment plan](../../../../docs/superpowers/plans/2026-10-03-qat08-mcu-experiment.md).

🟢 **Pairwise statistic definition.** The execution plan freezes an item-level definition: the mean, over items, of the share of wrong letters the gold letter beats, with a normal 95% item interval. On the chat arm it gives **50.36% [49.35, 51.38]**, against FP's 68.65%. §3 V1 quoted 50.3% [49.6, 51.0] from a per-letter average with a bootstrap. The conclusion — no discriminative signal — is unchanged.

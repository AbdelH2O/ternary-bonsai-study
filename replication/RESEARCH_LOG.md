# Research log — Bonsai ternary replication

A chronological record of every step: what question was asked, what was done, what evidence came out, and what was decided, by whom, and why. Its purpose is a verifiable line of reasoning from the first question to every claim, so the work can be shared or shown as a portfolio project. Newest entries at the bottom.

**How to read and write it.** The rules are in [AGENTS.md](AGENTS.md#recording-every-step). Each entry links the report, script and result file it relies on, with SHA-256 prefixes for artifacts that later claims depend on. Detailed numbers live in the linked reports; this log keeps the chain of reasoning. Failed attempts, incidents and dead ends are recorded like successes. Entries are added, never rewritten; corrections are new entries that point back.

**Backfill note.** Entries before 2026-10-06 were reconstructed on 2026-10-06 from the linked reports, which were written at the time and carry their own frozen hashes. They are summaries, not contemporaneous notes. From 2026-10-06 onward, entries are written as the work happens.

Evidence tags as in the reports: 🟢 measured; 🟡 inferred; 🟣 proposed.

---

## Backfilled: 2026-09-29 to 2026-10-06

### 2026-09-29 — Before replication: does Bonsai 2 27B's recurrent state accumulate errors?
- **Question:** is the compact 27B's quality loss driven by error build-up in its recurrent (GDN) state?
- **Did:** perturbation calibration, a recurrent-state pilot, validation gates, using the local Bonsai 2 27B PQ2_0 checkpoint.
- **Evidence:** [../EXPERIMENT_REPORT.md](../EXPERIMENT_REPORT.md), [../CALIBRATION_REPORT.md](../CALIBRATION_REPORT.md), [../VALIDATION_GATES_REPORT.md](../VALIDATION_GATES_REPORT.md), [../LEARNING_REFERENCE.md](../LEARNING_REFERENCE.md).
- **Outcome:** position-matched scoring, retrieval controls and freeze discipline carried into the replication. The causal findings did not identify a quantization recipe ([REPLICATION_PLAN.md](REPLICATION_PLAN.md) §1).

### 2026-09-30 — Replication plan and Phase 0 (format and mapping)
- **Question A:** which rules reproduce Bonsai's stored codes and scales from a public Qwen checkpoint? **Question B:** can we build a comparable ternary model ourselves?
- **Did:** wrote the plan, revised after two independent reviews ([REVIEW_astra.md](reviews/REVIEW_astra.md), [REVIEW_fable.md](reviews/REVIEW_fable.md)). Phase 0: tensor map, codec and layout fixtures, native F32 Hadamard fold tests on small hybrid models.
- **Evidence:** [REPLICATION_PLAN.md](REPLICATION_PLAN.md), [PHASE0_RESULTS.md](PHASE0_RESULTS.md).
- **Outcome:** runtime and fold mechanism validated on small models; the exact 27B FP-function gate stays open (the ancestor's full checkpoint wasn't run).

### 2026-09-30 — 0.8B quality pilot: one-shot rounding rules
- **Did:** absmax, fixed-86 and least-squares PQ2_0 assignment, unrotated and H512, on Qwen3.5-0.8B.
- **Evidence:** [QUALITY_PILOT.md](QUALITY_PILOT.md).
- **Outcome:** every packed arm far from FP quality.

### 2026-10-01 — Data-aware (GPTQ-style) arm, then refinements
- **Did:** frozen GPTQ-style arm; then 4× calibration, within-layer sequencing, and Bonsai 2's 42-zero budget.
- **Evidence:** [GPTQ_PILOT.md](GPTQ_PILOT.md), [GPTQ_V2.md](GPTQ_V2.md).
- **Outcome:** GPTQ closed 78% of the least-squares-to-FP gap but stayed +1.49 nat/token above FP. Refinements gave no material gain; the 42-zero budget cost +0.45. That met the predeclared entry criterion for the 1.7B comparison and for proposing training.

### 2026-10-02 — Phase 1: matched 1.7B comparison with Prism's model
- **Question:** on the same ancestor and in the identical format, how far is one-shot rounding from Prism's Ternary-Bonsai-1.7B?
- **Did:** frozen protocol (sha `b70dec33…`). Prism's file reproduced byte for byte from its unpacked checkpoint, confirming identical format. Six arms scored on books, MMLU-Redux, GSM8K and retrieval; static weight comparison.
- **Evidence:** [PHASE1_17B.md](PHASE1_17B.md), [results/q17b/](results/q17b/).
- **Outcome:** large gap. The best one-shot arm is +1.31 nat/token worse and at chance on tasks, while Prism stays near FP. Prism's weights moved far from the ancestor (89% sign agreement), pointing to end-to-end training. Phase 2 (training) entry criterion met.

### 2026-10-02 — Bounded quantization-aware distillation at 0.8B (QAT08)
- **Did:** 65.5M tokens of FineWeb-Edu, forward KL to the FP teacher, two quantizer arms (protocol sha `99897718…`).
- **Evidence:** [QAT08.md](QAT08.md).
- **Outcome:** training is the missing ingredient for loss (66% of the gap closed, still improving). The 42-zero budget cost nothing after training. Chat-format capability didn't recover.

### 2026-10-03 — Chat-aware data (QAT08_CHAT)
- **Did:** 25% teacher-answered chat data (design sha `185631c7…`, protocol sha `0a610877…`).
- **Evidence:** [QAT08_CHAT_DESIGN.md](QAT08_CHAT_DESIGN.md), [QAT08_CHAT.md](QAT08_CHAT.md).
- **Outcome:** GSM8K rose from 0.2% to 14.6%, but the predeclared recovery rule failed: MMLU stayed at chance.

### 2026-10-05 — Code drift (descriptive)
- **Question:** do our students move their codes as much as Prism's training did?
- **Evidence:** [CODE_DRIFT.md](CODE_DRIFT.md).
- **Outcome:** ours moved 14% of codes after 65.5M tokens, Prism's 39%. Proposed a full-length higher-learning-rate arm as the cheapest untested lever.

### 2026-10-05 to 2026-10-06 — Read-out diagnostics, then multiple-choice data (QAT08_MCU)
- **Did:** read-out diagnostics (rule result `harness_invalid`, amended after disclosure to `readout_broken`). Then arms M (multiple-choice data), U (trained norms and gates) and B (continued training); design v2 sha `8247ef1e…`, protocol sha `b99f4bdd…`.
- **Evidence:** [DIAG_READOUT.md](DIAG_READOUT.md), [QAT08_MCU_DESIGN.md](QAT08_MCU_DESIGN.md), [QAT08_MCU.md](QAT08_MCU.md).
- **Outcome:** M met the recovery gate, partly: MMLU-Redux 35.9%, GSM8K 16.3%. Frozen recommendation `rent_1p7b_with_recipe`, which authorized no rental.

### 2026-10-06 — MC12 (running at the time of writing)
- **Did:** prompt-robustness scoring, then 12.5% multiple-choice data at M's recipe (design sha `336237b7…`, protocol sha `de416a69…`).
- **Evidence:** [QAT08_MC12_DESIGN.md](QAT08_MC12_DESIGN.md).

---

## Contemporaneous entries from 2026-10-06

### 2026-10-06 — Where to get compute for the 1.7B run
- **Question (user):** could the free-tier strategy in a Gemini-written report fit the 1.7B training?
- **Did:** checked each technique against our recipe ([qat_08b.py](qat_08b.py), [PHASE1_17B.md](PHASE1_17B.md)).
- **Findings:**
  - chunked KL loss is already implemented;
  - QLoRA doesn't apply, because the deliverable is ternary weights and LoRA on an NF4 base reduces to one-shot rounding;
  - fp16 on T4 would change the recipe;
  - the real limit is GPU hours, not memory.
- **Options proposed:** local RTX 5070 with optimizer offload, or Modal's free plan credits running the recipe unchanged.
- **Decision (user):** "Let's try with Modal's free credits."

### 2026-10-06 — 1.7B profiling harness and H100 profile
- **Did:** built [qat_17b.py](qat_17b.py) (sha `59e7761d…`), the 1.7B adapter around the frozen 0.8B code, and [modal_qat17b.py](modal_qat17b.py) (sha `4d49b1d9…`). `prep`:
  - 196 projection names match the Phase 1 template;
  - 2.1M-token Qwen3-tokenized profile stream;
  - base checksums recorded in [prep_record.json](work/qat17b/prep_record.json) (sha `d0540da6…`).
- **Local check (run by the user, outside the sandbox):** the cached-ternary-weight path is bit-identical in forward value and gradient for absmean and top86.
- **Incident 1:** I reported the local GPU as broken. It was the bb sandbox hiding `/dev/nvidia*`; the user corrected this.
- **Incident 2:** the first Modal run finished but failed to return its result: a torch object couldn't be unpickled on the torch-free client. Fixed by returning JSON text and adding `--fetch`. The result was recovered from the volume without re-running.
- **Evidence:** [profile](work/qat17b/profile_NVIDIA_H100_80GB_HBM3.json) (sha `8c77b77e…`). 🟢 Pinned Qwen3-1.7B verified file by file on Modal.
  - Best safe setting: micro-batch 8 × 4, no checkpointing, cached weights; 2.63 s/step, 12,470 tok/s, 53 GiB peak.
  - Caching was 18% faster, with losses identical to every printed digit.
  - Step-0 KL 12.8 (0.8B: 8.3).
- **Conclusion:** 🟢 2,000 steps ≈ 1.46 h ≈ $6–7 on Modal.

### 2026-10-06 — 1.7B design draft v0
- **Question (user):** draft the protocol, with training data and setup that maximize the odds of a result in the fewest iterations.
- **Did:** wrote [QAT17B_DESIGN.md](QAT17B_DESIGN.md) v0:
  - one extendable 4,000-step run;
  - an added math slot with correctness-filtered teacher answers;
  - absmean quantizer, H1024 basis, warmup–stable–decay schedule;
  - free local pre-flight checks;
  - abort rules and a budget of about $20.
- **Status:** draft; the user's choices are open.

### 2026-10-06 — Is there more to learn about Prism's method? (forensics)
- **Question (user):** before more distillation work, are we sure there's nothing more to discover in Prism's quantization technique?
- **Did:**
  - downloaded Prism's ternary g64 and 1-bit 1.7B releases (Q2_0_g64 sha `6d0ecb3d…`, Q1_0 sha `3d7c6c90…`);
  - compared all 1.72B weights with the g128 release and the ancestor ([prism_variants_17b.py](prism_variants_17b.py));
  - ran a magnitude-conditioned lineage test with our own independent 0.8B runs as control ([prism_lineage_17b.py](prism_lineage_17b.py));
  - compared absmean zero fraction and scales against the untrained ancestor;
  - compared GSM8K answer styles from Phase 1 outputs;
  - ran a CPU style probe with Qwen3-4B/8B Q8_0 (sha `8c2f07f2…`, `408b9555…`) ([teacher_style_probe.py](teacher_style_probe.py)).
- **Evidence:** [PRISM_FORENSICS_17B.md](PRISM_FORENSICS_17B.md); [results/prism_variants_17b.json](results/prism_variants_17b.json) (sha `d2ca83df…`), [results/prism_lineage_17b.json](results/prism_lineage_17b.json) (sha `ea7ed14e…`), [results/teacher_style_probe.json](results/teacher_style_probe.json) (sha `5556f2c3…`).
- **Findings:** 🟢
  - the g64 file is a lossless repack;
  - every Prism scale is a BF16 value;
  - both releases share a data-free embedding;
  - Prism's weights grew about 2× and became more peaked (39.9% zeros vs 31.7% untrained; ours 31.8%), which extends [CODE_DRIFT.md](CODE_DRIFT.md);
  - flip coincidence between the 1-bit and ternary models is explained by a shared objective (our control shows as much), so there's no lineage evidence;
  - Prism's answer style matches neither Qwen3-1.7B, 4B nor 8B (same first line 12% / 15% / 4%).
- **Reading:** 🟡 Prism's targets likely included non-Qwen-style responses.
- **Incidents:**
  - I first read the flip coincidence as possible warm-starting; the control refuted it before the write-up.
  - The style probe was restarted at 100 instead of 200 questions for CPU time. Stopping it killed my own shell and orphaned a llama-server, which I stopped by PID; the separate MC12 training process was untouched.
  - First-token probabilities weren't captured (response-format mismatch).
  - Finding 4 initially didn't credit CODE_DRIFT.md; corrected in the note.

### 2026-10-06 — 1.7B design draft v0.1
- **Did:** folded the forensics proposals into [QAT17B_DESIGN.md](QAT17B_DESIGN.md) v0.1 (sha `803054f6…`), marked *(v0.1)*, with a revision history:
  - scales rounded through BF16;
  - weight-movement diagnostics against Prism's values;
  - a 300-step learning-rate probe adding 4e-4, ties broken toward the larger rate;
  - teacher stays Qwen3-1.7B;
  - multiple-choice share waits for MC12.
- **Status:** proposals, pending the user's approval.

### 2026-10-06 — Record keeping
- **Request (user):** keep recording each step so the experiments carry proof and a logical line of reasoning, now and in the future, for sharing or a portfolio.
- **Did:** started this log; added logging rules to [AGENTS.md](AGENTS.md#recording-every-step); saved the preference to agent memory.
- **Open:** `analysis/` isn't under version control (`git status`: untracked), so nothing yet gives independent timestamps or history. A tracked repository would. That needs the user's decision on where it lives and what is published.

### 2026-10-06 — Version control for the research
- **Decision (user):** "Let's do a separate repo." This follows the open item in the previous entry.
- **Did:** made `analysis/bonsai2/` its own git repository, in place, because the scripts depend on relative paths into this checkout (pinned runtime, models). It covers the 27B experiments and the replication.
  - [`.gitignore`](../.gitignore) excludes `work/` directories and large or regenerable artifacts (`*.gguf`, `*.bin`, `*.pt`, `*.npy`, `*.safetensors`, `*.parquet`, `*.u32`, logs). Their hashes stay in the tracked reports.
  - About 1,700 files and 456 MB of text are tracked; the largest is 11.5 MB.
  - A scan for API tokens, keys and email addresses found none.
  - Author identity is the user's global git config (AbdelH2O, university address).
- **Note:** everything before this first commit was created without version control. Its dates rest on the reports themselves and on file times, not on commits.
- **Open:** a remote (GitHub or elsewhere) and public vs private visibility await the user's decision.

### 2026-10-06 — Goal question: replicate Prism's model, or match its results?
- **Question (user):** "I'm also starting to question whether we want to replicate prism's model directly or simply trying to match its performance/results."
- **Context:** [REPLICATION_PLAN.md](REPLICATION_PLAN.md) §1 already separates Question A (recover Prism's mapping and process) from Question B (build a comparable model by any method). The 1.7B design is mostly Question B, with some choices made to mirror Prism.
- **Status:** recommendation given in conversation (make B the headline goal and keep A only where it serves B); decision pending.

### 2026-10-06 — Goal decided: match Prism's results; design v0.2
- **Decision (user):** "Update the design along these lines with the revision history." This accepts the recommendation in the previous entry: make Question B (matching results) the headline goal.
- **Did:** [QAT17B_DESIGN.md](QAT17B_DESIGN.md) v0.2 (sha `3c8092a5…`), changes marked *(v0.2)* with a revision-history row:
  - goal statement: base model, format and evaluation fixed; method free;
  - scope limited to Prism's ternary 1.7B;
  - BF16 scale rounding made optional, default off;
  - IFEval and HumanEval+ added as reported breadth metrics, FP and Prism scored on the same harness, HumanEval+ code run only in a network-less container;
  - top reading renamed `matched`; breadth must be reported next to every reading;
  - a hard-label data slot is allowed only if a cheap 0.8B test shows it raises scores.
- Added a goal-update note at the top of [REPLICATION_PLAN.md](REPLICATION_PLAN.md).
- **Status:** design still a draft; the open choices (token budget, thresholds, rejection sampling, teacher runtime) remain the user's.

### 2026-10-06 — Publication review and public repository
- **Decision (user):** "Create the remote and put it as public." During the review the user also instructed that their compute budget not be mentioned.
- **Did, before anything was pushed:**
  - removed every mention of the user's compute budget from the design, agent guide and this log;
  - stopped tracking two personal planning files (a compute and funding report and its grants notes). They discuss budget and the user's employer, aren't research evidence, and stay on disk, listed in `.gitignore`;
  - added a public orientation section to [README.md](../README.md): purpose, non-affiliation with Prism ML, where to start, and third-party content sources;
  - re-scanned for tokens, keys and email addresses: none.
- **History:** the three earlier local commits contained the removed text and were never pushed. They were replaced by a single initial commit, so the public history begins here. Entries above this one describe work done before version control existed.
- **Remote:** a public GitHub repository under the user's account.

### 2026-10-06 — Remote created
- Public repository: <https://github.com/AbdelH2O/ternary-bonsai-study>, branch `main`, first commit `c10cc14`. From here on, every log entry is committed and pushed with the files it refers to.

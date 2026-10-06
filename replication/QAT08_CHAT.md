# Chat-aware quantization-aware distillation on Qwen3.5-0.8B — 2026-10-03

**Update 2026-10-06:** the read-out diagnostics ([DIAG_READOUT.md](DIAG_READOUT.md)) and the frozen multiple-choice follow-up are reported in [QAT08_MCU.md](QAT08_MCU.md). Multiple-choice teacher data (arm M) met the frozen recovery gate, but only partially relative to FP.

**Decision (frozen rules): Chat capability did not meet the predeclared recovery rule.**

- 🟢 Chat recovery: **fail** under the joint MMLU/GSM8K rules below.
- 🟢 Book cost versus QAT08's 42-zero arm: **+0.005 nat/token [-0.012, +0.021]**; allowed upper bound +0.10, **pass**.
- 🟢 Overall amendment: **did not meet all acceptance gates**.

This amends [QAT08.md](QAT08.md). The [approved design](results/qat08_chat/design.json) and [final protocol](results/qat08_chat/protocol.json) (SHA-256 `0a6108773dfa513652beb82cffcad2ee3f5e7e7d57f6f9d0d8a79fe22f255519`) fixed the data contract, exact training-data hashes, arms, held-out cases and thresholds before student training. The [execution amendment](results/qat08_chat/resume_amendment_v1.json) preserved that scientific design while adding reboot-safe controls. Evidence tags: 🟢 measured; 🟡 inferred; 🟣 proposed.

## Recipe and data

One 42-zero student starts from the same folded FP projection weights as QAT08, with the same fixed ternary embedding and frozen non-projection tensors. H512 basis, full-vocabulary all-position forward KL, 65.536M tokens, 2,000 steps, LR 1e-4, warmup/cosine schedule, BF16 AdamW moments, clipping and checkpoints are identical. **Only the input stream changes.**

| Data item | Value |
|---|---|
| Source | [NVIDIA Daring-Anteater synthetic subsets](https://huggingface.co/datasets/nvidia/Daring-Anteater/tree/ae79f8ac44cf185fbd3250dbe46057a7f7c4ec40); CC BY 4.0; credit NVIDIA / Wang et al., *HelpSteer2* (2024) |
| Prompt candidates | 10,000 first user turns; original source answers discarded |
| Responses | FP F32 GGUF teacher, Qwen3.5 chat template, thinking disabled, greedy, 768-token limit |
| Retained complete chats | 4,722 |
| Unique chat tokens | 2,098,230 |
| Training mix | 25% chat (16,384,000 tokens), 75% FineWeb (49,152,000) |
| Chat-corpus passes | 7.81; repeated in fixed prompt order, complete transcripts concatenated without padding |
| Rejected generations | `{"not_naturally_terminated_or_empty": 5278}` |

The 25% mix provides 16.4M tokens of chat-format exposure while retaining most of the prose budget. It was frozen from the format hypothesis, not selected on evaluation. Generation-filter counts and repetition are limitations, not additional independent training examples.

## Results

🟢 [Full results](results/qat08_chat/results.json): twelve fresh held-out books, 5,330 MMLU-Redux items, 1,319 GSM8K items and 200 retrieval registries. Book values are all rescored on this new set; unchanged task/retrieval comparator outputs are reused only after case/model/output hash checks. Do not compare absolute book means across experiments.

| Arm | Book NLL (nat/token) | MMLU | GSM8K | Retrieval accuracy / margin |
|---|---:|---:|---:|---:|
| FP | 3.353 | 48.2% | 55.8% | 100.0% / 8.41 |
| GPTQ | 4.940 | 22.5% | 0.1% | 55.0% / 0.28 |
| QAT08 free | 3.865 | 22.5% | 1.1% | 94.5% / 3.44 |
| QAT08 42-zero | 3.838 | 22.5% | 0.2% | 90.5% / 3.25 |
| **Chat 42-zero** | 3.843 | 23.9% | 14.6% | 94.5% / 4.22 |

🟢 **Predeclared chat gates** (two-sided 95% Wilson intervals): MMLU lower bound >30%; both GSM8K lower bounds >10%. The old degenerate outputs' maximum parsing-coincidence upper bound was 1.774%, well below the GSM8K threshold.

| Metric | Accuracy [95% interval] | Gate |
|---|---:|---|
| MMLU | 23.86% [22.74, 25.03] | fail |
| GSM8K | 14.56% [12.76, 16.56] | pass |
| GSM8K, unfinished outputs count wrong | 14.40% [12.61, 16.40] | pass |

🟢 **Behavior diagnostics:** MMLU predicted letters `{"A": 4741, "B": 22, "C": 149, "D": 418}`, mean total letter probability 77.95%. GSM8K stops `{"eos": 1255, "limit": 64}`, 0 parse fallbacks and 351.6 mean output tokens. Raw generations remain in `results/qat08_chat/gsm8k/` for inspection.

🟢 **Validation book curve** (descriptive; final checkpoint fixed):

| Checkpoint | NLL |
|---|---:|
| init | 10.7152 |
| t8m | 4.6136 |
| t16m | 4.2665 |
| t33m | 4.0748 |
| final | 3.8767 |

🟢 **Contamination:** `{"gsm8k": {"cases": 1319, "cases_with_any_13gram_overlap": 4, "cases_with_majority_13gram_overlap": 0, "maximum_overlap_fraction": 0.01639344262295082, "mean_overlap_fraction": 4.351901772964589e-05}, "heldout_books": {"cases": 24, "cases_with_any_13gram_overlap": 0, "cases_with_majority_13gram_overlap": 0, "maximum_overlap_fraction": 0.0, "mean_overlap_fraction": 0.0}, "mmlu": {"cases": 5330, "cases_with_any_13gram_overlap": 36, "cases_with_majority_13gram_overlap": 0, "maximum_overlap_fraction": 0.14583333333333334, "mean_overlap_fraction": 0.00019438937938894837}, "retrieval": {"cases": 400, "cases_with_any_13gram_overlap": 0, "cases_with_majority_13gram_overlap": 0, "maximum_overlap_fraction": 0.0, "mean_overlap_fraction": 0.0}, "validation_books": {"cases": 6, "cases_with_any_13gram_overlap": 0, "cases_with_majority_13gram_overlap": 0, "maximum_overlap_fraction": 0.0, "mean_overlap_fraction": 0.0}}`. Every actual evaluation case was checked against the exact mixed stream with the same rolling 13-token hash as QAT08; no majority-overlap case was allowed through. Smaller overlap is disclosed, and literal checks cannot rule out paraphrases or ancestor pretraining exposure.

## Interpretation and limits

🟡 This fixed mixture and budget were insufficient to establish chat recovery. That does not rule out chat-aware distillation generally: the frozen embedding, limited unique chat data, mixture and token budget remain possible constraints. No alternative threshold, checkpoint or mixture was selected after seeing these scores.

🟣 A larger-scale training comparison remains a separate decision; this amendment does not authorize renting hardware or claim FP equivalence. The remaining measured gaps and this single-model, single-mixture result must guide that decision.

Limits: one 0.8B hybrid model and training run; repeated synthetic chat data; embedding and non-projection tensors frozen; MMLU uses letter logits and GSM8K one fixed greedy prompt/parser; intervals treat books, benchmark items and registries as independent. No claim about tools, vision or PrismML's training history follows.

## Compute and incidents

🟢 Recorded stage durations (hours): `{"analyze": 0.002, "curve": 0.013, "export-final": 0.002, "export-init": 0.002, "export-t16m": 0.003, "export-t33m": 0.002, "export-t8m": 0.003, "generate": 1.965, "gsm8k": 0.09, "heldout": 0.044, "pack": 0.002, "protocol": 0.011, "train": 6.083}`. Pilot generation took 58.5 seconds; planned total ceiling was 12 GPU-hours. Stage wall time includes startup/I/O and is conservative relative to active GPU compute. Not independently reprofiled; the unchanged QAT08 path measured 7.9 GiB peak.

- Title checks caught wrong remembered Gutenberg IDs: 668 was a dictionary and 427 was The Great War Syndicate. Downloads preserved as rejected artifacts; corrected to 6688 and 4274 before cases or scoring. A parallel title check caught 918 (Sketches of Young Gentlemen); Villette corrected to 9182 before download for the experiment.
- Prompt preparation first stopped because only 135 JSON-format prompts met the fixed 384-token length cap, below the draft 500 quota. Before any generation or evaluation, the JSON quota was set to 100 and the general-conversation quota to 6400, keeping 10,000 prompts total.
- A CPU transcript check found that this installed tokenizer returns an encoding object rather than an ID list for tokenize=True. The packing assertion now renders text then encodes it, matching the existing harness. The pilot transcripts were checked before the design freeze.
- The user requested reboot-safe pause/resume after approving the design. A separately hashed execution amendment added durable storage, optimizer-boundary pauses, RNG restoration and a restart controller; the original frozen source, recipe and decisions stayed unchanged. A separate-process CPU fixture passed exact optimizer/loss restart parity.
- Training pauses saved fresh checkpoints at steps 315.
- Graceful execution pauses: generate, train; later invocations resumed durable progress.

## Reproduction

From `analysis/bonsai2/replication`, using the project venv. GPU execution is unsandboxed; the controller creates systemd user units.

```bash
python freeze_qat08_chat.py verify-design
python qat08_chat_control.py start      # full pipeline; requires the recorded user approval
python qat08_chat_control.py status
python qat08_chat_control.py pause      # save at an optimizer boundary, then exit
python qat08_chat_control.py resume     # same command after a reboot; recreates the unit
```

The controller generates/resumes teacher data, packs the fixed mixture, seals `protocol.json`, trains/resumes, exports, scores the fixed harness, analyzes and writes this report. The scientific functions are the original hash-frozen scripts. The separate runtime wrapper is hashed in the execution amendment. Logs/state are in `work/qat08_chat/run.log`, `run_state.json`, `execution.jsonl` and `qat_chat_42/train.jsonl`; optimizer checkpoints are `qat_chat_42/resume.pt`. Graceful pause saves all completed steps; a sudden power loss can replay up to the original 20-minute periodic checkpoint interval. No commit was made by this pipeline.

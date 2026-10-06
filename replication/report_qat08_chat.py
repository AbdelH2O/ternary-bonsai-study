"""Write the requested evidence report after the frozen evaluation completes."""
from __future__ import annotations

import datetime
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/qat08_chat"
WORK = ROOT / "work/qat08_chat"


def atomic_document(path, text):
    from qat08_chat_runtime import fsync_dir
    tmp = path.with_name(path.name+".pending")
    with tmp.open("w") as f:
        f.write(text); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
    fsync_dir(path.parent)


def write_report():
    from score_17b import sha
    p = json.loads((OUT / "protocol.json").read_text())
    r = json.loads((OUT / "results.json").read_text())
    assert r["protocol_sha256"] == sha(OUT / "protocol.json")
    amendment = json.loads((OUT / "resume_amendment_v1.json").read_text())
    for name, digest in amendment["implementation_sha256"].items():
        assert sha(ROOT / name) == digest
    date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    decision = ("Chat capability recovered, within the allowed book-loss cost." if r["amendment_success"] else
                "Chat capability recovered, but the allowed book-loss cost was exceeded." if r["chat_recovered"] else
                "Chat capability did not meet the predeclared recovery rule.")
    d = p["data"]
    a = "qat_chat_42"
    cost = r["chat_minus_comparators"]["qat_42"]["book_nll"]
    events = [json.loads(l) for l in (WORK / "execution.jsonl").read_text().splitlines()]
    finished = [x for x in events if x.get("event") == "stage_finished"]
    hours = {}
    for e in finished:
        hours[e["stage"]] = hours.get(e["stage"], 0) + e["seconds"]/3600
    # An abrupt reboot can omit an end event. Do not present such totals as complete.
    starts = sum(x.get("event") == "stage_started" for x in events)
    incomplete_sessions = starts-len(finished)
    table = []
    labels = {"fp": "FP", "gptq": "GPTQ", "qat_free": "QAT08 free", "qat_42": "QAT08 42-zero", a: "**Chat 42-zero**"}
    for arm in p["heldout_arms"]["books_mmlu_retrieval"]:
        table.append(f"| {labels[arm]} | {r['book_nll'][arm]:.3f} | {r['mmlu'][arm]['accuracy']:.1f}% | "
                     f"{r['gsm8k'][arm]['accuracy']['mean']:.1f}% | {r['retrieval'][arm]['accuracy']:.1f}% / {r['retrieval'][arm]['margin']:.2f} |")
    rows = "\n".join(table)
    intervals = "\n".join(f"| {label} | {ci['mean']:.2f}% [{ci['lo']:.2f}, {ci['hi']:.2f}] | {'pass' if r['gates'][key] else 'fail'} |"
        for label, ci, key in [("MMLU",r["mmlu"][a]["ci"],"mmlu"),
                              ("GSM8K",r["gsm8k"][a]["accuracy"],"gsm8k"),
                              ("GSM8K, unfinished outputs count wrong",r["gsm8k"][a]["naturally_terminated_correct"],"gsm8k_without_nontermination_credit")])
    baseline = max(v["hi"] for v in p["gsm8k_parsing_coincidence_baseline"].values())
    curve = "\n".join(f"| {tag} | {r['curve'][a+'-'+tag]:.4f} |" for tag in ("init","t8m","t16m","t33m","final"))
    mm = r["mmlu"][a]
    gs = r["gsm8k"][a]
    interpretations = ("🟡 This result supports the chat-data hypothesis on this model and budget: changing the input mix recovered chat-mode behavior while the model, loss and optimizer remained fixed. It does not identify PrismML's recipe or show that 25% is optimal." if r["chat_recovered"] else
                      "🟡 This fixed mixture and budget were insufficient to establish chat recovery. That does not rule out chat-aware distillation generally: the frozen embedding, limited unique chat data, mixture and token budget remain possible constraints. No alternative threshold, checkpoint or mixture was selected after seeing these scores.")
    incidents = list(p["pre_freeze_incidents"])
    incidents.append("The user requested reboot-safe pause/resume after approving the design. A separately hashed execution amendment added durable storage, optimizer-boundary pauses, RNG restoration and a restart controller; the original frozen source, recipe and decisions stayed unchanged. A separate-process CPU fixture passed exact optimizer/loss restart parity.")
    pauses = [x for x in events if x.get("event") == "training_checkpoint_on_pause"]
    if pauses:
        incidents.append("Training pauses saved fresh checkpoints at steps " + ", ".join(str(x["step"]) for x in pauses) + ".")
    lifecycle_pauses = [x for x in finished if x.get("outcome") == "paused"]
    if lifecycle_pauses:
        incidents.append("Graceful execution pauses: " + ", ".join(x["stage"] for x in lifecycle_pauses) + "; later invocations resumed durable progress.")
    for e in events:
        if e.get("event") == "artifact_preserved":
            incidents.append(f"Preserved interrupted artifact {e['file']}: {e['reason']}; retained copy: {e['archive']}.")
    if incomplete_sessions:
        incidents.append(f"{incomplete_sessions} stage start(s) lack a durable finish event. Recorded GPU-time totals are lower bounds; interrupted time cannot be reconstructed exactly from these records.")
    if gs["parse_fallback"]:
        incidents.append(f"The unchanged llama-server streaming parse fallback was used on {gs['parse_fallback']} new GSM8K outputs.")
    incident_text = "\n".join("- " + x for x in incidents)
    memory = "Not independently reprofiled; the unchanged QAT08 path measured 7.9 GiB peak."
    body = f"""# Chat-aware quantization-aware distillation on Qwen3.5-0.8B — {date}

**Decision (frozen rules): {decision}**

- 🟢 Chat recovery: **{'pass' if r['chat_recovered'] else 'fail'}** under the joint MMLU/GSM8K rules below.
- 🟢 Book cost versus QAT08's 42-zero arm: **{cost['mean']:+.3f} nat/token [{cost['lo']:+.3f}, {cost['hi']:+.3f}]**; allowed upper bound +0.10, **{'pass' if r['gates']['book_cost'] else 'fail'}**.
- 🟢 Overall amendment: **{'success' if r['amendment_success'] else 'did not meet all acceptance gates'}**.

This amends [QAT08.md](QAT08.md). The [approved design](results/qat08_chat/design.json) and [final protocol](results/qat08_chat/protocol.json) (SHA-256 `{r['protocol_sha256']}`) fixed the data contract, exact training-data hashes, arms, held-out cases and thresholds before student training. The [execution amendment](results/qat08_chat/resume_amendment_v1.json) preserved that scientific design while adding reboot-safe controls. Evidence tags: 🟢 measured; 🟡 inferred; 🟣 proposed.

## Recipe and data

One 42-zero student starts from the same folded FP projection weights as QAT08, with the same fixed ternary embedding and frozen non-projection tensors. H512 basis, full-vocabulary all-position forward KL, 65.536M tokens, 2,000 steps, LR 1e-4, warmup/cosine schedule, BF16 AdamW moments, clipping and checkpoints are identical. **Only the input stream changes.**

| Data item | Value |
|---|---|
| Source | [NVIDIA Daring-Anteater synthetic subsets](https://huggingface.co/datasets/nvidia/Daring-Anteater/tree/ae79f8ac44cf185fbd3250dbe46057a7f7c4ec40); CC BY 4.0; credit NVIDIA / Wang et al., *HelpSteer2* (2024) |
| Prompt candidates | 10,000 first user turns; original source answers discarded |
| Responses | FP F32 GGUF teacher, Qwen3.5 chat template, thinking disabled, greedy, 768-token limit |
| Retained complete chats | {d['kept_chats']:,} |
| Unique chat tokens | {d['unique_chat_tokens']:,} |
| Training mix | 25% chat ({d['chat_tokens']:,} tokens), 75% FineWeb ({d['fineweb_tokens']:,}) |
| Chat-corpus passes | {d['chat_corpus_passes']:.2f}; repeated in fixed prompt order, complete transcripts concatenated without padding |
| Rejected generations | `{json.dumps(d['excluded'], sort_keys=True)}` |

The 25% mix provides 16.4M tokens of chat-format exposure while retaining most of the prose budget. It was frozen from the format hypothesis, not selected on evaluation. Generation-filter counts and repetition are limitations, not additional independent training examples.

## Results

🟢 [Full results](results/qat08_chat/results.json): twelve fresh held-out books, 5,330 MMLU-Redux items, 1,319 GSM8K items and 200 retrieval registries. Book values are all rescored on this new set; unchanged task/retrieval comparator outputs are reused only after case/model/output hash checks. Do not compare absolute book means across experiments.

| Arm | Book NLL (nat/token) | MMLU | GSM8K | Retrieval accuracy / margin |
|---|---:|---:|---:|---:|
{rows}

🟢 **Predeclared chat gates** (two-sided 95% Wilson intervals): MMLU lower bound >30%; both GSM8K lower bounds >10%. The old degenerate outputs' maximum parsing-coincidence upper bound was {baseline:.3f}%, well below the GSM8K threshold.

| Metric | Accuracy [95% interval] | Gate |
|---|---:|---|
{intervals}

🟢 **Behavior diagnostics:** MMLU predicted letters `{json.dumps(mm['letters'], sort_keys=True)}`, mean total letter probability {100*mm['mean_letter_probability_mass']:.2f}%. GSM8K stops `{json.dumps(gs['stop_types'], sort_keys=True)}`, {gs['parse_fallback']} parse fallbacks and {gs['mean_output_tokens']:.1f} mean output tokens. Raw generations remain in `results/qat08_chat/gsm8k/` for inspection.

🟢 **Validation book curve** (descriptive; final checkpoint fixed):

| Checkpoint | NLL |
|---|---:|
{curve}

🟢 **Contamination:** `{json.dumps(p['contamination_13gram']['sets'], sort_keys=True)}`. Every actual evaluation case was checked against the exact mixed stream with the same rolling 13-token hash as QAT08; no majority-overlap case was allowed through. Smaller overlap is disclosed, and literal checks cannot rule out paraphrases or ancestor pretraining exposure.

## Interpretation and limits

{interpretations}

🟣 A larger-scale training comparison remains a separate decision; this amendment does not authorize renting hardware or claim FP equivalence. The remaining measured gaps and this single-model, single-mixture result must guide that decision.

Limits: one 0.8B hybrid model and training run; repeated synthetic chat data; embedding and non-projection tensors frozen; MMLU uses letter logits and GSM8K one fixed greedy prompt/parser; intervals treat books, benchmark items and registries as independent. No claim about tools, vision or PrismML's training history follows.

## Compute and incidents

🟢 Recorded stage durations (hours): `{json.dumps({k: round(v,3) for k,v in hours.items()}, sort_keys=True)}`. Pilot generation took 58.5 seconds; planned total ceiling was 12 GPU-hours. Stage wall time includes startup/I/O and is conservative relative to active GPU compute. {memory}

{incident_text}

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
"""
    atomic_document(ROOT / "QAT08_CHAT.md", body)
    # Read the latest live documents before editing; preserve unrelated content.
    previous = ROOT / "QAT08.md"
    text = previous.read_text()
    if "[QAT08_CHAT.md](QAT08_CHAT.md)" not in text:
        lines = text.splitlines(keepends=True)
        lines.insert(1, f"\n**Update {date}:** the frozen chat-aware data amendment is reported in [QAT08_CHAT.md](QAT08_CHAT.md).\n")
        atomic_document(previous, "".join(lines))
    plan = ROOT / "REPLICATION_PLAN.md"
    text = plan.read_text()
    if "[QAT08_CHAT.md](QAT08_CHAT.md)" not in text:
        lines = text.splitlines()
        i = next(i for i,l in enumerate(lines) if l.startswith("**Status:**"))
        lines[i] += f" Chat amendment status ({date}): {decision} Book cost versus the plain-text 42-zero arm is {cost['mean']:+.3f} nat/token [{cost['lo']:+.3f}, {cost['hi']:+.3f}] (see [QAT08_CHAT.md](QAT08_CHAT.md))."
        atomic_document(plan, "\n".join(lines)+"\n")
    print("Wrote QAT08_CHAT.md and updated QAT08.md / REPLICATION_PLAN.md", flush=True)


if __name__ == "__main__":
    write_report()

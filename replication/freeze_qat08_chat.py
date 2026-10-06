"""Freeze exact data, arms, held-out cases and decisions before chat QAT training.

Two immutable stages: `design` before user review; `protocol` after full teacher
generation and before any student training. Protocol extends the approved design
only with actual data hashes/timings and its mandatory contamination audit.
"""
from __future__ import annotations

import argparse
import collections
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np

from freeze_qat08 import ngram_hashes
from prepare_qat08_chat import DATA, OUT, PRIOR, ROOT, VALIDATION, WORK, read_cases, wilson, write_new
from score_17b import sha
import gsm8k_17b as gsm


def coincidence():
    result = {}
    for arm in ("gptq", "qat_free", "qat_42"):
        path = PRIOR / "gsm8k" / f"{arm}.jsonl"
        rows = [json.loads(l) for l in path.read_text().splitlines()]
        n = len(rows)
        preds = collections.Counter(gsm.parse(r["text"]) for r in rows)
        golds = collections.Counter(r["gold"] for r in rows)
        k = sum(gsm.correct(gsm.parse(r["text"]), r["gold"]) for r in rows)
        null = sum(c * sum(v for y, v in golds.items() if gsm.correct(x, y)) for x, c in preds.items()) / n**2
        result[arm] = {**wilson(k, n), "randomly_reassigned_gold_expected_points": 100*null,
                       "token_limit_outputs": sum(r["stop_type"] == "limit" for r in rows), "sha256": sha(path)}
    return result


def contamination():
    train = np.memmap(DATA / "train.u32", dtype=np.uint32, mode="r")
    seen = np.unique(ngram_hashes(train))
    result = {}
    path = OUT / "contamination_cases.jsonl"
    assert not path.exists(), "completed contamination audit already exists"
    tmp = path.with_suffix(".partial")
    with tmp.open("w") as f:
        for key, rows in read_cases().items():
            overlaps = []
            for case_id, ids in rows:
                hashes = ngram_hashes(ids) if len(ids) >= 13 else np.asarray([], dtype=np.uint64)
                if len(hashes):
                    pos = np.minimum(np.searchsorted(seen, hashes), len(seen)-1)
                    hits = int((seen[pos] == hashes).sum())
                else:
                    hits = 0
                fraction = hits / len(hashes) if len(hashes) else 0
                overlaps.append(fraction)
                f.write(json.dumps({"set": key, "id": case_id, "ngrams": len(hashes),
                                    "hits": hits, "fraction": fraction}) + "\n")
            result[key] = {"cases": len(rows), "cases_with_any_13gram_overlap": sum(v > 0 for v in overlaps),
                           "cases_with_majority_13gram_overlap": sum(v > .5 for v in overlaps),
                           "maximum_overlap_fraction": max(overlaps), "mean_overlap_fraction": float(np.mean(overlaps))}
    tmp.replace(path)
    return {"sets": result, "file": path.name, "sha256": sha(path),
            "rule": "same uint64 rolling 13-token gram as QAT08, against the entire exact 65,536,000-token mixed stream; every case including short cases; collisions can only add apparent overlap",
            "gate": "no case may have majority (>50%) overlap; any lower overlap is reported, not called zero contamination"}


def verify_design():
    design = json.loads((OUT / "design.json").read_text())
    for group in ("implementation_sha256", "input_sha256", "runtime_sha256", "comparator_output_sha256"):
        for name, digest in design[group].items():
            assert sha(ROOT / name) == digest, f"frozen design dependency changed: {name}"
    for name, digest in design["data_sha256"].items():
        assert sha(DATA / name) == digest, f"frozen design data changed: {name}"
    read_cases()
    return design


def freeze_design():
    if (OUT / "design.json").exists():
        raise SystemExit("design already frozen; use a new versioned amendment")
    data = {"prompts_record": json.loads((DATA / "prompts_record.json").read_text()),
            "prior_fineweb_record": json.loads((ROOT / "work/qat08/data/data_record.json").read_text()),
            "train_tokens": 65_536_000, "chat_tokens": 16_384_000, "fineweb_tokens": 49_152_000,
            "mix_ratio_tokens": .25,
            "seal": "exact generated response, unique-chat and mixed-stream hashes added by protocol stage before training",
            "packing": "Qwen template with thinking disabled; strip answer edges as template does; concatenate complete naturally terminated chats plus im_end; repeat corpus in fixed prompt order without padding",
            "fixed_filters": "discard non-eos/empty/special-token/thinking responses, any response with a benchmark-content 13-word overlap, and chats over 1024 tokens; require >=3000 retained chats and >=1M unique tokens; if gate fails, stop without training",
            "order": "three contiguous FineWeb 1024-token sequences then one chat sequence; FineWeb starts at the same QAT08 offset; monitor unchanged"}
    prior = json.loads((PRIOR / "protocol.json").read_text())
    cases = json.loads((OUT / "cases_record.json").read_text())
    assert data["train_tokens"] == prior["training"]["train_tokens"] == 65_536_000
    assert data["mix_ratio_tokens"] == .25
    # Record inputs/helpers in addition to direct entrypoints, because they have changed outside sessions.
    implementations = ["prepare_qat08_chat.py", "freeze_qat08_chat.py", "qat_08b_chat.py", "score_qat08_chat.py",
                       "run_qat08_chat.sh", "run_qat08_chat_prepare.sh", "qat_08b.py", "prepare_qat08_data.py", "freeze_qat08.py", "freeze_17b.py",
                       "score_qat08.py", "score_17b.py", "score_17b.cpp", "gsm8k_17b.py", "make_gptq_pilot.py",
                       "codec_probe.py", "phase0.py", "../inspect_gguf.py", "work/qwen3_17b/score_17b"]
    input_paths = ["results/qat08/protocol.json", "results/qat08/results.json", "results/qat08/curve.json",
                   "work/qwen35_08b/base-f32.gguf", "work/qwen35_08b/folded-pq2.gguf",
                   "work/qwen35_08b/folded-pq2-gptq.gguf", "work/qwen35_08b/folded/hadamard_packing.json",
                   "work/qat08/qat_free-final.gguf", "work/qat08/qat_42-final.gguf", "results/qat08_chat/cases_record.json",
                   "../../../bin/cuda/llama-server", "work/qat08/data/train.u32", "work/qat08/data/monitor.u32",
                   "work/qat08/data/data_record.json"]
    input_paths += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "work/qwen35_08b/base").iterdir())
                    if p.is_file()]
    implementation_sha = {n: sha(ROOT / n) for n in implementations}
    input_sha = {n: sha(ROOT / n) for n in input_paths}
    assert input_sha["work/qat08/data/train.u32"] == data["prior_fineweb_record"]["train_sha256"]
    assert input_sha["work/qat08/data/monitor.u32"] == data["prior_fineweb_record"]["monitor_sha256"]
    data_files = ["prompts.jsonl", "prompts_record.json", "prompt_audit.json", "pilot_responses.jsonl",
                  "pilot_timing.json", "source/train.jsonl", "source/README.md"]
    data_sha = {n: sha(DATA / n) for n in data_files}
    comparator_outputs = [ROOT / "work/qat08/scores" / f"{a}-{k}.jsonl"
                          for a in ("fp", "gptq", "qat_free", "qat_42") for k in ("mmlu", "retrieval")]
    comparator_outputs += [PRIOR / "gsm8k" / f"{a}.jsonl" for a in ("fp", "gptq", "qat_free", "qat_42")]
    comparator_output_sha = {str(path.relative_to(ROOT)): sha(path) for path in comparator_outputs}
    previous_result = json.loads((PRIOR / "results.json").read_text())
    for arm in ("fp", "gptq", "qat_free", "qat_42"):
        assert input_sha[previous_result["models"][arm]["file"]] == previous_result["models"][arm]["sha256"]
    contamination_record = {"status": "mandatory before protocol.json and before training; after full data generation",
                            "ngram": 13, "gate": "no evaluation case may have majority (>50%) 13-token overlap with the exact training stream; every case audited and any smaller overlap reported",
                            "prompt_disjointness": data["prompts_record"]["benchmark_disjointness"]}
    contamination_record["additional_prompt_audit"] = json.loads((DATA / "prompt_audit.json").read_text())
    p = {
        "date": "2026-10-02", "role": "chat-aware data-only amendment of QAT08 on Qwen3.5-0.8B; one top86 arm",
        "amends": {"protocol_sha256": sha(PRIOR / "protocol.json"), "results_sha256": sha(PRIOR / "results.json")},
        **{k: prior[k] for k in ("source", "basis", "student", "teacher", "loss", "rules", "training", "book_scoring", "scoring_backend")},
        "data": data,
        "mix_justification": "25% of training tokens replaces plain prose with teacher chat transcripts: 16.384M chat tokens repeatedly exercise role delimiters, empty think block, answers and turn endings while preserving 49.152M FineWeb tokens (75%) for book ability. Pilot tests format hypothesis, not instruction-domain optimum; repetitions are disclosed.",
        "initialization": "same folded FP projection latents and fixed top86 ternary embedding as QAT08, NOT a continuation of its trained student",
        "only_training_change": "input stream: exact 3 FineWeb sequences followed by 1 chat sequence per microbatch; all-position KL, unchanged monitors, no response-only masking",
        "arms": {"fp": {"file": "work/qwen35_08b/base-f32.gguf"},
                 "gptq": {"file": "work/qwen35_08b/folded-pq2-gptq.gguf"},
                 "qat_free": {"file": "work/qat08/qat_free-final.gguf"},
                 "qat_42": {"file": "work/qat08/qat_42-final.gguf"},
                 "qat_chat_42": {"rule": "top86", "checkpoint": "final", "file": "work/qat08_chat/qat_chat_42-final.gguf"}},
        "splits": {"validation_books": list(VALIDATION), "heldout_books": [k for k in cases["titles"] if k not in VALIDATION]},
        **{k: cases[k] for k in ("book_entries", "book_cases", "mmlu", "gsm8k", "retrieval")},
        "book_title_checks": cases["titles"], "prior_freeze_inventories": cases["prior_freeze_inventories"],
        "validation_use": "descriptive checkpoint curve only; final checkpoint fixed; no selection or early stopping",
        "heldout_arms": {"books_mmlu_retrieval": ["fp", "gptq", "qat_free", "qat_42", "qat_chat_42"],
                         "gsm8k": ["fp", "gptq", "qat_free", "qat_42", "qat_chat_42"]},
        "comparator_reuse": "rescore all arms on fresh books; MMLU, GSM8K and retrieval reuse hash-checked QAT08 raw outputs because case IDs, token IDs, models and harness are identical; no pooled books across experiments",
        "contamination_13gram": contamination_record,
        "gsm8k_parsing_coincidence_baseline": coincidence(),
        "decision": {
            "mmlu_lower_bound_points": 30.0, "gsm8k_lower_bound_points": 10.0,
            "allowed_book_nll_cost": 0.10,
            "chat_recovered": "MMLU Wilson 95% lower bound >30% AND GSM8K Wilson 95% lower bound >10%, also >10% when every non-eos/limit/parse-error output counts as incorrect. Same primary letter/number parser as QAT08; stricter termination diagnostic blocks loop coincidences. Joint claim uses intersection of both gates.",
            "book_cost_pass": "upper two-sided 95% paired t bound (12 books) for qat_chat_42 minus QAT08 qat_42 <= +0.10 nat/token; compare to qat_free secondarily",
            "amendment_success": "chat_recovered AND book_cost_pass; otherwise report chat-only, text-cost-only, or failed recovery without moving thresholds",
            "uncertainty": "Wilson intervals for individual binary accuracies; QAT08 paired normal item intervals for task differences; paired t over books/registries for continuous differences; one run/seed, descriptive generalization",
            "retrieval": "same 200 raw-text registries and paired margins, reported as a retention diagnostic without a new acceptance gate",
            "claim_limit": "one small hybrid model, one mix and seed, repeated synthetic prompts; checks remove literal overlaps but not paraphrases or ancestor pretraining contamination; task harness differs from publisher; this does not identify PrismML's training recipe",
        },
        "teacher_generation": {"pilot": json.loads((DATA / "pilot_timing.json").read_text()),
                               "estimated_hours": 2.6, "limit_tokens": 768, "slots": 8,
                               "template": "pinned Qwen3.5 chat template, user prompt, add_generation_prompt=True, enable_thinking=False; empty think block",
                               "model": "the original FP F32 GGUF in the pinned runtime; distillation teacher unchanged at BF16"},
        "compute_budget": {"total_gpu_hours_ceiling": 12.0, "student_training_gpu_hours_ceiling": 6.5,
                           "teacher": "2.6 h estimate from the 64-prompt training-domain pilot, which took 58.5 seconds",
                           "evaluation_gpu_hours_estimate": 2.0,
                           "no_second_arm": "top86 matched or beat absmean in QAT08; single arm isolates data within local budget",
                           "execution": "systemd user units, response-by-response teacher resume, atomic trainer resume.pt, evaluation item resume; no extension or threshold change without a versioned amendment"},
        "implementation_sha256": implementation_sha, "input_sha256": input_sha, "data_sha256": data_sha,
        "environment": {"python": platform.python_version(),
                        "packages": {name: importlib.metadata.version(name) for name in
                                     ("torch", "transformers", "flash-linear-attention", "triton", "numpy", "scipy", "pyarrow")}},
        "comparator_output_sha256": comparator_output_sha,
        "runtime_sha256": {"../../../bin/cuda/" + p.name: sha(p) for p in sorted((ROOT.parents[2] / "bin/cuda").glob("*.so*"))
                           if p.is_file()},
        "approval": "training requires explicit user go-ahead AFTER review of this frozen design; final protocol extends this design only with actual data hashes/audit/timings, before training; training_approval.json binds both hashes",
        "pre_freeze_incidents": [
            "Title checks caught wrong remembered Gutenberg IDs: 668 was a dictionary and 427 was The Great War Syndicate. Downloads preserved as rejected artifacts; corrected to 6688 and 4274 before cases or scoring. A parallel title check caught 918 (Sketches of Young Gentlemen); Villette corrected to 9182 before download for the experiment.",
            "Prompt preparation first stopped because only 135 JSON-format prompts met the fixed 384-token length cap, below the draft 500 quota. Before any generation or evaluation, the JSON quota was set to 100 and the general-conversation quota to 6400, keeping 10,000 prompts total.",
            "A CPU transcript check found that this installed tokenizer returns an encoding object rather than an ID list for tokenize=True. The packing assertion now renders text then encodes it, matching the existing harness. The pilot transcripts were checked before the design freeze.",
        ],
    }
    write_new(OUT / "design.json", p)
    print("froze design", sha(OUT / "design.json"), flush=True)


def freeze_protocol():
    if (OUT / "protocol.json").exists():
        raise SystemExit("protocol already frozen; use a new versioned amendment")
    design = verify_design()
    data = json.loads((DATA / "data_record.json").read_text())
    assert data["train_tokens"] == design["training"]["train_tokens"]
    assert data["chat_tokens"] == design["data"]["chat_tokens"]
    assert data["fineweb_tokens"] == design["data"]["fineweb_tokens"]
    assert data["monitor_sha256"] == design["data"]["prior_fineweb_record"]["monitor_sha256"]
    assert data["prompts_record"] == design["data"]["prompts_record"]
    files = ["train.u32", "monitor.u32", "chat_unique.u32", "teacher.jsonl", "kept_chats.json",
             "data_record.json", "teacher_timing.json"]
    data_sha = {**design["data_sha256"], **{n: sha(DATA / n) for n in files}}
    audit = contamination()
    assert all(v["cases_with_majority_13gram_overlap"] == 0 for v in audit["sets"].values()), audit
    teacher_timing = json.loads((DATA / "teacher_timing.json").read_text())
    elapsed_h = (teacher_timing["seconds"] + design["teacher_generation"]["pilot"]["seconds"]) / 3600
    assert elapsed_h + 6.5 + 2.0 <= design["compute_budget"]["total_gpu_hours_ceiling"], "budget projection exceeded; stop before training"
    p = {**design, "design_sha256": sha(OUT / "design.json"), "data": data, "data_sha256": data_sha,
         "contamination_13gram": audit, "teacher_generation": {**design["teacher_generation"], "actual": teacher_timing}}
    write_new(OUT / "protocol.json", p)
    print("froze protocol", sha(OUT / "protocol.json"), flush=True)
    print(json.dumps({"data": {k: data[k] for k in ("train_tokens", "chat_tokens", "unique_chat_tokens", "kept_chats", "chat_corpus_passes")},
                      "contamination": audit["sets"], "decision": p["decision"]}, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["design", "protocol", "verify-design"])
    args = ap.parse_args()
    {"design": freeze_design, "protocol": freeze_protocol, "verify-design": verify_design}[args.stage]()

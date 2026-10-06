"""Freeze untruncated LongBench-E QA cases, stratified by published length bins."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from chat_tokens import MODEL, chat_template, count_tokens, render_chat


BASE = Path(__file__).resolve().parent / "benchmark"
SOURCE = BASE / "source"
MAX_PROMPT_TOKENS = 16384 - 2000
BIN_TARGETS = {"0-4k": 7, "4-8k": 7, "8k+": 6}
SEED = "bonsai2-longbench-20260929"


def length_bin(words: int) -> str:
    return "0-4k" if words < 4000 else "4-8k" if words < 8000 else "8k+"


def source_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    template = chat_template()
    prompt_config = json.loads((SOURCE / "dataset2prompt.json").read_text())
    selected = []
    summary = {"selection_seed": SEED, "max_prompt_tokens": MAX_PROMPT_TOKENS,
               "context_tokens": 16384, "output_and_template_reserve": 2000,
               "source_sha256": {}, "tasks": {}}
    for task in ("qasper_e", "2wikimqa_e"):
        path = SOURCE / f"{task}.jsonl"
        summary["source_sha256"][task] = source_hash(path)
        rows = [json.loads(line) for line in path.open()]
        buckets = {key: [] for key in BIN_TARGETS}
        counts = Counter()
        for row in rows:
            bucket = length_bin(row["length"])
            counts[f"total_{bucket}"] += 1
            prompt = prompt_config[task.removesuffix("_e")].format(**row)
            tokens = count_tokens(render_chat(template, prompt))
            if tokens > MAX_PROMPT_TOKENS:
                counts[f"excluded_over_context_{bucket}"] += 1
                continue
            counts[f"eligible_{bucket}"] += 1
            buckets[bucket].append({"id": row["_id"], "task": task, "length_bin": bucket,
                                    "source_length_words": row["length"],
                                    "prompt_tokens": tokens, "prompt": prompt,
                                    "answers": row["answers"]})
        for bucket, target in BIN_TARGETS.items():
            ranked = sorted(buckets[bucket], key=lambda row: hashlib.sha256(
                f"{SEED}:{task}:{row['id']}".encode()).hexdigest())
            chosen = ranked[:target]
            counts[f"selected_{bucket}"] = len(chosen)
            selected.extend(chosen)
        summary["tasks"][task] = dict(sorted(counts.items()))
    selected_path = BASE / "selected.jsonl"
    if selected_path.exists():
        raise FileExistsError("LongBench selection already frozen")
    selected_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
    summary["source_sha256"]["prompt_config"] = source_hash(SOURCE / "dataset2prompt.json")
    summary["selected_sha256"] = source_hash(selected_path)
    summary["selected_ids"] = {task: [row["id"] for row in selected if row["task"] == task]
                               for task in ("qasper_e", "2wikimqa_e")}
    (BASE / "selection_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["tasks"], indent=2))


if __name__ == "__main__":
    main()

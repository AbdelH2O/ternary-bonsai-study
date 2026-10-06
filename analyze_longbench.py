"""Summarize paired LongBench-E subset F1 without calling it a full benchmark score."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


BASE = Path(__file__).resolve().parent


def load(arm):
    path = BASE / "runs" / f"longbench-{arm}.jsonl"
    return {row["id"]: row for row in (json.loads(line) for line in path.open())}


def interval(values, replicates=10000):
    rng = random.Random(20260929)
    samples = sorted(statistics.mean(values[rng.randrange(len(values))] for _ in values)
                     for _ in range(replicates))
    return [samples[249], samples[9749]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-arm", required=True)
    parser.add_argument("--mlp-arm", required=True)
    args = parser.parse_args()
    arms = {name: load(name) for name in ("baseline", args.state_arm, args.mlp_arm)}
    selected = [json.loads(line) for line in (BASE / "benchmark/selected.jsonl").open()]
    items = [row for row in selected if all(row["id"] in arms[name] and
             arms[name][row["id"]].get("score") is not None for name in arms)]
    if not items:
        raise ValueError("No complete scored LongBench-E items")
    ids = [row["id"] for row in items]
    summary = {"complete_items": len(ids), "expected_items": len(selected),
               "subset_mean_f1": {name: statistics.mean(arms[name][id]["score"] for id in ids)
                                  for name in arms},
               "task_mean_f1": {name: {task: statistics.mean(arms[name][row["id"]]["score"]
                                                          for row in items if row["task"] == task)
                                      for task in ("qasper_e", "2wikimqa_e")} for name in arms},
               "state_minus_mlp_mean_f1": statistics.mean(
                   arms[args.state_arm][id]["score"] - arms[args.mlp_arm][id]["score"] for id in ids),
               "paired_bootstrap_95pct": interval([
                   arms[args.state_arm][id]["score"] - arms[args.mlp_arm][id]["score"] for id in ids]),
               "bootstrap_seed": 20260929}
    out = BASE / "runs" / f"longbench-{args.state_arm}-vs-{args.mlp_arm}.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

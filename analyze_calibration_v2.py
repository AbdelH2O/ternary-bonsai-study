"""Summarize paired correct-answer log probabilities on calibration v2."""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path


BASE = Path(__file__).resolve().parent / "runs"


def load(arm: str):
    path = BASE / f"calibration_v2-logprobs-{arm}.jsonl"
    return {row["id"]: row for row in (json.loads(line) for line in path.open())}


def interval(deltas):
    rng = random.Random(20260929)
    samples = sorted(statistics.mean(deltas[rng.randrange(len(deltas))] for _ in deltas)
                     for _ in range(10000))
    return [samples[249], samples[9749]]


def main():
    baseline = load("baseline")
    arms = sorted(path.name.removeprefix("calibration_v2-logprobs-").removesuffix(".jsonl")
                  for path in BASE.glob("calibration_v2-logprobs-*.jsonl")
                  if not path.name.endswith("baseline.jsonl"))
    result = {"baseline_accuracy": statistics.mean(row["score"] for row in baseline.values()),
              "baseline_valid_ids": sorted(key for key, row in baseline.items() if row["logprob_valid"]),
              "arms": {}}
    for arm in arms:
        rows = load(arm)
        if set(rows) != set(baseline):
            continue  # The sequential runner may still be writing this arm.
        valid = [key for key in result["baseline_valid_ids"] if rows[key]["logprob_valid"]]
        deltas = [baseline[key]["answer_logprob"] - rows[key]["answer_logprob"] for key in valid]
        result["arms"][arm] = {
            "accuracy": statistics.mean(row["score"] for row in rows.values()),
            "baseline_correct_treatment_wrong": sorted(key for key in result["baseline_valid_ids"]
                                                       if not rows[key]["logprob_valid"]),
            "baseline_wrong_treatment_correct": sorted(key for key, row in baseline.items()
                                                       if not row["logprob_valid"] and rows[key]["logprob_valid"]),
            "paired_valid_items": len(valid),
            "mean_excess_answer_nll": statistics.mean(deltas) if deltas else None,
            "paired_bootstrap_95pct": interval(deltas) if deltas else None,
        }
    out = BASE / "calibration_v2_summary.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    for arm, data in result["arms"].items():
        print(arm, "accuracy", data["accuracy"], "excess_nll", data["mean_excess_answer_nll"],
              "new_failures", len(data["baseline_correct_treatment_wrong"]))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

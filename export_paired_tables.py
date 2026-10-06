"""Export item-level paired contrasts from the frozen exploratory pilot."""

from __future__ import annotations

import csv
import json
from pathlib import Path


BASE = Path(__file__).resolve().parent
RUNS = BASE / "runs"
ARMS = ("baseline", "state-d02", "mlp-d02", "state-d03", "mlp-d03")


def load(path: Path):
    rows = [json.loads(line) for line in path.open()]
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"Duplicate item ID in {path}")
    return by_id


def write(path: Path, fields: list[str], rows: list[dict]):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def primary():
    cases = load(BASE / "prompts/primary.jsonl")
    results = {arm: load(RUNS / f"primary-logprobs-{arm}.jsonl") for arm in ARMS}
    if any(set(rows) != set(cases) for rows in results.values()):
        raise ValueError("Primary item coverage differs from frozen prompts")
    pairs = sorted({row["pair_id"] for row in cases.values()})
    rows = []
    for delta in ("02", "03"):
        state, mlp = f"state-d{delta}", f"mlp-d{delta}"
        for pair in pairs:
            short, long = f"{pair}-short", f"{pair}-long"
            if not all(results[arm][item]["logprob_valid"] for arm in ARMS for item in (short, long)):
                raise ValueError(f"Answer logprob invalid for {pair}")
            def additional_long_loss(arm: str, field: str):
                return ((results["baseline"][long][field] - results[arm][long][field]) -
                        (results["baseline"][short][field] - results[arm][short][field]))

            rows.append({
                "strength_pct": int(delta), "pair_id": pair,
                "target_position": cases[short]["placement"],
                "short_prompt_tokens": results["baseline"][short]["prompt_tokens"],
                "long_prompt_tokens": results["baseline"][long]["prompt_tokens"],
                "baseline_short_accuracy": results["baseline"][short]["score"],
                "baseline_long_accuracy": results["baseline"][long]["score"],
                "state_short_accuracy": results[state][short]["score"],
                "state_long_accuracy": results[state][long]["score"],
                "mlp_short_accuracy": results[mlp][short]["score"],
                "mlp_long_accuracy": results[mlp][long]["score"],
                "state_additional_long_accuracy_loss": additional_long_loss(state, "score"),
                "mlp_additional_long_accuracy_loss": additional_long_loss(mlp, "score"),
                "state_minus_mlp_additional_long_accuracy_loss":
                    additional_long_loss(state, "score") - additional_long_loss(mlp, "score"),
                "state_additional_long_answer_nll": additional_long_loss(state, "answer_logprob"),
                "mlp_additional_long_answer_nll": additional_long_loss(mlp, "answer_logprob"),
                "state_minus_mlp_additional_long_answer_nll":
                    additional_long_loss(state, "answer_logprob") - additional_long_loss(mlp, "answer_logprob"),
            })
    write(RUNS / "paired_primary.csv", list(rows[0]), rows)


def longbench():
    cases = load(BASE / "benchmark/selected.jsonl")
    results = {arm: load(RUNS / f"longbench-{arm}.jsonl") for arm in ARMS}
    if any(set(rows) != set(cases) for rows in results.values()):
        raise ValueError("LongBench item coverage differs from frozen selection")
    rows = []
    for delta in ("02", "03"):
        state, mlp = f"state-d{delta}", f"mlp-d{delta}"
        for item, case in cases.items():
            scores = {arm: results[arm][item]["score"] for arm in ("baseline", state, mlp)}
            if any(score is None for score in scores.values()):
                raise ValueError(f"Unscored LongBench item: {item}")
            rows.append({
                "strength_pct": int(delta), "id": item, "task": case["task"],
                "length_bin": case["length_bin"], "prompt_tokens": results["baseline"][item]["prompt_tokens"],
                "baseline_f1": scores["baseline"], "state_f1": scores[state], "mlp_f1": scores[mlp],
                "state_minus_baseline_f1": scores[state] - scores["baseline"],
                "mlp_minus_baseline_f1": scores[mlp] - scores["baseline"],
                "state_minus_mlp_f1": scores[state] - scores[mlp],
            })
    write(RUNS / "paired_longbench.csv", list(rows[0]), rows)


if __name__ == "__main__":
    primary()
    longbench()

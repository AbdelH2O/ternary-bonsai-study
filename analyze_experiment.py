"""Compute the paired short-to-long contrast and question bootstrap interval."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


BASE = Path(__file__).resolve().parent


def load(arm: str):
    path = BASE / "runs" / f"primary-logprobs-{arm}.jsonl"
    if not path.exists():
        path = BASE / "runs" / f"primary-{arm}.jsonl"
    return {row["id"]: row for row in (json.loads(line) for line in path.open())}


def mean(values):
    return sum(values) / len(values)


def bootstrap_interval(values, replicates):
    rng = random.Random(20260929)
    boot = sorted(mean([values[rng.randrange(len(values))] for _ in values])
                  for _ in range(replicates))
    return [boot[int(.025 * (len(boot) - 1))], boot[int(.975 * (len(boot) - 1))]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-arm", required=True)
    parser.add_argument("--mlp-arm", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    arms = {arm: load(arm) for arm in ("baseline", args.state_arm, args.mlp_arm)}
    pair_ids = sorted({id.removesuffix("-short").removesuffix("-long") for id in arms["baseline"]})
    complete = []
    for pair_id in pair_ids:
        if all((row := arms[arm].get(f"{pair_id}-{length}")) is not None and row.get("score") is not None
               for arm in arms for length in ("short", "long")):
            complete.append(pair_id)
    if not complete:
        raise ValueError("No complete scored question pairs")

    def excess(arm, question):
        loss_short = arms["baseline"][f"{question}-short"]["score"] - arms[arm][f"{question}-short"]["score"]
        loss_long = arms["baseline"][f"{question}-long"]["score"] - arms[arm][f"{question}-long"]["score"]
        return loss_long - loss_short

    contrasts = [excess(args.state_arm, q) - excess(args.mlp_arm, q) for q in complete]
    summary = {"complete_pairs": len(complete), "missing_pairs": sorted(set(pair_ids) - set(complete)),
               "state_arm": args.state_arm, "mlp_arm": args.mlp_arm,
               "accuracy": {arm: {length: mean([arms[arm][f"{q}-{length}"]["score"] for q in complete])
                                  for length in ("short", "long")} for arm in arms},
               "excess_long_loss": {arm: mean([excess(arm, q) for q in complete])
                                    for arm in (args.state_arm, args.mlp_arm)},
               "state_minus_mlp_excess_long_loss": mean(contrasts),
               "paired_bootstrap_95pct": bootstrap_interval(contrasts, args.bootstrap),
               "bootstrap_seed": 20260929, "bootstrap_replicates": args.bootstrap}
    lp_pairs = [q for q in complete if all(arms[arm][f"{q}-{length}"].get("logprob_valid")
                                        for arm in arms for length in ("short", "long"))]
    if lp_pairs:
        def lp_excess(arm, question):
            short = (arms["baseline"][f"{question}-short"]["answer_logprob"] -
                     arms[arm][f"{question}-short"]["answer_logprob"])
            long = (arms["baseline"][f"{question}-long"]["answer_logprob"] -
                    arms[arm][f"{question}-long"]["answer_logprob"])
            return long - short

        lp_contrast = [lp_excess(args.state_arm, q) - lp_excess(args.mlp_arm, q)
                       for q in lp_pairs]
        summary["answer_logprob_secondary"] = {
            "complete_pairs": len(lp_pairs),
            "excess_long_nll": {arm: mean([lp_excess(arm, q) for q in lp_pairs])
                                for arm in (args.state_arm, args.mlp_arm)},
            "state_minus_mlp_excess_long_nll": mean(lp_contrast),
            "paired_bootstrap_95pct": bootstrap_interval(lp_contrast, args.bootstrap),
        }
    out = BASE / "runs" / f"contrast-{args.state_arm}-vs-{args.mlp_arm}.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

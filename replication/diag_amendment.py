"""Post-results amendment to the read-out diagnostics decision (written 2026-10-04, after results.json was seen).

The frozen FP binding floor used contextual-calibrated accuracy. Mean-log-prob calibration penalizes a near-perfect
model: FP scored 99.25% uncalibrated and 94.0% calibrated, because the per-letter offsets (A -3.50 ... D -4.70 nats)
flip 22 gold-A items whose margin is under 1.2 nats, although FP's predictions are balanced (99/101/101/99). The
amendment applies the same 95% floor to FP's uncalibrated binding accuracy and changes nothing else in the rule.
decision.json (harness_invalid) stays unmodified; Plan B reads this file only after checking its hashes.

    python diag_amendment.py "<user's exact words>"   # exclusive results/diag_readout/decision_amendment.json
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

from diag_cases import OUT
from diag_metrics import branch
from score_17b import sha

REASON = ("the frozen FP binding floor used calibrated accuracy; mean-log-prob calibration subtracts per-letter offsets "
          "that measure how strongly a confident model rejects wrong letters, not a letter prior, and so flips "
          "low-margin items of a near-perfect model. The floor now applies to FP's uncalibrated binding accuracy. "
          "Chosen after results were seen; the branch is the same under any FP floor FP passes")


def amend(results: dict, decision: dict) -> dict:
    """The frozen branch rule with only the FP binding floor moved to uncalibrated accuracy."""
    rule = decision["rule"]
    raw = results["arms"]["fp"]["binding"]["raw_accuracy"]
    if raw < rule["fp_binding_min"]:
        return {"branch": "harness_invalid", "arms_required": [], "arms_optional": [], "stop": True,
                "reasons": [f"FP binding uncalibrated accuracy {raw:.2f}% < {rule['fp_binding_min']}% (amended floor)"]}
    out = branch(results, {**rule, "fp_binding_min": 0.0})
    out["reasons"] = [f"FP binding uncalibrated accuracy {raw:.2f}% >= {rule['fp_binding_min']}% (amended floor)",
                      *out.get("reasons", [])]
    return out


def write(out: Path, instruction: str) -> dict:
    decision_path, results_path, plan_path = out / "decision.json", out / "results.json", out / "plan.json"
    decision, results = json.loads(decision_path.read_text()), json.loads(results_path.read_text())
    assert decision["results_sha256"] == sha(results_path), "results.json changed since the decision"
    assert decision["plan_sha256"] == sha(plan_path), "plan.json changed since the decision"
    fp = results["arms"]["fp"]["binding"]
    record = {"amends": "results/diag_readout/decision.json", "amends_decision_sha256": sha(decision_path),
              "results_sha256": sha(results_path), "plan_sha256": sha(plan_path), "written_after_results": True,
              "original_branch": decision["branch"], "reason": REASON,
              "changed_rule": {"fp_binding_floor_metric": {"from": "calibrated_accuracy", "to": "raw_accuracy"},
                               "fp_binding_min": decision["rule"]["fp_binding_min"]},
              "fp_binding": {k: fp[k] for k in ("raw_accuracy", "calibrated_accuracy", "pairwise", "pairwise_lo")},
              "amended": amend(results, decision), "amendment_script_sha256": sha(Path(__file__)),
              "user_instruction": instruction,
              "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with (out / "decision_amendment.json").open("x") as f:
        json.dump(record, f, indent=2)
    return record


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("instruction", help="the user's exact words approving the amendment")
    rec = write(OUT, ap.parse_args().instruction)
    print(json.dumps(rec["amended"], indent=2))

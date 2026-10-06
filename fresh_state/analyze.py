"""Analyze the predeclared document-level fresh-state contrast and controls."""

import hashlib
import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(sorted_values, q):
    at = q * (len(sorted_values) - 1)
    lo = int(at)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (at - lo) * (sorted_values[hi] - sorted_values[lo])


def bootstrap_ci(document_values, seed, reps, coverage):
    rng = random.Random(seed)
    n = len(document_values)
    draws = sorted(statistics.mean(document_values[rng.randrange(n)] for _ in range(n))
                   for _ in range(reps))
    tail = (1 - coverage) / 2
    return [percentile(draws, tail), percentile(draws, 1 - tail)]


def verify_gate(name, freeze_hash):
    path = HERE / name
    row = json.loads(path.read_text())
    if not row["passed"] or row["freeze_sha256"] != freeze_hash:
        raise ValueError(f"Required gate failed: {name}")
    return row


def load_reference(frozen):
    result = {}
    for arm in frozen["models"]:
        rows = [json.loads(x) for x in (HERE / f"reference-{arm}.jsonl").read_text().splitlines()]
        if [r["case_id"] for r in rows] != [c["case_id"] for c in frozen["cases"]]:
            raise ValueError(f"Reference identities changed: {arm}")
        result[arm] = {r["case_id"]: r["nll"] for r in rows}
    return result


def main():
    frozen = json.loads((HERE / "FRESH_STATE_FREEZE.json").read_text())
    freeze_hash = sha(HERE / "FRESH_STATE_FREEZE.json")
    reference_gate = verify_gate("reference_gate.json", freeze_hash)
    capture_gate = verify_gate("capture_gate.json", freeze_hash)
    positive_gate = verify_gate("positive_gate.json", freeze_hash)
    treatment_gate = verify_gate("treatment_gate.json", freeze_hash)
    if (len(capture_gate["captures"]), len(positive_gate["comparisons"]),
            len(treatment_gate["comparisons"])) != (72, 24, 72):
        raise ValueError("Gate counts are incomplete")
    references = load_reference(frozen)
    if reference_gate["raw_sha256"] != {arm: sha(HERE / f"reference-{arm}.jsonl")
                                           for arm in frozen["models"]}:
        raise ValueError("Reference raw changed")
    for group in (positive_gate["comparisons"], treatment_gate["comparisons"]):
        for item in group:
            raw = ROOT / item["raw_path"]
            if sha(raw) != item["raw_sha256"]:
                raise ValueError(f"Comparison raw changed: {raw}")
            row = json.loads(raw.read_text())
            reference = references[item["destination"]][item["case_id"]]
            if max(abs(x-y) for x, y in zip(row["direct_nll"], reference)) > 0.0001:
                raise ValueError(f"Direct reference mismatch: {item['case_id']}")
            for actual, recorded in zip(row["conditions"], item["metrics"]):
                if actual["label"] != recorded["label"] or sha(Path(actual["state_path"])) != recorded["source_state_sha256"]:
                    raise ValueError(f"Source state changed: {item['case_id']}")
                if max(abs(x-y) for x, y in zip(row["direct_nll"], actual["restored_nll"])) > 0.0001:
                    raise ValueError(f"A-B-A mismatch: {item['case_id']}")
                if actual["imported_nll"][0] != row["direct_nll"][0]:
                    raise ValueError(f"Target 0 changed: {item['case_id']}")
                delta = statistics.mean(x-y for x, y in zip(actual["imported_nll"][1:], row["direct_nll"][1:]))
                if abs(delta - recorded["mean_excess_nll"]) > 1e-10:
                    raise ValueError(f"Recorded state effect changed: {item['case_id']}")
    by_case = {(r["case_id"], r["destination"]): {m["label"]: m for m in r["metrics"]}
               for r in treatment_gate["comparisons"]}
    spans = []
    for doc in frozen["documents"]:
        for p in frozen["positions"]:
            case_1 = f"{doc['id']}-p{p}-h1024"
            case_12 = f"{doc['id']}-p{p}-h12288"
            a1 = by_case[case_1, "original"]["alpha_recurrent"]["mean_excess_nll"]
            a12 = by_case[case_12, "original"]["alpha_recurrent"]["mean_excess_nll"]
            m1 = by_case[case_1, "original"]["mlp_recurrent"]["mean_excess_nll"]
            m12 = by_case[case_12, "original"]["mlp_recurrent"]["mean_excess_nll"]
            original_gain = statistics.mean(references["original"][case_1][1:]) - statistics.mean(references["original"][case_12][1:])
            spans.append({"document_id": doc["id"], "position": p,
                          "alpha_recurrent_excess_1k": a1,
                          "alpha_recurrent_excess_12k": a12,
                          "mlp_recurrent_excess_1k": m1,
                          "mlp_recurrent_excess_12k": m12,
                          "alpha_long_minus_short": a12 - a1,
                          "mlp_long_minus_short": m12 - m1,
                          "primary_interaction": (a12 - a1) - (m12 - m1),
                          "short_state_dose_alpha_minus_mlp": a1 - m1,
                          "original_1k_minus_12k_gain": original_gain})
    fields = ("primary_interaction", "short_state_dose_alpha_minus_mlp",
              "alpha_long_minus_short", "mlp_long_minus_short", "original_1k_minus_12k_gain")
    documents = []
    for doc in frozen["documents"]:
        selected = [s for s in spans if s["document_id"] == doc["id"]]
        if len(selected) != 2:
            raise ValueError(f"Wrong span count: {doc['id']}")
        documents.append({"document_id": doc["id"], **{field: statistics.mean(x[field] for x in selected)
                                                       for field in fields}})
    seed = frozen["gates"]["bootstrap_seed"]
    reps = frozen["gates"]["bootstrap_replicates"]
    primary = [d["primary_interaction"] for d in documents]
    dose = [d["short_state_dose_alpha_minus_mlp"] for d in documents]
    primary_ci = bootstrap_ci(primary, seed, reps, 0.95)
    dose_ci = bootstrap_ci(dose, seed + 1, reps, 0.90)
    point = statistics.mean(primary)
    dose_equivalent = dose_ci[0] > -frozen["gates"]["short_state_dose_equivalence_band"] and dose_ci[1] < frozen["gates"]["short_state_dose_equivalence_band"]
    support = (point >= frozen["gates"]["primary_smallest_meaningful_nats_per_token"] and
               primary_ci[0] > 0 and sum(x > 0 for x in primary) >= 5)
    exclude = primary_ci[1] < frozen["gates"]["primary_smallest_meaningful_nats_per_token"]
    decision = "support" if support else "exclude_0p005" if exclude else "unresolved"
    output = {"freeze_sha256": freeze_hash,
              "gate_sha256": {name: sha(HERE / name) for name in (
                  "reference_gate.json", "capture_gate.json", "positive_gate.json", "treatment_gate.json")},
              "independent_document_count": len(documents), "span_count": len(spans),
              "target_tokens_per_case": 96, "state_affected_tokens_per_case": 95,
              "primary_mean": point, "primary_document_bootstrap_95ci": primary_ci,
              "primary_positive_document_count": sum(x > 0 for x in primary),
              "short_state_dose_mean": statistics.mean(dose),
              "short_state_dose_document_bootstrap_90ci": dose_ci,
              "short_state_dose_equivalent": dose_equivalent,
              "primary_decision": decision,
              "alpha_specific_comparable_dose_support": support and dose_equivalent,
              "secondary_document_means": {field: statistics.mean(d[field] for d in documents)
                                           for field in fields[2:]},
              "positive_control_min_mean_absolute_delta": min(
                  r["metrics"][0]["mean_absolute_nll_delta"] for r in positive_gate["comparisons"]),
              "documents": documents, "spans": spans,
              "interpretation": "Cross-checkpoint state replacement can be off-manifold; permanent weight changes, F32 runtime state; six document clusters are the uncertainty units."}
    (HERE / "analysis_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(f"Primary {point:+.6f} nat/token, 95% document CI [{primary_ci[0]:+.6f}, {primary_ci[1]:+.6f}]")
    print(f"Short-dose {statistics.mean(dose):+.6f}, 90% CI [{dose_ci[0]:+.6f}, {dose_ci[1]:+.6f}]; equivalent={dose_equivalent}")
    print(f"Decision: {decision}; alpha-specific comparable-dose support={support and dose_equivalent}")


if __name__ == "__main__":
    main()

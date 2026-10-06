"""Audit held-out scorer identity and summarize the frozen alpha-minus-MLP contrast."""

import hashlib
import json
import random
import statistics
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SCORER = ROOT / "analysis/bonsai2/scorer"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def id_sha(ids):
    return hashlib.sha256(struct.pack("<" + "i" * len(ids), *ids)).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def mean(values):
    return statistics.mean(values)


def percentile_interval(values):
    values.sort()
    return [values[int(0.025 * len(values))], values[int(0.975 * len(values))]]


def main():
    frozen = json.loads((HERE / "HELDOUT_FREEZE.json").read_text())
    manifest_path = SCORER / "frozen_manifest.json"
    if sha(manifest_path) != frozen["scorer_manifest_sha256"]:
        raise ValueError("Scorer manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if sha(SCORER / "baseline_results.jsonl") != frozen["baseline_results_sha256"]:
        raise ValueError("Baseline artifact changed")
    case_meta = {c["case_id"]: c for c in manifest["cases"]}
    doc_meta = {d["id"]: d for d in manifest["documents"]}
    for doc in doc_meta.values():
        if sha(ROOT / doc["source_path"]) != doc["source_sha256"] or sha(ROOT / doc["ids_path"]) != doc["ids_file_sha256"]:
            raise ValueError(f"Held-out document changed: {doc['id']}")
        ids = [int(x) for x in (ROOT / doc["ids_path"]).read_text().splitlines()]
        for case in (c for c in manifest["cases"] if c["document_id"] == doc["id"]):
            p, h, m = case["target_position"], case["history_length"], case["target_length"]
            if id_sha(ids[p - h:p]) != case["history_ids_sha256"] or id_sha(ids[p:p + m]) != case["target_ids_sha256"]:
                raise ValueError(f"Token slice changed: {case['case_id']}")
    baseline = {r["case_id"]: r for r in read_rows(SCORER / "baseline_results.jsonl")}
    rerun = {r["case_id"]: r for r in read_rows(HERE / "baseline_rerun.jsonl")}
    if set(baseline) != set(rerun) or len(rerun) != 12:
        raise ValueError("Baseline rerun incomplete")
    rerun_floor = max(abs(a - b) for case_id, raw in rerun.items()
                      for a, b in zip(raw["nll"], baseline[case_id]["gold_nll_nats"]))
    if rerun_floor > 0.0001:
        raise ValueError("Baseline numerical floor changed")
    results = []
    means = {}
    for variant in frozen["variants"]:
        name = variant["name"]
        if sha(ROOT / variant["model_path"]) != variant["model_sha256"]:
            raise ValueError(f"Variant model changed: {name}")
        if sha(ROOT / variant["patch_manifest_path"]) != variant["patch_manifest_sha256"]:
            raise ValueError(f"Patch changed: {name}")
        rows = read_rows(HERE / f"{name}.jsonl")
        if len(rows) != 12 or [r["case_id"] for r in rows] != frozen["case_ids"]:
            raise ValueError(f"Case mismatch: {name}")
        for raw in rows:
            case = case_meta[raw["case_id"]]
            doc = doc_meta[case["document_id"]]
            if (raw["p"], raw["h"], raw["m"], raw["n_batch"], raw["n_ubatch"]) != (
                    case["target_position"], case["history_length"], 96, 512, 512):
                raise ValueError(f"Profile mismatch: {name} {raw['case_id']}")
            if len(raw["nll"]) != 96:
                raise ValueError(f"Scored ID count mismatch: {name} {raw['case_id']}")
            baseline_nll = baseline[raw["case_id"]]["gold_nll_nats"]
            excess = mean(a - b for a, b in zip(raw["nll"], baseline_nll))
            key = (variant["arm"], variant["seed"], doc["id"], raw["h"])
            means[key] = excess
            results.append({"case_id": raw["case_id"], "arm": variant["arm"],
                "seed": variant["seed"], "document_id": doc["id"],
                "source_sha256": doc["source_sha256"], "all_ids_sha256": doc["all_ids_sha256"],
                "target_position": raw["p"], "history_length": raw["h"],
                "target_length": raw["m"], "history_ids_sha256": case["history_ids_sha256"],
                "target_ids_sha256": case["target_ids_sha256"],
                "model_sha256": variant["model_sha256"],
                "patch_manifest_sha256": variant["patch_manifest_sha256"],
                "scorer_source_sha256": frozen["scorer_source_sha256"],
                "gold_nll_nats": raw["nll"], "mean_gold_nll_nats": mean(raw["nll"]),
                "excess_mean_nll_nats": excess, "seconds": raw["seconds"],
                "backend": frozen["profile"], "error": None})
    (HERE / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    docs = frozen["documents"]
    seeds = sorted({v["seed"] for v in frozen["variants"]})
    by_arm_history = {arm: {str(h): mean(means[arm, seed, doc, h]
                                        for seed in seeds for doc in docs)
                             for h in (1024, 4096, 12288)}
                      for arm in ("alpha", "mlp")}
    short_a = by_arm_history["alpha"]["1024"]
    short_m = by_arm_history["mlp"]["1024"]
    short_match = (0.002 <= short_a <= 0.020 and 0.002 <= short_m <= 0.020
                   and abs(short_a - short_m) <= 0.003)
    paired = []
    for seed in seeds:
        for doc in docs:
            a1 = means["alpha", seed, doc, 1024]
            m1 = means["mlp", seed, doc, 1024]
            paired.append({"seed": seed, "document_id": doc,
                           "alpha_short_excess": a1, "mlp_short_excess": m1,
                           "primary": (means["alpha", seed, doc, 12288] - a1)
                                      - (means["mlp", seed, doc, 12288] - m1),
                           "secondary_4k": (means["alpha", seed, doc, 4096] - a1)
                                           - (means["mlp", seed, doc, 4096] - m1)})
    lookup = {(r["seed"], r["document_id"]): r for r in paired}
    primary = mean(r["primary"] for r in paired)
    secondary = mean(r["secondary_4k"] for r in paired)
    rng = random.Random(20260929)
    boot = []
    for _ in range(20000):
        sampled_docs = [rng.choice(docs) for _ in docs]
        sampled_seeds = [rng.choice(seeds) for _ in seeds]
        boot.append(mean(lookup[seed, doc]["primary"] for seed in sampled_seeds for doc in sampled_docs))
    interval = percentile_interval(boot)
    threshold = frozen["estimands"]["smallest_meaningful_primary_nats_per_token"]
    support = short_match and primary >= threshold and interval[0] > 0
    summary = {"freeze_sha256": sha(HERE / "HELDOUT_FREEZE.json"),
        "baseline_rerun_max_token_nll_delta": rerun_floor,
        "document_count": len(docs), "seed_count": len(seeds),
        "results": len(results), "target_ids_scored": len(results) * 96,
        "mean_excess_nll_by_arm_and_history": by_arm_history,
        "heldout_short_dose_matched": short_match,
        "heldout_short_dose_difference_alpha_minus_mlp": short_a - short_m,
        "primary_interaction_nats_per_token": primary,
        "secondary_4k_interaction_nats_per_token": secondary,
        "primary_smallest_meaningful_effect": threshold,
        "primary_bootstrap_95pct": interval,
        "per_seed_primary": {str(seed): mean(lookup[seed, doc]["primary"] for doc in docs) for seed in seeds},
        "per_document_primary": {doc: mean(lookup[seed, doc]["primary"] for seed in seeds) for doc in docs},
        "paired_values": paired,
        "support_gate_passed": support,
        "interpretation": ("pre-frozen support gate passed" if support else
            "pre-frozen support gate not met; inspect dose match, magnitude, direction, and uncertainty separately"),
        "limitations": "Four treatment-held-out books but their baseline scorer values were seen; only one span per book, three sign seeds, and a permanent checkpoint intervention. Natural-text context interaction alone does not isolate recurrent state."
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key not in ("paired_values", "limitations")}, indent=2))


if __name__ == "__main__":
    main()

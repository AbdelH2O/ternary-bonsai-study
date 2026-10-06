"""Audit the 1K natural-text dose sweep and compare three-seed alpha/MLP matching."""

import csv
import hashlib
import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
VARIANTS = ROOT / "analysis/bonsai2/variants/dose_v3"
SEEDS = (20260929, 20260930, 20260931)
ALPHA = lambda seed: f"alpha-d200-c1000-s{seed}"
MLP = lambda seed: f"mlp-d200-c200-s{seed}"
BETA = lambda seed: f"beta-d200-c1000-s{seed}"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def mean(values):
    return statistics.mean(values)


def interval(samples):
    samples.sort()
    return [samples[int(0.025 * len(samples))], samples[int(0.975 * len(samples))]]


def main():
    frozen = json.loads((HERE / "calibration_manifest.json").read_text())
    cases = {x["case_id"]: x for x in frozen["cases"]}
    documents = {x["id"]: x for x in frozen["documents"]}
    baseline = read_rows(HERE / "baseline_run1.jsonl")
    rerun = read_rows(HERE / "baseline_run2.jsonl")
    assert len(baseline) == len(rerun) == len(cases) == 8
    assert [x["case_id"] for x in baseline] == [x["case_id"] for x in rerun]
    rerun_floor = max(abs(a - b) for x, y in zip(baseline, rerun)
                      for a, b in zip(x["nll"], y["nll"]))
    base_by_id = {x["case_id"]: x for x in baseline}
    all_runs = [("baseline_run1", None, baseline), ("baseline_run2", None, rerun)]
    for file in sorted(HERE.glob("*-s*.jsonl")):
        name = file.stem
        patch = VARIANTS / (name + ".json")
        if not patch.exists():
            raise ValueError(f"Missing patch manifest: {name}")
        record = json.loads(patch.read_text())
        assert record["source_sha256"] == frozen["model_sha256"]
        assert record["candidate_sha256"] == sha(VARIANTS / (name + ".gguf"))
        all_runs.append((name, record, read_rows(file)))
    records = []
    sweep = []
    excess_by_arm = {}
    for name, patch, rows in all_runs:
        if len(rows) != len(cases) or {x["case_id"] for x in rows} != set(cases):
            raise ValueError(f"Case identity mismatch: {name}")
        deltas = []
        for row in rows:
            case = cases[row["case_id"]]
            doc = documents[case["document_id"]]
            if (row["p"], row["h"], row["m"], row["n_batch"], row["n_ubatch"]) != (
                    case["target_position"], 1024, 96, 512, 512):
                raise ValueError(f"Scoring profile mismatch: {name} {row['case_id']}")
            if len(row["nll"]) != 96:
                raise ValueError(f"Target count mismatch: {name} {row['case_id']}")
            excess = mean(row["nll"]) - mean(base_by_id[row["case_id"]]["nll"])
            deltas.append(excess)
            records.append({"arm": name, "case_id": row["case_id"], "document_id": doc["id"],
                "source_sha256": doc["source_sha256"], "target_position": row["p"],
                "history_length": row["h"], "target_length": row["m"],
                "history_ids_sha256": case["history_ids_sha256"],
                "target_ids_sha256": case["target_ids_sha256"],
                "model_sha256": frozen["model_sha256"] if patch is None else patch["candidate_sha256"],
                "patch_manifest_sha256": None if patch is None else sha(VARIANTS / (name + ".json")),
                "scorer_revision_sha256": frozen["scorer_source_sha256"],
                "gold_nll_nats": row["nll"], "mean_gold_nll_nats": mean(row["nll"]),
                "excess_mean_nll_nats": excess, "seconds": row["seconds"],
                "backend": {"cuda_layers": 99, "n_ctx": 16384, "n_seq_max": 1,
                            "n_batch": 512, "n_ubatch": 512, "kv_k": "f16", "kv_v": "f16",
                            "flash_attention": "on"}, "error": None})
        excess_by_arm[name] = deltas
        sweep.append({"arm": name, "seed": None if patch is None else patch["seed"],
                      "delta": None if patch is None else patch["delta"],
                      "coverage": None if patch is None else patch["coverage_requested"],
                      "mean_excess_nll": mean(deltas), "positive_spans": sum(v > 0 for v in deltas),
                      "min_span_excess_nll": min(deltas), "max_span_excess_nll": max(deltas)})
    (HERE / "calibration_results.jsonl").write_text("".join(json.dumps(x) + "\n" for x in records))
    with (HERE / "sweep_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sweep[0].keys())
        writer.writeheader()
        writer.writerows(sweep)

    paired = []
    for seed in SEEDS:
        for label, fn in (("alpha", ALPHA), ("beta", BETA), ("mlp", MLP)):
            if fn(seed) not in excess_by_arm:
                raise ValueError(f"Missing three-seed arm {fn(seed)}")
        paired.append({"seed": seed, "alpha_mean": mean(excess_by_arm[ALPHA(seed)]),
                       "beta_mean": mean(excess_by_arm[BETA(seed)]),
                       "mlp_mean": mean(excess_by_arm[MLP(seed)]),
                       "mlp_minus_alpha": mean(excess_by_arm[MLP(seed)]) - mean(excess_by_arm[ALPHA(seed)])})
    alpha_mean = mean(x["alpha_mean"] for x in paired)
    mlp_mean = mean(x["mlp_mean"] for x in paired)
    # Cluster bootstrap: independently resample source documents and sign seeds.
    rng = random.Random(20260929)
    boot_alpha, boot_mlp, boot_diff = [], [], []
    for _ in range(20000):
        doc_indices = [rng.randrange(4) for _ in range(4)]
        seed_indices = [rng.randrange(3) for _ in range(3)]
        a = mean(excess_by_arm[ALPHA(SEEDS[s])][2 * d + k]
                 for s in seed_indices for d in doc_indices for k in (0, 1))
        m = mean(excess_by_arm[MLP(SEEDS[s])][2 * d + k]
                 for s in seed_indices for d in doc_indices for k in (0, 1))
        boot_alpha.append(a)
        boot_mlp.append(m)
        boot_diff.append(m - a)
    summary = {"calibration_manifest_sha256": sha(HERE / "calibration_manifest.json"),
        "baseline_rerun_max_token_nll_delta": rerun_floor,
        "selected": {"alpha": {"delta": 0.20, "coverage": 1.0},
                     "mlp": {"delta": 0.20, "coverage": 0.20}, "seeds": SEEDS},
        "per_seed": paired,
        "mean_excess_nll": {"alpha": alpha_mean, "mlp": mlp_mean,
                            "beta_at_20pct": mean(x["beta_mean"] for x in paired),
                            "mlp_minus_alpha": mlp_mean - alpha_mean},
        "cluster_bootstrap_95pct": {"alpha": interval(boot_alpha),
                                     "mlp": interval(boot_mlp), "mlp_minus_alpha": interval(boot_diff)},
        "gate": {"all_alpha_seed_means_positive": all(x["alpha_mean"] > 0 for x in paired),
                 "all_mlp_seed_means_positive": all(x["mlp_mean"] > 0 for x in paired),
                 "both_modest": 0.002 <= alpha_mean <= 0.02 and 0.002 <= mlp_mean <= 0.02,
                 "matched_within_0.003_nats": abs(mlp_mean - alpha_mean) <= 0.003,
                 "beta_reproducibly_damaging": all(x["beta_mean"] > 0 for x in paired)},
        "sweep": sweep}
    summary["gate"]["go_to_small_retrieval_baseline"] = all(
        summary["gate"][k] for k in ("all_alpha_seed_means_positive", "all_mlp_seed_means_positive",
                                      "both_modest", "matched_within_0.003_nats"))
    (HERE / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("mean_excess_nll", "cluster_bootstrap_95pct", "gate")}, indent=2))


if __name__ == "__main__":
    main()

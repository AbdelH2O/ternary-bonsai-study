"""Freeze the held-out natural-text estimand before treatment inference."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SCORER = ROOT / "analysis/bonsai2/scorer"
DOSE = ROOT / "analysis/bonsai2/dose_calibration"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    scorer_manifest = SCORER / "frozen_manifest.json"
    scorer_data = json.loads(scorer_manifest.read_text())
    dose_file = DOSE / "DOSE_FREEZE.json"
    dose_data = json.loads(dose_file.read_text())
    if sha(SCORER / "position_scorer.cpp") != scorer_data["scorer_source_sha256"]:
        raise ValueError("Scorer source changed")
    if sha(ROOT / scorer_data["model_path"]) != scorer_data["model_sha256"]:
        raise ValueError("Original model changed")
    if dose_data["heldout_scorer_manifest_sha256"] != sha(scorer_manifest):
        raise ValueError("Dose freeze references a different held-out set")
    if dose_data["baseline_model_sha256"] != scorer_data["model_sha256"]:
        raise ValueError("Original model identity differs")
    for doc in scorer_data["documents"]:
        if sha(ROOT / doc["source_path"]) != doc["source_sha256"]:
            raise ValueError(f"Source changed: {doc['id']}")
        if sha(ROOT / doc["ids_path"]) != doc["ids_file_sha256"]:
            raise ValueError(f"Token IDs changed: {doc['id']}")
    cases_file = SCORER / "baseline_cases.tsv"
    cases = [x.split() for x in cases_file.read_text().splitlines()]
    expected = {c["case_id"] for c in scorer_data["cases"]}
    if len(cases) != 12 or {x[0] for x in cases} != expected:
        raise ValueError("Case TSV differs from frozen manifest")
    baseline = SCORER / "baseline_results.jsonl"
    rows = [json.loads(x) for x in baseline.read_text().splitlines()]
    if len(rows) != 12 or {r["case_id"] for r in rows} != expected:
        raise ValueError("Scorer baseline differs from frozen cases")
    for row in rows:
        if row["model_sha256"] != scorer_data["model_sha256"] or len(row["gold_nll_nats"]) != 96:
            raise ValueError("Invalid scorer baseline")
    doc_targets = {}
    for case in scorer_data["cases"]:
        previous = doc_targets.setdefault(case["document_id"], case["target_ids_sha256"])
        if previous != case["target_ids_sha256"] or case["scored_ids_sha256"] != previous:
            raise ValueError("Target IDs differ across histories")
    variants = dose_data["selected_variants"]
    if len(variants) != 6 or {v["arm"] for v in variants} != {"alpha", "mlp"}:
        raise ValueError("Unexpected selected variants")
    frozen = {
        "purpose": "held-out, position-matched natural-text alpha-versus-MLP contrast",
        "scorer_manifest_sha256": sha(scorer_manifest),
        "dose_freeze_sha256": sha(dose_file),
        "baseline_results_sha256": sha(baseline),
        "cases_tsv_sha256": sha(cases_file),
        "scorer_source_sha256": scorer_data["scorer_source_sha256"],
        "scorer_binary_sha256": sha(SCORER / "position_scorer"),
        "original_model_path": scorer_data["model_path"],
        "original_model_sha256": scorer_data["model_sha256"],
        "documents": [d["id"] for d in scorer_data["documents"]],
        "case_ids": [c["case_id"] for c in scorer_data["cases"]],
        "target_policy": scorer_data["position_convention"],
        "history_lengths": [1024, 4096, 12288],
        "target_length": 96,
        "variants": variants,
        "profile": {"n_ctx": 16384, "n_seq_max": 1, "n_batch": 512,
                    "n_ubatch": 512, "cuda_layers": 99,
                    "kv_k": "f16", "kv_v": "f16", "flash_attention": "on"},
        "estimands": {
            "primary": "mean over documents and paired seeds of (alpha_12288-alpha_1024)-(mlp_12288-mlp_1024), where each value is mean 96-token gold NLL on identical target IDs",
            "secondary": "same interaction at 4096 versus 1024; per-document and per-seed values",
            "direction": "positive supports extra long-history alpha damage",
            "smallest_meaningful_primary_nats_per_token": 0.005,
            "heldout_dose_match": "alpha and MLP 1024-history excess NLL versus original must each average 0.002 to 0.020 nats/token, and their difference must be at most 0.003 nats/token in absolute value",
            "sampling": "four documents and three paired sign seeds; equal weight per document-seed; no token-level independence assumption",
            "interval": "20,000 percentile cluster-bootstrap replicates, resampling four documents and three paired seeds independently with replacement, fixed RNG seed 20260929",
            "support_gate": "held-out doses match, primary point estimate >= +0.005, and bootstrap 95% lower bound > 0; otherwise report inconclusive or contrary direction without changing threshold",
        },
    }
    HERE.mkdir(parents=True, exist_ok=True)
    target = HERE / "HELDOUT_FREEZE.json"
    encoded = json.dumps(frozen, indent=2) + "\n"
    if target.exists() and target.read_text() != encoded:
        raise ValueError("Existing held-out freeze differs")
    target.write_text(encoded)
    print(f"Frozen {len(cases)} cases, {len(variants)} variants, threshold +0.005 nat/token")


if __name__ == "__main__":
    main()

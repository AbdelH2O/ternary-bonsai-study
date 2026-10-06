"""Freeze matched calibration doses and a small, baseline-only retrieval pilot."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
VARIANTS = ROOT / "analysis/bonsai2/variants/dose_v3"
PROMPTS = ROOT / "analysis/bonsai2/prompts/retrieval_v2.jsonl"
SEEDS = (20260929, 20260930, 20260931)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    summary = json.loads((HERE / "analysis_summary.json").read_text())
    if not summary["gate"]["go_to_small_retrieval_baseline"]:
        raise ValueError("Dose gate did not pass")
    variants = []
    for seed in SEEDS:
        for arm, delta, coverage in (("alpha", 0.20, 1.0), ("mlp", 0.20, 0.20)):
            name = f"{arm}-d200-c{round(coverage * 1000):03d}-s{seed}"
            patch = VARIANTS / (name + ".json")
            record = json.loads(patch.read_text())
            if (record["arm"], record["delta"], record["coverage_requested"], record["seed"]) != (
                    arm, delta, coverage, seed):
                raise ValueError(f"Wrong candidate: {name}")
            variants.append({"name": name, "seed": seed, "arm": arm,
                "model_path": record["candidate_path"], "model_sha256": record["candidate_sha256"],
                "patch_manifest_path": str(patch.relative_to(ROOT)),
                "patch_manifest_sha256": sha(patch),
                "changed_values_after_rounding": record["changed_values_after_rounding"]})
    prompts = [json.loads(line) for line in PROMPTS.read_text().splitlines()]
    pilot = [r for r in prompts if r["registry_id"] in {f"registry-{i:02d}" for i in range(4)}]
    if len(pilot) != 48:
        raise ValueError("Expected 4 registries x 2 questions x 3 layouts x 2 swaps")
    frozen = {"status": "calibration doses frozen; baseline retrieval pilot only",
        "calibration_manifest_sha256": sha(HERE / "calibration_manifest.json"),
        "calibration_analysis_sha256": sha(HERE / "analysis_summary.json"),
        "baseline_model_sha256": "3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1",
        "heldout_scorer_manifest_sha256": sha(ROOT / "analysis/bonsai2/scorer/frozen_manifest.json"),
        "scorer_source_sha256": sha(ROOT / "analysis/bonsai2/scorer/position_scorer.cpp"),
        "short_history_profile": {"history": 1024, "target": 96, "n_ctx": 16384,
                                  "n_batch": 512, "n_ubatch": 512, "n_seq_max": 1,
                                  "kv": "f16", "cuda_layers": 99},
        "selected_variants": variants,
        "mean_short_excess_nll": summary["mean_excess_nll"],
        "retrieval_pilot": {"purpose": "baseline-only accuracy and answer-swap sensitivity; no perturbation inference",
                            "prompt_file": str(PROMPTS.relative_to(ROOT)),
                            "prompt_file_sha256": sha(PROMPTS),
                            "registry_ids": [f"registry-{i:02d}" for i in range(4)],
                            "item_ids": [r["id"] for r in pilot], "item_count": len(pilot),
                            "generation": {"temperature": 0, "seed": 20260929,
                                           "thinking_budget_tokens": 0, "max_tokens": 32}}}
    out = HERE / "DOSE_FREEZE.json"
    encoded = json.dumps(frozen, indent=2) + "\n"
    if out.exists() and out.read_text() != encoded:
        raise ValueError("Existing dose freeze differs")
    out.write_text(encoded)
    print(f"Frozen {len(variants)} selected candidates and {len(pilot)} baseline pilot prompts")


if __name__ == "__main__":
    main()

"""Freeze a balanced, small retrieval subset before variant inference."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = HERE / "DOSE_FREEZE.json"
    dose = json.loads(source.read_text())
    ids = [item for item in dose["retrieval_pilot"]["item_ids"]
           if item.startswith(("rv2-00-", "rv2-01-"))]
    assert len(ids) == 24
    for registry in ("00", "01"):
        for question in ("q1", "q2"):
            for swap in ("base", "swap"):
                assert {f"rv2-{registry}-{question}-{layout}-{swap}"
                        for layout in ("short-near", "long-near", "long-far")} <= set(ids)
    baseline = HERE / "retrieval_baseline_pilot.jsonl"
    rows = {r["id"]: r for r in map(json.loads, baseline.read_text().splitlines())}
    assert all(rows[item]["exact_correct"] and rows[item]["error"] is None for item in ids)
    frozen = {
        "purpose": "small matched alpha/MLP perturbation retrieval sensitivity pilot",
        "source_dose_freeze_sha256": sha(source),
        "source_baseline_responses_sha256": sha(baseline),
        "prompt_file": dose["retrieval_pilot"]["prompt_file"],
        "prompt_file_sha256": dose["retrieval_pilot"]["prompt_file_sha256"],
        "baseline_model_sha256": dose["baseline_model_sha256"],
        "variant_order": [r["name"] for r in dose["selected_variants"]],
        "variants": dose["selected_variants"],
        "item_ids": ids,
        "item_count_per_variant": len(ids),
        "generation": dose["retrieval_pilot"]["generation"],
        "profile": {"context": 16384, "parallel": 1, "n_batch": 512,
                    "n_ubatch": 512, "kv": "f16", "gpu_layers": 99,
                    "thinking": "off", "cache_prompt": False},
        "outcomes": {
            "primary": "exact code accuracy by layout and arm; paired far-minus-near error count",
            "secondary": "extracted-code accuracy and base/swap tracking",
            "sensitivity_gate": "At least one exact error in an intervention arm is required to treat this task as accuracy-sensitive at the frozen doses. Zero errors means no-go for an accuracy-only full retrieval matrix.",
            "interpretation": "Descriptive pilot only; a far-specific mechanism or held-out effect cannot be established from two registries."
        },
    }
    target = HERE / "PERTURBED_RETRIEVAL_PILOT_FREEZE.json"
    encoded = json.dumps(frozen, indent=2) + "\n"
    if target.exists() and target.read_text() != encoded:
        raise ValueError("Existing freeze differs")
    target.write_text(encoded)
    print(f"Frozen {len(ids)} IDs and {len(frozen['variants'])} variants: {target}")


if __name__ == "__main__":
    main()

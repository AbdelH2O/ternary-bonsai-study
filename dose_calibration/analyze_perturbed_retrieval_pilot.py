"""Audit and summarize the frozen matched-dose retrieval sensitivity pilot."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    freeze = json.loads((HERE / "PERTURBED_RETRIEVAL_PILOT_FREEZE.json").read_text())
    if sha(HERE / "DOSE_FREEZE.json") != freeze["source_dose_freeze_sha256"]:
        raise ValueError("Dose freeze changed")
    if sha(HERE / "retrieval_baseline_pilot.jsonl") != freeze["source_baseline_responses_sha256"]:
        raise ValueError("Baseline responses changed")
    prompts_path = ROOT / freeze["prompt_file"]
    if sha(prompts_path) != freeze["prompt_file_sha256"]:
        raise ValueError("Frozen prompts changed")
    prompts = {r["id"]: r for r in map(json.loads, prompts_path.read_text().splitlines())}
    raw = [json.loads(line) for line in (HERE / "retrieval_perturbed_pilot.jsonl").read_text().splitlines()]
    variants = {v["name"]: v for v in freeze["variants"]}
    expected_keys = {(name, item) for name in variants for item in freeze["item_ids"]}
    actual_keys = {(r["variant"], r["id"]) for r in raw}
    if len(raw) != len(expected_keys) or actual_keys != expected_keys:
        raise ValueError(f"Missing or duplicate rows: {len(raw)} vs {len(expected_keys)}")
    by_key = {(r["variant"], r["id"]): r for r in raw}
    baseline = {r["id"]: r for r in map(json.loads, (HERE / "retrieval_baseline_pilot.jsonl").read_text().splitlines())}
    for row in raw:
        src = prompts[row["id"]]
        variant = variants[row["variant"]]
        for key, value in (("prompt_sha256", src["prompt_sha256"]),
                           ("chat_sha256", src["chat_sha256"]),
                           ("gold", src["answer"]),
                           ("model_sha256", variant["model_sha256"]),
                           ("patch_manifest_sha256", variant["patch_manifest_sha256"]),
                           ("arm", variant["arm"]), ("seed", variant["seed"]),
                           ("expected_prompt_tokens", src["prompt_tokens"]),
                           ("prompt_tokens", src["prompt_tokens"])):
            if row[key] != value:
                raise ValueError(f"Mismatch: {row['variant']} {row['id']} {key}")
        if row["error"] is not None:
            raise ValueError(f"Request failed: {row['variant']} {row['id']}: {row['error']}")
        if not baseline[row["id"]]["exact_correct"]:
            raise ValueError(f"Baseline error on pilot item: {row['id']}")
    per_variant = []
    for name in freeze["variant_order"]:
        variant = variants[name]
        rows = [by_key[name, item] for item in freeze["item_ids"]]
        layouts = {}
        for layout in ("short-near", "long-near", "long-far"):
            selected = [r for r in rows if r["layout"] == layout]
            layouts[layout] = {"n": len(selected),
                               "exact_correct": sum(r["exact_correct"] for r in selected),
                               "extracted_correct": sum(r["extracted_correct"] for r in selected)}
        if any(d["n"] != 8 for d in layouts.values()):
            raise ValueError(f"Unbalanced layouts: {name}")
        swap_pairs_followed = 0
        for registry in ("00", "01"):
            for question in ("q1", "q2"):
                for layout in ("short-near", "long-near", "long-far"):
                    base = by_key[name, f"rv2-{registry}-{question}-{layout}-base"]
                    swap = by_key[name, f"rv2-{registry}-{question}-{layout}-swap"]
                    if base["gold"] == swap["gold"]:
                        raise ValueError("Frozen swap did not change answer")
                    swap_pairs_followed += bool(base["exact_correct"] and swap["exact_correct"])
        near_errors = 8 - layouts["long-near"]["exact_correct"]
        far_errors = 8 - layouts["long-far"]["exact_correct"]
        per_variant.append({"variant": name, "arm": variant["arm"], "seed": variant["seed"],
                            "exact_correct": sum(r["exact_correct"] for r in rows),
                            "extracted_correct": sum(r["extracted_correct"] for r in rows),
                            "layout": layouts,
                            "long_far_minus_near_error_count": far_errors - near_errors,
                            "swap_pairs_followed": swap_pairs_followed,
                            "wrong": [{"id": r["id"], "gold": r["gold"],
                                       "response": r["response"],
                                       "extracted_code": r["extracted_code"]}
                                      for r in rows if not r["exact_correct"]]})
    by_arm = defaultdict(list)
    for item in per_variant:
        by_arm[item["arm"]].append(item)
    arm_summary = {}
    for arm, items in by_arm.items():
        arm_summary[arm] = {"n": sum(24 for _ in items),
                            "exact_correct": sum(x["exact_correct"] for x in items),
                            "long_far_minus_near_error_count": sum(
                                x["long_far_minus_near_error_count"] for x in items),
                            "seed_exact_correct": {str(x["seed"]): x["exact_correct"] for x in items}}
    errors = sum(24 - r["exact_correct"] for r in per_variant)
    decision = ("sensitivity observed; design a larger held-out test with a frozen meaningful effect threshold"
                if errors else "no-go for an accuracy-only full retrieval matrix at these doses; validate likelihood or harder exchangeable interference")
    summary = {"freeze_sha256": sha(HERE / "PERTURBED_RETRIEVAL_PILOT_FREEZE.json"),
               "prompt_file_sha256": freeze["prompt_file_sha256"],
               "baseline_subset_exact_correct": 24,
               "requests": len(raw), "errors": errors,
               "per_variant": per_variant, "by_arm": arm_summary,
               "decision": decision,
               "limitations": "Two registries, post-calibration pilot; accuracy only. Any contrast is descriptive, not a recurrent-state mechanism test."}
    out = HERE / "retrieval_perturbed_pilot_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_variant"}, indent=2))
    for item in per_variant:
        print(item["variant"], item["exact_correct"], "/24", "far-near errors", item["long_far_minus_near_error_count"])


if __name__ == "__main__":
    main()

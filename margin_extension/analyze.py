"""Apply the frozen validity gate and summarize registry-level margin sensitivity."""

import hashlib
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_scores(path, cases):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != len(cases) or [r["case_id"] for r in rows] != [c["case_id"] for c in cases]:
        raise ValueError(f"Scored case identities differ: {path}")
    for row, case in zip(rows, cases):
        if len(row["nll"]) != case["target_token_count"]:
            raise ValueError(f"Incomplete candidate: {case['case_id']}")
    return {r["case_id"]: r for r in rows}


def margins(cases, scores, control):
    selected = [c for c in cases if c["control"] == control]
    prompts = list(dict.fromkeys(c["prompt_id"] for c in selected))
    by_key = {(c["prompt_id"], c["candidate_role"]): c for c in selected}
    if len(prompts) != 24 or len(selected) != 48:
        raise ValueError(f"Unbalanced {control} candidate set")
    result = []
    for prompt in prompts:
        gold_case = by_key[prompt, "gold"]
        wrong_case = by_key[prompt, "wrong"]
        gold = scores[gold_case["case_id"]]
        wrong = scores[wrong_case["case_id"]]
        result.append({"prompt_id": prompt, "registry_id": gold_case["registry_id"],
                       "layout": gold_case["layout"], "swap": gold_case["swap"],
                       "margin_nats": sum(wrong["nll"]) - sum(gold["nll"]),
                       "gold_nll_nats": gold["nll"], "wrong_nll_nats": wrong["nll"]})
    return result


def summary(rows):
    return {"n_prompts": len(rows), "positive_count": sum(r["margin_nats"] > 0 for r in rows),
            "mean_margin_nats": statistics.mean(r["margin_nats"] for r in rows),
            "min_margin_nats": min(r["margin_nats"] for r in rows),
            "max_margin_nats": max(r["margin_nats"] for r in rows)}


def group_means(rows, key):
    return {str(value): statistics.mean(r["margin_nats"] for r in rows if r[key] == value)
            for value in dict.fromkeys(r[key] for r in rows)}


def matched_far_minus_near(rows):
    by_key = {(r["registry_id"], r["swap"], r["layout"]): r for r in rows}
    pairs = []
    for registry, swap in dict.fromkeys((registry, swap) for registry, swap, _ in by_key):
        near = by_key[registry, swap, "long-near"]
        far = by_key[registry, swap, "long-far"]
        pairs.append({"registry_id": registry, "swap": swap,
                      "far_minus_near_margin_nats": far["margin_nats"] - near["margin_nats"]})
    return {"mean_far_minus_near_margin_nats": statistics.mean(p["far_minus_near_margin_nats"] for p in pairs),
            "pairs": pairs}


def main():
    frozen = json.loads((HERE / "MARGIN_EXTENSION_FREEZE.json").read_text())
    cases = frozen["cases"]
    present_cases = [c for c in cases if c["control"] == "present"]
    base_scores = read_scores(HERE / "baseline.jsonl", cases)
    present = margins(cases, base_scores, "present")
    removed = margins(cases, base_scores, "target-removed")
    by_removed = {r["prompt_id"]: r for r in removed}
    pairs = [{**row, "removed_margin_nats": by_removed[row["prompt_id"]]["margin_nats"],
              "removal_margin_drop_nats": row["margin_nats"] - by_removed[row["prompt_id"]]["margin_nats"]}
             for row in present]
    by_id = {c["case_id"]: c for c in cases}
    for registry in frozen["registries"]:
        for layout in ("short-near", "long-near", "long-far"):
            stem = f"rv2-{registry:02d}-q1-{layout}"
            a = by_id[f"{stem}-base-present-gold"]
            b = by_id[f"{stem}-swap-present-gold"]
            if (a["candidate_code"], a["wrong_code"]) != (b["wrong_code"], b["candidate_code"]):
                raise ValueError(f"Base/swap control failed: {stem}")
        for swap in ("base", "swap"):
            near = by_id[f"rv2-{registry:02d}-q1-long-near-{swap}-present-gold"]
            far = by_id[f"rv2-{registry:02d}-q1-long-far-{swap}-present-gold"]
            if near["prefix_token_count"] != far["prefix_token_count"]:
                raise ValueError(f"Long near/far lengths differ: registry {registry} {swap}")
    positive = sum(p["margin_nats"] > 0 for p in pairs)
    drops = sum(p["removal_margin_drop_nats"] >= 1 for p in pairs)
    gate = {"freeze_sha256": sha(HERE / "MARGIN_EXTENSION_FREEZE.json"),
            "baseline_raw_sha256": sha(HERE / "baseline.jsonl"),
            "present_positive": positive, "target_removal_drop_ge_1_nats": drops,
            "passed": positive >= 20 and drops >= 20,
            "baseline_present": summary(present), "baseline_removed": summary(removed),
            "mean_removal_margin_drop_nats": statistics.mean(p["removal_margin_drop_nats"] for p in pairs),
            "by_registry_present": group_means(present, "registry_id"),
            "by_layout_present": group_means(present, "layout"),
            "matched_far_minus_near": matched_far_minus_near(present), "pairs": pairs}
    (HERE / "baseline_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    print(f"Baseline gate: {gate['passed']}; positive {positive}/24, removal drop >=1 {drops}/24", flush=True)
    if not gate["passed"]:
        return
    variants = []
    base_by_prompt = {r["prompt_id"]: r["margin_nats"] for r in present}
    for variant in frozen["variants"]:
        path = HERE / f"{variant['name']}.jsonl"
        if not path.exists():
            continue
        scores = read_scores(path, present_cases)
        rows = margins(cases, scores, "present")
        delta = [{**r, "delta_vs_baseline_nats": r["margin_nats"] - base_by_prompt[r["prompt_id"]]}
                 for r in rows]
        variants.append({"name": variant["name"], "arm": variant["arm"],
                         "raw_sha256": sha(path), **summary(rows),
                         "mean_delta_vs_baseline_nats": statistics.mean(r["delta_vs_baseline_nats"] for r in delta),
                         "by_registry_margin": group_means(rows, "registry_id"),
                         "by_layout_margin": group_means(rows, "layout"),
                         "matched_far_minus_near": matched_far_minus_near(rows),
                         "change_in_far_minus_near_vs_baseline_nats":
                             matched_far_minus_near(rows)["mean_far_minus_near_margin_nats"] -
                             gate["matched_far_minus_near"]["mean_far_minus_near_margin_nats"],
                         "paired_prompts": delta})
        print(f"{variant['name']}: positive {variants[-1]['positive_count']}/24, "
              f"mean margin {variants[-1]['mean_margin_nats']:.6f}, "
              f"delta {variants[-1]['mean_delta_vs_baseline_nats']:.6f}", flush=True)
    if variants:
        output = {"freeze_sha256": gate["freeze_sha256"], "baseline_gate_passed": True,
                  "baseline": gate, "variants": variants,
                  "interpretation": "Four registry clusters and one perturbation seed per arm; descriptive sensitivity only."}
        (HERE / "variant_summary.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()

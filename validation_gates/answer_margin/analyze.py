"""Check the predeclared answer-margin validity controls and variant sensitivity."""

import hashlib
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_scores(path, cases):
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    if len(rows) != len(cases) or [r["case_id"] for r in rows] != [c["case_id"] for c in cases]:
        raise ValueError("Scored case identity mismatch")
    return {r["case_id"]: r for r in rows}


def summarize(name, cases, rows):
    by_key = {(c["prompt_id"], c["control"], c["candidate_role"]): c for c in cases}
    prompt_ids = list(dict.fromkeys(c["prompt_id"] for c in cases))
    pairs = []
    for prompt_id in prompt_ids:
        group = {}
        for control in ("present", "target-removed"):
            gold = rows[by_key[prompt_id, control, "gold"]["case_id"]]
            wrong = rows[by_key[prompt_id, control, "wrong"]["case_id"]]
            # log p(gold) - log p(wrong) = NLL(wrong) - NLL(gold).
            margin = sum(wrong["nll"]) - sum(gold["nll"])
            group[control] = {"margin_nats": margin,
                              "gold_nll_nats": gold["nll"], "wrong_nll_nats": wrong["nll"]}
        pairs.append({"prompt_id": prompt_id, "layout": by_key[prompt_id, "present", "gold"]["layout"],
                      "swap": by_key[prompt_id, "present", "gold"]["swap"],
                      "present": group["present"], "target_removed": group["target-removed"],
                      "removal_margin_drop_nats": group["present"]["margin_nats"] - group["target-removed"]["margin_nats"]})
    positive = sum(p["present"]["margin_nats"] > 0 for p in pairs)
    removal_drops = sum(p["removal_margin_drop_nats"] >= 1 for p in pairs)
    return {"name": name, "present_positive": positive, "present_total": len(pairs),
            "target_removal_drop_ge_1_nats": removal_drops,
            "mean_present_margin_nats": statistics.mean(p["present"]["margin_nats"] for p in pairs),
            "mean_removed_margin_nats": statistics.mean(p["target_removed"]["margin_nats"] for p in pairs),
            "pairs": pairs}


def main():
    frozen = json.loads((HERE / "ANSWER_MARGIN_FREEZE.json").read_text())
    cases = frozen["cases"]
    base = summarize("baseline", cases, read_scores(HERE / "baseline.jsonl", cases))
    # Each base/swap counterpart uses the same two candidate codes in reverse roles.
    for layout in ("short-near", "long-near", "long-far"):
        rows = [c for c in cases if c["layout"] == layout and c["control"] == "present"]
        if len(rows) != 4:
            raise ValueError("Missing base/swap candidate counterpart")
        base_codes = {c["candidate_role"]: c["candidate_code"] for c in rows if c["swap"] == "base"}
        swap_codes = {c["candidate_role"]: c["candidate_code"] for c in rows if c["swap"] == "swap"}
        if base_codes["gold"] != swap_codes["wrong"] or base_codes["wrong"] != swap_codes["gold"]:
            raise ValueError("Value swap did not reverse candidates")
    gate = {"freeze_sha256": sha(HERE / "ANSWER_MARGIN_FREEZE.json"),
            "baseline": base,
            "passed": base["present_positive"] == 6 and base["target_removal_drop_ge_1_nats"] >= 4}
    (HERE / "baseline_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    print(json.dumps({"baseline_gate_passed": gate["passed"],
                      "positive": base["present_positive"],
                      "removal_drops": base["target_removal_drop_ge_1_nats"],
                      "mean_present_margin": base["mean_present_margin_nats"],
                      "mean_removed_margin": base["mean_removed_margin_nats"]}, indent=2))
    if not gate["passed"]:
        return
    variants = []
    for variant in frozen["selected_variants"]:
        path = HERE / f"{variant['name']}.jsonl"
        if path.exists():
            variants.append(summarize(variant["name"], cases, read_scores(path, cases)))
    if variants:
        out = {"freeze_sha256": gate["freeze_sha256"], "baseline": base,
               "variants": variants,
               "interpretation": "Exploratory one-seed sensitivity only; six related prompts do not establish a mechanism or population effect."}
        (HERE / "variant_summary.json").write_text(json.dumps(out, indent=2) + "\n")
        for v in variants:
            print(v["name"], "positive", v["present_positive"],
                  "mean margin", v["mean_present_margin_nats"])


if __name__ == "__main__":
    main()

"""Prompt robustness of FP, C and M on development items (frozen in results/qat08_mc12/protocol.json). Descriptive:
it never changes the MC12 mix, the evaluation wording or any held-out selection.

    python robust_qat08_mc12.py score     # GPU: 3 models x 4 wordings x {letter 1,508, binding 400}; resumes by shard
    python robust_qat08_mc12.py analyze   # CPU: results/qat08_mc12/robustness.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from diag_metrics import binding_metrics, letter_metrics
from diag_run import score_cases
from score_17b import paired_items, sha

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/qat08_mc12"
WORK = ROOT / "work/qat08_mc12"
SCORES = WORK / "robust"
DIAG_SCORES = {"fp": "work/diag_readout/scores/fp-{key}.jsonl", "C": "work/diag_readout/scores/qat_chat_42-final-{key}.jsonl"}


def protocol() -> dict:
    return json.loads((OUT / "protocol.json").read_text())


def model(p: dict, name: str) -> Path:
    entry = p["comparators"][name]
    path = ROOT / entry["file"]
    assert sha(path) == entry["sha256"], name
    return path


def case(p: dict, variant: str, key: str) -> Path:
    entry = p["robustness"]["record"]["files"][f"{variant}-{key}"]
    path = OUT / entry["file"]
    assert sha(path) == entry["sha256"], entry["file"]
    return path


def score() -> None:
    p = protocol()
    SCORES.mkdir(parents=True, exist_ok=True)
    for name in p["robustness"]["models"]:
        for variant in p["robustness"]["variants"]:
            for key in p["robustness"]["sets"]:
                score_cases(model(p, name), case(p, variant, key), SCORES / f"{name}-{variant}-{key}.jsonl")
                print(name, variant, key, "scored", flush=True)


def _row(letter: dict, binding: dict) -> dict:
    return {"letter_raw_accuracy": letter["raw_accuracy"], "letter_calibrated_accuracy": letter["calibrated_accuracy"],
            "letter_bias": letter["bias"], "letter_pairwise": letter["pairwise"], "letter_pairwise_ci": [letter["pairwise_lo"], letter["pairwise_hi"]],
            "letter_mass": letter["letter_mass"], "letter_predicted": letter["predicted"],
            "binding_raw_accuracy": binding["raw_accuracy"], "binding_raw_wilson": binding["raw_wilson"],
            "binding_calibrated_accuracy": binding["calibrated_accuracy"], "binding_calibrated_wilson": binding["calibrated_wilson"],
            "binding_pairwise": binding["pairwise"], "binding_pairwise_ci": [binding["pairwise_lo"], binding["pairwise_hi"]],
            "binding_predicted": binding["predicted"]}


def evaluate(metrics: dict, rules: dict, variants: list[str]) -> dict:
    """metrics[model][variant] = {"letter": letter_metrics(...), "binding": binding_metrics(...)} -> frozen verdict."""
    floor, crit = rules["fp_floor"], rules["criterion"]
    pct = lambda xs: [100 * x for x in xs]
    table, gains, sanity = {}, {}, {}
    for v in variants:
        fp_l, fp_b = metrics["fp"][v]["letter"], metrics["fp"][v]["binding"]
        reasons = []
        if fp_b["raw_accuracy"] < floor["binding_raw_accuracy_min"]:
            reasons.append(f"FP binding raw {fp_b['raw_accuracy']:.2f} < {floor['binding_raw_accuracy_min']}")
        if fp_l["pairwise"] < floor["letter_pairwise_min"]:
            reasons.append(f"FP letter pairwise {fp_l['pairwise']:.2f} < {floor['letter_pairwise_min']}")
        sanity[v] = {"fp_sane": not reasons, "reasons": reasons}
        table[v] = {m: _row(metrics[m][v]["letter"], metrics[m][v]["binding"]) for m in metrics}
        gains[v] = paired_items(pct(metrics["M"][v]["letter"]["pairwise_items"]), pct(metrics["C"][v]["letter"]["pairwise_items"]))
    novel = [v for v in variants if v != "original"]
    sane_novel = [v for v in novel if sanity[v]["fp_sane"]]
    if not sanity["original"]["fp_sane"]:
        status = "harness_invalid"
    elif gains["original"]["lo"] <= crit["original_gain_lo_above"] or len(sane_novel) < crit["min_sane_novel_variants"]:
        status = "inconclusive"
    else:
        ok = {v: gains[v]["lo"] > crit["variant_gain_lo_above"] and gains[v]["mean"] >= crit["variant_gain_share_min"] * gains["original"]["mean"]
              for v in sane_novel}
        status = "wording_robust" if all(ok.values()) else "not_robust"
    return {"status": status, "fp_sanity": sanity, "sane_novel_variants": sane_novel, "m_minus_c_letter_pairwise": gains,
            "table": table, "criterion_variants": ["original", *sane_novel],
            "excluded_from_criterion_only": [v for v in novel if not sanity[v]["fp_sane"]]}


def analyze() -> dict:
    p = protocol()
    r = p["robustness"]
    metrics, files = {}, {}
    for name in r["models"]:
        metrics[name] = {}
        for v in r["variants"]:
            got = {}
            for key in r["sets"]:
                path = SCORES / f"{name}-{v}-{key}.jsonl"
                rows = score_cases(model(p, name), case(p, v, key), path)  # existing file: checked for completeness only
                files[str(path.relative_to(ROOT))] = sha(path)
                got[key] = letter_metrics(rows) if key == "letter" else binding_metrics(rows)
            metrics[name][v] = got
    result = evaluate(metrics, r, list(r["variants"]))
    reproduction = {}
    diag = json.loads((ROOT / "results/diag_readout/results.json").read_text())
    for name, pattern in DIAG_SCORES.items():
        for key in r["sets"]:
            source = ROOT / pattern.format(key=key)
            assert sha(source) == p["input_sha256"][pattern.format(key=key)], f"diagnostics score file changed: {source}"
            old = {x["id"]: x["values"] for x in map(json.loads, source.read_text().splitlines())}
            arm = {"fp": "fp", "C": "qat_chat_42-final"}[name]
            recomputed = (letter_metrics if key == "letter" else binding_metrics)(old)["pairwise"]
            assert abs(recomputed - diag["arms"][arm][key]["pairwise"]) < 1e-9, f"{source} does not reproduce the frozen diagnostics"
            new = {x["id"]: x["values"] for x in map(json.loads, (SCORES / f"{name}-original-{key}.jsonl").read_text().splitlines())}
            reproduction[f"{name}-{key}"] = max(abs(a - b) for i in old for a, b in zip(old[i], new[i]))
    result.update({"protocol_sha256": sha(OUT / "protocol.json"), "score_sha256": files,
                   "reproduction_vs_diagnostics_max_abs_logprob_diff": reproduction})
    tmp = OUT / "robustness.tmp"
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    tmp.replace(OUT / "robustness.json")
    print(json.dumps({"status": result["status"], "fp_sanity": result["fp_sanity"],
                      "gains": {v: {k: round(g[k], 2) for k in ("mean", "lo", "hi")} for v, g in result["m_minus_c_letter_pairwise"].items()}}, indent=2))
    return result


if __name__ == "__main__":
    {"score": score, "analyze": analyze}[sys.argv[1]]()

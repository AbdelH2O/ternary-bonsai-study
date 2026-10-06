"""Render QAT08_MCU.md from results/qat08_mcu/results.json: frozen decision first, then tables and costs."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/qat08_mcu"
WORK = ROOT / "work/qat08_mcu"


def _ci(w: dict) -> str:
    return f"{w['mean']:.1f} [{w['lo']:.1f}, {w['hi']:.1f}]"


def write_report() -> None:
    r = json.loads((OUT / "results.json").read_text())
    p = json.loads((OUT / "protocol.json").read_text())
    s, d = r["summary"]["arms"], r["decision"]
    events = [json.loads(l) for l in (WORK / "execution.jsonl").read_text().splitlines()]
    hours = sum(e["seconds"] for e in events if e.get("event") == "stage_finished") / 3600
    lines = [f"# QAT08-MCU: read-out data x trained FP tensors on Qwen3.5-0.8B — results", "",
             f"**Decision (frozen rules): `{d['recommendation']}`**" + (f" with recipe arm(s) {d['recipe_arms']}." if d["recipe_arms"] else "."), ""]
    for arm, v in d["arms"].items():
        flags = ", ".join(f"{k}={'yes' if val else 'no'}" for k, val in v.items())
        lines.append(f"- 🟢 **{arm}**: {flags}")
    lines += ["", f"Diagnostics branch: `{p['diagnostics']['branch']}`. [Design](results/qat08_mcu/design.json), "
              f"[protocol](results/qat08_mcu/protocol.json), [results](results/qat08_mcu/results.json). "
              "Evidence tags: 🟢 measured; 🟡 inferred; 🟣 proposed.", "",
              "## 🟢 Held-out results (C = chat arm, comparator)", "",
              "| Arm | Books NLL (vs C) | MMLU-Redux acc | Calibrated pairwise (vs C) | Letter mass | MMLU-fresh acc | Fresh pairwise | Binding cal. | GSM8K | GSM8K strict | Retrieval |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n, a in s.items():
        bv, pv = a["books_vs_C"], a["mmlu_pairwise_vs_C"]
        lines.append(f"| {n} | {a['books']:.3f} ({bv['mean']:+.3f} [{bv['lo']:+.3f}, {bv['hi']:+.3f}]) | {_ci(a['mmlu'])} | "
                     f"{a['mmlu_pairwise']:.1f} ({pv['mean']:+.1f} [{pv['lo']:+.1f}, {pv['hi']:+.1f}]) | {a['letter_mass']:.2f} | "
                     f"{_ci(a['mmlu_fresh'])} | {a['mmlu_fresh_pairwise']:.1f} | {a['binding_calibrated']:.1f} | {_ci(a['gsm8k'])} | "
                     f"{_ci(a['gsm8k_strict'])} | {a['retrieval_accuracy']:.1f}% |")
    lines += ["", "## 🟢 Validation curve (descriptive; final checkpoints fixed)", "", "| Model | NLL |", "|---|---:|"]
    lines += [f"| {k} | {v:.4f} |" for k, v in r["curve"].items()]
    lines += ["", "## 🟢 Data, contamination and compute", "",
              f"- MC corpus: {p['data_record']['mc_kept']} kept items, {p['data_record']['mc_unique_tokens']:,} unique tokens, "
              f"{p['data_record']['mc_passes']:.2f} passes; by source {p['data_record']['kept_by_source']}; teacher letter accuracy {p['data_record']['teacher_letter_accuracy']}.",
              f"- MC screen: {p['data']['mc_record']['screen']['dropped']}; eval-cosine quantiles {p['data']['mc_record']['screen']['cosine_quantiles']}.",
              f"- FP LR probe: {p['fp_lr_probe']}.",
              f"- 13-token audit (no majority-overlap case allowed): " + "; ".join(
                  f"{st}: " + ", ".join(f"{k} {v['cases_with_any_13gram_overlap']}/{v['cases']}" for k, v in a['sets'].items())
                  for st, a in p["contamination_13gram"].items()),
              f"- Recorded stage time: {hours:.2f} GPU-h against the 25.0 ceiling (stage wall time, conservative).", "",
              "## Interpretation and limits", "",
              "🟡 (Written by the executing agent: which hypothesis each arm supports, what the result means for the 1.7B rental, and limits: one seed per arm, single 0.8B hybrid, letter-logit MMLU, intervals treat items as independent.)", ""]
    (ROOT / "QAT08_MCU.md").write_text("\n".join(lines) + "\n")
    print("wrote QAT08_MCU.md")

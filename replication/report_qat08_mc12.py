"""Render QAT08_MC12.md from results/qat08_mc12/{results,robustness}.json (decision first). Interpretation, incidents
and reproduction are completed by hand after verification; re-rendering rewrites the generated parts."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/qat08_mc12"
REPORT = ROOT / "QAT08_MC12.md"


def _ci(x: dict, d: int = 1) -> str:
    return f"{x['mean']:+.{d}f} [{x['lo']:+.{d}f}, {x['hi']:+.{d}f}]"


def _w(x: dict) -> str:
    return f"{x['mean']:.1f} [{x['lo']:.1f}, {x['hi']:.1f}]"


def write_report() -> None:
    import mc12_control as c
    r = json.loads((OUT / "results.json").read_text())
    rb = json.loads((OUT / "robustness.json").read_text())
    p = json.loads((OUT / "protocol.json").read_text())
    d, s = r["decision"], r["summary"]["arms"]
    flags = ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in d.items() if isinstance(v, bool))
    lines = [
        "# QAT08-MC12: 12.5% multiple-choice data at M's recipe, with a prompt-robustness check — results", "",
        f"**Decision (frozen rule): `{d['recommendation']}`.** No recommendation authorizes a rental.", "",
        f"- 🟢 Frozen flags: {flags}.",
        f"- 🟢 MC12 minus M, MMLU-Redux calibrated pairwise: {_ci(s['MC12']['mmlu_pairwise_vs_M'])} points; MMLU-fresh2: "
        f"{_ci(s['MC12']['mmlu_fresh2_pairwise_vs_M'])}.",
        f"- 🟢 Binding raw: MC12 {s['MC12']['binding_raw']:.1f}% vs M {s['M']['binding_raw']:.1f}% "
        f"(paired {_ci(s['MC12']['binding_raw_vs_M'])}); target 90 {'met' if d['binding_raw_target_met'] else 'not met'}.",
        f"- 🟢 Prompt robustness of M (development items, descriptive): `{rb['status']}`.", "",
        f"[Design](results/qat08_mc12/design.json), [protocol](results/qat08_mc12/protocol.json), "
        f"[results](results/qat08_mc12/results.json), [robustness](results/qat08_mc12/robustness.json). "
        "Evidence tags: 🟢 measured; 🟡 inferred; 🟣 proposed.", "",
        "## 🟢 Held-out results", "",
        "| Model | Books NLL | vs M | vs C | MMLU-Redux acc | Pairwise | vs M | MMLU-fresh2 acc | Fresh2 pairwise | vs M | "
        "Binding raw | Binding cal. | GSM8K | GSM8K strict | vs M | Retrieval |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n, a in s.items():
        lines.append(
            f"| {n} | {a['books']:.3f} | {_ci(a['books_vs_M'], 3) if a['books_vs_M'] else '—'} | "
            f"{_ci(a['books_vs_C'], 3) if a['books_vs_C'] else '—'} | {_w(a['mmlu'])} | {a['mmlu_pairwise']:.1f} | "
            f"{_ci(a['mmlu_pairwise_vs_M'])} | {_w(a['mmlu_fresh2'])} | {a['mmlu_fresh2_pairwise']:.1f} | "
            f"{_ci(a['mmlu_fresh2_pairwise_vs_M'])} | {a['binding_raw']:.1f} | {a['binding_calibrated']:.1f} | {_w(a['gsm8k'])} | "
            f"{_w(a['gsm8k_strict'])} | {_ci(a['gsm8k_vs_M'])} | {a['retrieval_accuracy']:.1f}% |")
    lines += ["", "## 🟢 Prompt robustness (development items; every wording reported)", "",
              f"FP sanity per wording (binding raw >= 95, letter pairwise >= 60): " +
              "; ".join(f"`{v}` {'sane' if x['fp_sane'] else 'invalid: ' + ', '.join(x['reasons'])}" for v, x in rb["fp_sanity"].items()), "",
              "| Wording | Model | Letter raw | Letter cal. | Letter pairwise [95%] | Letter mass | Binding raw | Binding cal. | Binding pairwise |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for v, rows in rb["table"].items():
        for m, x in rows.items():
            lines.append(f"| {v} | {m} | {x['letter_raw_accuracy']:.1f} | {x['letter_calibrated_accuracy']:.1f} | "
                         f"{x['letter_pairwise']:.1f} [{x['letter_pairwise_ci'][0]:.1f}, {x['letter_pairwise_ci'][1]:.1f}] | "
                         f"{x['letter_mass']:.2f} | {x['binding_raw_accuracy']:.1f} | {x['binding_calibrated_accuracy']:.1f} | "
                         f"{x['binding_pairwise']:.1f} |")
    lines += ["", "M minus C letter pairwise by wording: " +
              "; ".join(f"`{v}` {_ci(g)}" for v, g in rb["m_minus_c_letter_pairwise"].items()) + ".", "",
              "## 🟢 Validation curve (descriptive)", "", "| Model | NLL |", "|---|---:|"]
    lines += [f"| {k} | {v:.4f} |" for k, v in r["curve"].items()]
    st = p["stream"]
    lines += ["", "## 🟢 Data, contamination and compute", "",
              f"- MC12 stream: {st['fineweb_seqs']:,} FineWeb + {st['chat_seqs']:,} chat + {st['mc_seqs']:,} MC sequences; "
              f"{st['mc_tokens']:,} MC tokens from {st['mc_unique_tokens']:,} unique ({st['mc_passes']:.2f} passes).",
              "- Pre-freeze 13-token audit (complete rendered cases, wrappers included): zero majority-overlap cases in every "
              "set and stream; any-overlap counts and maxima are in protocol.json `pre_freeze_stream_audit`.",
              f"- GPU-hours (supervisor ledger, provisional while this report stage runs; final value in the reviewed report): {c.used():.3f} of {c.TOTAL} "
              f"({', '.join(f'{g} {c.used(grp=g):.3f}' for g in c.CEILING)}).", "",
              "## Interpretation and limits", "",
              "🟡 (Written after verification.)", ""]
    REPORT.write_text("\n".join(lines))


if __name__ == "__main__":
    write_report()

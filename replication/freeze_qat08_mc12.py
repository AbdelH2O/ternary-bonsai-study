"""Freeze QAT08-MC12 before any GPU stage (CPU): design.json then protocol.json, both exclusive.

The protocol binds every input: the MC12 stream and its MCU sources, reused MCU comparator outputs, robustness dev
cases, fresh and reused evaluation cases, the frozen trainer/scorer code, the runtime and the environment. GPU stages
need protocol_approval.json bound to both files; it records the user's pre-hash scoped instruction and the parent review.

    python freeze_qat08_mc12.py design     # design.json + protocol.json (exclusive)
    python freeze_qat08_mc12.py verify     # re-check every frozen hash (run before every stage)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import freeze_qat08_mcu as fm
from freeze_qat08 import ngram_hashes
from prepare_qat08_chat import write_new
from prepare_qat08_mc12 import (AUDIT_STREAMS, DATA, LICENSES, MCU_OUT, MCU_PROTOCOL_SHA, OUT, ROOT, SEED, STREAM,
                                VARIANTS, WORK, check_mcu_data, read_cases)
from score_17b import sha

DESIGN, PROTOCOL, APPROVAL = OUT / "design.json", OUT / "protocol.json", OUT / "protocol_approval.json"
ARMS = {"MC12": {"rule": "top86", "stream": "work/qat08_mc12/data/train_MC12.u32", "unfreeze": False, "init": "folded",
                 "schedule": "cosine", "steps": 2000, "seed": SEED}}
COMPARATOR_NAMES = ("fp", "C", "M")
REUSED_KEYS = ("mmlu", "binding", "retrieval")   # MCU scores of fp, C and M on the reused (not fresh) sets
ROBUSTNESS = {
    "models": list(COMPARATOR_NAMES), "variants": VARIANTS, "sets": ["letter", "binding"],
    "dev_only": "MMLU validation letter items (1,508) and synthetic binding items (400) of the read-out diagnostics",
    "fp_floor": {"binding_raw_accuracy_min": 95.0, "letter_pairwise_min": 60.0,
                 "note": "the binding floor uses uncalibrated accuracy (the calibrated floor misfired in the diagnostics); "
                         "the letter floor is FP's pairwise score"},
    "criterion": {"min_sane_novel_variants": 2, "original_gain_lo_above": 0.0, "variant_gain_lo_above": 0.0,
                  "variant_gain_share_min": 0.5},
    "text": "harness_invalid if the original wording fails the FP floor (stop, incident). Otherwise M is wording_robust "
            "when the original-wording M-C letter-pairwise gain has paired 95% lower bound > 0, at least 2 of the 3 novel "
            "wordings pass the FP floor, and on every FP-sane novel wording the M-C gain has lower bound > 0 and mean >= "
            "50% of the original-wording mean; not_robust if those conditions hold except a sane variant fails its gain "
            "test; inconclusive if fewer than 2 novel wordings are FP-sane or the original gain lower bound <= 0. Every "
            "variant row is reported, FP-invalid ones included with the reason. Descriptive only: it never changes the "
            "MC12 mix, the evaluation wording or any held-out selection.",
}
DECISION = {
    "material_pairwise_points": 2.0, "fresh_lo_above": 0.0, "binding_raw_target": 90.0,
    "costs": {"books_vs_M_hi": 0.05, "books_vs_C_hi": 0.10, "gsm8k_vs_M_lo": -3.0, "retrieval_min": 90.0},
    "recovery": {"mmlu_lo": 30.0, "fresh_lo": 30.0, "gsm8k_lo": 10.0},
    "text": {
        "material_mc_effect": "MC12 minus M MMLU-Redux calibrated pairwise mean >= +2.0 points with paired 95% lower bound > 0",
        "fresh_corroborated": "MC12 minus M MMLU-fresh2 calibrated pairwise paired 95% lower bound > 0",
        "binding": "MC12 raw binding accuracy and its difference from M are reported; target raw >= 90 reported as met/unmet",
        "costs_ok": "book NLL MC12 minus M upper bound <= +0.05 AND MC12 minus C upper bound <= +0.10 AND GSM8K MC12 minus M "
                    "paired lower bound >= -3 points AND retrieval accuracy >= 90",
        "recovery_floor": "MMLU-Redux and MMLU-fresh2 Wilson lower > 30 AND GSM8K and strict GSM8K Wilson lower > 10 (inherited)",
        "recommendation": "increase_mc_to_12.5 if material_mc_effect AND fresh_corroborated AND costs_ok AND recovery_floor; "
                          "else keep_m_6.25_mc12_worse if MC12 minus M MMLU-Redux pairwise upper bound < 0; else "
                          "keep_m_6.25_inconclusive. No recommendation authorizes a rental.",
        "justification": "+2 pairwise points is about a quarter of M's +8.6 gain over C and above the paired interval "
                         "half-width observed for M vs C (about 1.3), so a smaller change would not be material; the cost "
                         "and recovery thresholds are the MCU ones, with the book cost against M halved to +0.05 because "
                         "MC12 replaces FineWeb rows. Fixed before any MC12 or robustness scoring.",
    },
}
BUDGET = {"total_gpu_hours": 8.0, "robust_gpu_hours": 0.75, "train_gpu_hours": 6.5, "eval_gpu_hours": 0.75,
          "grace_s": 300, "heartbeat_s": 600,
          "accounting": "supervisor wall time of every stage, failed and paused sessions included; an unfinished stage is "
                        "charged to its last heartbeat plus one heartbeat interval (capped at the next supervisor start); "
                        "the 300 s pause grace is reserved inside each ceiling; paused downtime is not charged; no raise"}
NEW_FILES = ("prepare_qat08_mc12.py", "freeze_qat08_mc12.py", "robust_qat08_mc12.py", "mc12_control.py",
             "score_qat08_mc12.py", "report_qat08_mc12.py", "test_qat08_mc12.py")
REUSED_CODE = ("qat_08b_mcu.py", "qat_08b.py", "qat08_mcu_control.py", "qat08_chat_runtime.py", "prepare_qat08_mcu.py",
               "freeze_qat08_mcu.py", "score_qat08_mcu.py", "diag_cases.py", "diag_metrics.py", "diag_run.py",
               "score_17b.py", "score_17b.cpp", "gsm8k_17b.py", "freeze_17b.py", "freeze_qat08.py",
               "prepare_qat08_chat.py", "phase0.py", "test_qat08_mcu.py", "make_gptq_pilot.py", "../inspect_gguf.py",
               "work/qwen3_17b/score_17b",
               "../../../bin/cuda/llama-server")


def mcu_results() -> dict:
    return json.loads((MCU_OUT / "results.json").read_text())


def comparators() -> dict:
    mcu, res = json.loads((MCU_OUT / "protocol.json").read_text()), mcu_results()
    out = {n: dict(mcu["comparators"][n]) for n in ("fp", "C")}
    out["M"] = {**res["models"]["M"], "export_record": "results/qat08_mcu/export_M-final.json",
                "export_record_sha256": sha(MCU_OUT / "export_M-final.json")}
    return out


def reused_outputs() -> dict:
    res = mcu_results()
    scores = {f"work/qat08_mcu/scores/{n}-{k}.jsonl": res["raw_output_sha256"][f"work/qat08_mcu/scores/{n}-{k}.jsonl"]
              for n in COMPARATOR_NAMES for k in REUSED_KEYS}
    gsm8k = {f"results/qat08_mcu/gsm8k/{n}.jsonl": res["gsm8k_output_sha256"][n] for n in COMPARATOR_NAMES}
    return {"scores": scores, "gsm8k": gsm8k, "mcu_results_sha256": sha(MCU_OUT / "results.json")}


def stream_audit(streams=AUDIT_STREAMS, cases=None) -> dict:
    """Every evaluation case (complete rendered case, wrappers included) against each stream; majority (> 0.5) blocks."""
    cases = cases if cases is not None else read_cases()
    out, flagged = {}, []
    for name in streams:
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        seen = np.unique(ngram_hashes(np.asarray(np.memmap(path, dtype=np.uint32, mode="r"))))
        sets = {}
        for key, rows in cases.items():
            scored = []
            for case_id, ids in rows:
                h = ngram_hashes(ids) if len(ids) >= 13 else np.asarray([], dtype=np.uint64)
                hits = int((seen[np.minimum(np.searchsorted(seen, h), len(seen) - 1)] == h).sum()) if len(h) else 0
                scored.append((hits / len(h) if len(h) else 0.0, case_id))
            scored.sort(reverse=True)
            flagged += [(name, key, cid, round(fr, 4)) for fr, cid in scored if fr > .5]
            sets[key] = {"cases": len(rows), "cases_with_majority_13gram_overlap": sum(fr > .5 for fr, _ in scored),
                         "cases_with_any_13gram_overlap": sum(fr > 0 for fr, _ in scored),
                         "cases_over_0.3": sum(fr > .3 for fr, _ in scored), "max": round(scored[0][0], 4) if scored else 0.0,
                         "top": [[cid, round(fr, 4)] for fr, cid in scored[:3]]}
        out[name] = {"sha256": sha(path), "sets": sets}
    assert not flagged, f"evaluation cases with majority 13-gram overlap: {flagged}"
    return out


CASE_KEYS = ("mmlu", "mmlu_fresh2", "binding", "gsm8k", "retrieval")


def case_entries(cases: dict) -> dict[str, str]:
    """Every actual evaluation file of the cases record with its recorded digest."""
    entries = {cases["book_cases"][k]["file"]: cases["book_cases"][k]["sha256"] for k in ("validation", "heldout")}
    entries.update({cases[k]["file"]: cases[k]["sha256"] for k in CASE_KEYS})
    return entries


def inputs() -> list[str]:
    base = sorted(str(q.relative_to(ROOT)) for q in (ROOT / "work/qwen35_08b/base").iterdir() if q.is_file())
    robust = json.loads((OUT / "robust_record.json").read_text())
    cases = json.loads((OUT / "cases_record.json").read_text())
    return ["work/qat08_mc12/data/train_MC12.u32", "work/qat08_mc12/data/stream_record.json",
            "results/qat08_mc12/cases_record.json", "results/qat08_mc12/robust_record.json",
            *[f"results/qat08_mc12/{f}" for f in case_entries(cases)],
            *[f"results/qat08_mc12/{e['file']}" for e in robust["files"].values()],
            "results/qat08_mcu/protocol.json", "results/qat08_mcu/design.json", "results/qat08_mcu/results.json",
            "results/qat08_mcu/export_M-final.json", "results/qat08_mcu/cases_record.json",
            *[f"work/qat08_mcu/data/{n}" for n in ("mc_prompts.jsonl", "mc_record.json", "mc_teacher.jsonl", "mc_unique.u32",
                                                  "kept_mc.json", "data_record.json", "teacher_timing.json", "train_M.u32")],
            "work/qat08_chat/data/train.u32", "work/qat08/data/train.u32", "work/qat08/data/monitor.u32",
            "work/qwen35_08b/folded-pq2.gguf", "work/qwen35_08b/folded/hadamard_packing.json",
            "results/diag_readout/plan.json", "results/diag_readout/cases/letter.tsv", "results/diag_readout/cases/binding.tsv",
            "results/diag_readout/results.json", "results/diag_readout/decision.json",
            *[f"work/diag_readout/scores/{a}-{k}.jsonl" for a in ("fp", "qat_chat_42-final") for k in ("letter", "binding")],
            *base]


def check_runtime_matches_mcu() -> dict:
    mcu = json.loads((MCU_OUT / "protocol.json").read_text())
    runtime = {"../../../bin/cuda/" + q.name: sha(q) for q in sorted((ROOT.parents[2] / "bin/cuda").glob("*.so*")) if q.is_file()}
    assert runtime == mcu["runtime_sha256"], "GPU runtime differs from the MCU run"
    for n in ("work/qwen3_17b/score_17b", "../../../bin/cuda/llama-server"):
        assert sha(ROOT / n) == mcu["implementation_sha256"][n], n
    return runtime


def design() -> dict:
    if DESIGN.exists() or PROTOCOL.exists():
        raise SystemExit("MC12 design already frozen; write a versioned amendment instead")
    assert sha(MCU_OUT / "protocol.json") == MCU_PROTOCOL_SHA
    mcu = json.loads((MCU_OUT / "protocol.json").read_text())
    for n, d in mcu["implementation_sha256"].items():
        assert sha(ROOT / n) == d, f"frozen MCU dependency changed: {n}"
    check_mcu_data()
    runtime = check_runtime_matches_mcu()
    stream_rec = json.loads((DATA / "stream_record.json").read_text())
    assert sha(STREAM) == stream_rec["stream_sha256"]
    for name, digest in stream_rec["sources_sha256"].items():
        assert sha(ROOT / name) == digest, f"stream source changed since the stream was built: {name}"
    cases = json.loads((OUT / "cases_record.json").read_text())
    for name, digest in case_entries(cases).items():
        assert sha(OUT / name) == digest, f"evaluation file changed: {name}"
    for name in inputs():  # an input the sealed MCU protocol also pinned must still match MCU's own record
        if name in mcu["input_sha256"]:
            assert sha(ROOT / name) == mcu["input_sha256"][name], f"input differs from the sealed MCU record: {name}"
    audit = stream_audit()
    incidents = WORK / "pre_freeze_incidents.json"
    d = {
        "date": "2026-10-06", "role": "QAT08-MC12: prompt robustness of M, then one arm with 12.5% multiple-choice data at "
                                       "M's recipe and 65.536M tokens",
        "spec": "user instruction 'Proceed with the recommendation' (parent thread thr_46yhdt5qhc, phist_f8bxskdn7s), "
                "answering the parent's recommendation of a prompt-robustness check then one 12.5% MC arm",
        "arms": ARMS, "training": mcu["training"],
        "training_note": "identical to the sealed MCU training block; fp_peak_lr is unused because MC12 keeps FP tensors "
                         "frozen with M's BF16 load policy. The worker pins seed 20261007 (M's start-up RNG was not "
                         "recorded, so seed parity with M is not claimed; data order, folded start and recipe are identical)",
        "comparators": comparators(), "reused_outputs": reused_outputs(),
        "stream": stream_rec, "cases": cases, "robustness": {**ROBUSTNESS, "record": json.loads((OUT / "robust_record.json").read_text())},
        "pre_freeze_stream_audit": audit, "pre_freeze_incidents": json.loads(incidents.read_text()) if incidents.exists() else [],
        "decision": DECISION, "budget": BUDGET, "licenses": LICENSES,
        "stages": {"robust": ["robust-score", "robust-analyze"],
                   "run": ["train-MC12", *[f"export-MC12-{c}" for c in ("init", "t8m", "t16m", "t33m", "final")],
                           "curve", "heldout", "gsm8k", "analyze", "report"]},
        "implementation_sha256": {n: sha(ROOT / n) for n in (*NEW_FILES, *REUSED_CODE)},
        "input_sha256": {n: sha(ROOT / n) for n in inputs()},
        "runtime_sha256": runtime, "environment": fm.current_environment(),
        "approval": "GPU stages need protocol_approval.json bound to design and protocol: the user's pre-hash scoped "
                    "instruction plus the parent's independent design review; the user has not reviewed this hash",
    }
    write_new(DESIGN, d)
    write_new(PROTOCOL, {**d, "design_sha256": sha(DESIGN)})
    print("froze MC12 design", sha(DESIGN), "protocol", sha(PROTOCOL), flush=True)
    return d


def verify(root: Path = ROOT, out: Path = OUT, check_env: bool = True) -> dict:
    """Re-check every frozen hash; called by the controller at launch, resume and before every stage."""
    p = json.loads((out / "protocol.json").read_text())
    assert sha(out / "design.json") == p["design_sha256"], "design.json changed"
    for group in ("implementation_sha256", "input_sha256", "runtime_sha256"):
        for name, digest in p[group].items():
            assert sha(root / name) == digest, f"frozen MC12 dependency changed: {name}"
    for name, entry in p["comparators"].items():
        assert sha(root / entry["file"]) == entry["sha256"], name
    for group in ("scores", "gsm8k"):
        for name, digest in p["reused_outputs"][group].items():
            assert sha(root / name) == digest, f"reused output changed: {name}"
    for name, digest in case_entries(p["cases"]).items():
        assert sha(out / name) == digest, f"evaluation file changed: {name}"
    for entry in p["robustness"]["record"]["files"].values():
        assert sha(out / entry["file"]) == entry["sha256"], f"robustness case file changed: {entry['file']}"
    assert sha(root / p["stream"]["stream"]) == p["stream"]["stream_sha256"], "MC12 stream changed"
    if check_env:
        fm.check_environment(p["environment"])
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["design", "verify"])
    a = ap.parse_args()
    if a.cmd == "design":
        design()
    else:
        verify()
        print("MC12 protocol verified", sha(PROTOCOL))

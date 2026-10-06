"""Freeze QAT08-MCU. `design`: after cases, MC prompts, teacher pilot and FP LR probe, before full generation.
`protocol`: after generation and packing, before any training. Both exclusive; protocol only adds data hashes,
the per-stream contamination audit and timings. Training additionally needs the user's protocol_approval.json.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np

from freeze_qat08 import ngram_hashes
from prepare_qat08_mcu import CHAT_OUT, DATA, OUT, ROOT, VALIDATION, WORK, pre_freeze_incidents, read_cases, write_new
from score_17b import sha

DIAG = ROOT / "results/diag_readout"
PROBE_LRS = (0.0, 1e-5, 1e-4, 1e-3)
ARM_SPECS = {
    "M": {"rule": "top86", "stream": "work/qat08_mcu/data/train_M.u32", "unfreeze": False, "init": "folded", "schedule": "cosine"},
    "U": {"rule": "top86", "stream": "work/qat08_chat/data/train.u32", "unfreeze": True, "init": "folded", "schedule": "cosine",
          "fp_init": "checkpoint_f32"},
    "MU": {"rule": "top86", "stream": "work/qat08_mcu/data/train_M.u32", "unfreeze": True, "init": "folded", "schedule": "cosine",
           "fp_init": "checkpoint_f32"},
    "B": {"rule": "top86", "stream": "work/qat08_mcu/data/train_B.u32", "unfreeze": False,
          "init": "work/qat08_chat/qat_chat_42/resume.pt", "init_step": 2000, "schedule": "constant", "lr": 1e-5},
}
COMPARATORS = {"fp": "work/qwen35_08b/base-f32.gguf", "qat_42": "work/qat08/qat_42-final.gguf",
               "C": "work/qat08_chat/qat_chat_42-final.gguf"}
DEVIATIONS = [
    "the diagnostics decision is read through results/diag_readout/decision_amendment.json, written after the results were seen: the frozen FP binding floor (calibrated accuracy, 94.0%) flagged harness_invalid only because mean-log-prob calibration flips low-margin items of a near-perfect model (uncalibrated 99.25%); the same 95% floor now applies to uncalibrated accuracy, giving readout_broken (M, U, B; optional MU); the original decision.json is kept and hash-bound",
    "OpenBookQA card licence is unknown: closed-book MC training uses ARC train (CC BY-SA 4.0); the fresh knowledge set is 3,000 MMLU test items outside MMLU-Redux 2.0 (any error_type) instead of ARC test",
    "protocol requires a separate explicit user approval (protocol_approval.json); the controller never writes it",
    "arm B continues C's saved optimizer state at constant LR 1e-5 for 2,000 steps on C's layout from the next unseen data",
    "measured while planning: transformers loads the 18 GDN norms in BF16 even with dtype=float32, so the frozen QAT08/chat students (and arms M, B, which keep that recipe) train on BF16-rounded copies while their GGUFs keep exact F32 values; arms U/MU start from and export the exact F32 checkpoint values (fp_init=checkpoint_f32)",
    "U/MU clip gradients jointly over the projections and the FP tensors (C, M and B clip projections only)",
    "U/MU export the 36 trained in_proj_a/in_proj_b gates in their BF16 template storage, a small rounding of the trained FP32 values",
    "total ceiling is 25 GPU-h (the review's 13.5-20 h estimate covered training only; prep, data and evaluation phases are included here)",
    "in-context MC items appear in a single option order with exactly balanced gold letters; ARC items appear in two orders",
    "item mix is about 48% in-context / 52% ARC by count before screening (the review proposed about 60/40)",
    "teacher limit is 512 tokens (letter-only answers plus 20% short-reasoning answers) instead of the review's 1,536",
]
DECISION = {
    "effect_points": 4.0, "ladder_fresh_pairwise_lo_above": 52.0, "fresh_pairwise_lo_above": 50.0, "binding_min": 90.0, "book_gain": -0.05,
    "recovery": {"mmlu_lo": 30.0, "gsm8k_lo": 10.0}, "fresh_recovery_lo": 30.0,
    "costs": {"book_hi": 0.10, "gsm8k_lo": -3.0, "retrieval_min": 90.0},
    "format_only": {"letter_mass_min": 0.5, "pairwise_below": 52.0},
    "text": {
        "H_M": "M (or MU) minus C MMLU-Redux calibrated pairwise >= +4 points with paired lower bound > 0, AND arm MMLU-fresh pairwise lower bound > 50, AND binding calibrated accuracy >= 90",
        "H_U": "U (or MU) minus C book NLL mean <= -0.05 with upper bound < 0, OR MMLU-Redux pairwise effect as above",
        "H_B": "B minus C book NLL mean <= -0.05 with upper bound < 0, OR MMLU-Redux pairwise effect as above (budget effect)",
        "recovered": "MMLU-Redux Wilson lower > 30 AND GSM8K Wilson lower > 10 AND GSM8K natural-EOS Wilson lower > 10 (unchanged QAT08_CHAT gate)",
        "fresh_confirmed": "MMLU-fresh Wilson lower > 30",
        "costs_ok": "book minus C upper bound <= +0.10 AND GSM8K minus C lower bound >= -3 points AND retrieval accuracy >= 90",
        "format_only": "letter mass >= 0.5 and MMLU-Redux pairwise < 52: report as format learning, never as recovery",
        "recommendation": "rent_1p7b_with_recipe if any arm is recovered AND fresh_confirmed AND costs_ok; else iterate_0p8b_mc_data if M or MU shows the pairwise effect; else rent_1p7b_token_ladder if the diagnostics branch was knowledge_lost AND B shows a knowledge signal (MMLU-Redux pairwise effect, or MMLU-fresh pairwise lower bound > 52) AND U/MU show no pairwise effect (a book gain alone never triggers rental); else iterate_0p8b if any arm shows an effect or book gain; else stop_data_iteration_0p8b",
    },
}
BUDGET = {"total_gpu_hours": 25.0, "prep_gpu_hours": 2.0, "data_gpu_hours": 1.0, "per_arm_train_hours": 6.5,
          "per_arm_eval_hours": 0.15, "fixed_eval_hours": 0.5}


def select_arms(decision: dict, prep_hours: float) -> list[str]:
    """Required arms always; each optional arm only if the projected total stays within the 25 GPU-hour ceiling."""
    arms = list(decision["arms_required"])
    per_arm = BUDGET["per_arm_train_hours"] + BUDGET["per_arm_eval_hours"]
    for extra in decision["arms_optional"]:
        projected = prep_hours + BUDGET["data_gpu_hours"] + (len(arms) + 1) * per_arm + BUDGET["fixed_eval_hours"]
        if projected <= BUDGET["total_gpu_hours"]:
            arms.append(extra)
    return arms


def choose_fp_lr(records: list[dict]) -> dict:
    final = {r["fp_lr"]: r["curve"][-1]["monitor_kl"] for r in records}
    candidates = {lr: kl for lr, kl in final.items() if lr > 0}
    selected = min(candidates, key=lambda lr: (candidates[lr], lr))
    return {"selected_lr": selected, "final_monitor_kl": final, "baseline_frozen_kl": final[0.0],
            "unfreeze_helps_in_probe": candidates[selected] < final[0.0],
            "rule": "lowest final monitor KL after 120 probe steps among 1e-5, 1e-4, 1e-3; frozen-FP baseline recorded"}


def _prep_hours() -> float:
    import qat08_mcu_control as control
    return control._used(phase="prep")


def check_runtime_matches_chat() -> None:
    """Fresh arms must be scored with the runtime that produced the reused comparator outputs."""
    import diag_cases as dc
    chat = json.loads((CHAT_OUT / "protocol.json").read_text())
    assert dc.runtime_hashes() == chat["runtime_sha256"], "bin/cuda libraries differ from the chat run's"
    assert sha(ROOT / "work/qwen3_17b/score_17b") == chat["implementation_sha256"]["work/qwen3_17b/score_17b"], "scorer changed"
    assert sha(ROOT / "../../../bin/cuda/llama-server") == chat["input_sha256"]["../../../bin/cuda/llama-server"], "llama-server changed"


PACKAGES = ("torch", "transformers", "flash-linear-attention", "triton", "numpy", "scipy", "pyarrow", "scikit-learn")


def current_environment() -> dict:
    return {"python": platform.python_version(), "packages": {n: importlib.metadata.version(n) for n in PACKAGES}}


def check_environment(env: dict) -> None:
    now = current_environment()
    assert env["python"] == now["python"], "python version changed"
    for name, version in env["packages"].items():
        assert now["packages"].get(name) == version, f"frozen package version changed: {name}"


IMPLEMENTATION = ("prepare_qat08_mcu.py", "freeze_qat08_mcu.py", "qat_08b_mcu.py", "score_qat08_mcu.py",
                  "qat08_mcu_control.py", "report_qat08_mcu.py", "test_qat08_mcu.py", "diag_cases.py", "diag_metrics.py",
                  "diag_run.py", "diag_amendment.py", "qat_08b.py", "qat08_chat_runtime.py", "prepare_qat08_chat.py", "freeze_qat08.py",
                  "freeze_17b.py", "score_17b.py", "score_17b.cpp", "gsm8k_17b.py", "make_gptq_pilot.py", "phase0.py",
                  "work/qwen3_17b/score_17b", "../../../bin/cuda/llama-server")


def design() -> None:
    if (OUT / "design.json").exists():
        raise SystemExit("design already frozen; write a versioned amendment instead")
    from qat08_mcu_control import diagnostics
    decision, provenance = diagnostics(DIAG)
    assert decision["stop"] is False, "the diagnostics stopped the experiment"
    check_runtime_matches_chat()
    chat = json.loads((CHAT_OUT / "protocol.json").read_text())
    cases = json.loads((OUT / "cases_record.json").read_text())
    mc = json.loads((DATA / "mc_record.json").read_text())
    arms = select_arms(decision, _prep_hours())
    probe = None
    if any(ARM_SPECS[a]["unfreeze"] for a in arms):
        probe = choose_fp_lr([json.loads((WORK / "probe" / f"fp_lr{lr:g}.json").read_text()) for lr in PROBE_LRS])
    training = {**chat["training"], "fp_peak_lr": probe["selected_lr"] if probe else None}
    inputs = ["results/qat08_chat/protocol.json", "results/qat08_chat/results.json", "results/diag_readout/plan.json",
              "results/diag_readout/results.json", "results/diag_readout/decision.json",
              *(["results/diag_readout/decision_amendment.json"] if provenance["amendment_sha256"] else []),
              "work/qat08_chat/data/train.u32", "work/qat08_chat/data/chat_unique.u32", "work/qat08_chat/qat_chat_42/resume.pt",
              "work/qat08/data/train.u32", "work/qat08/data/monitor.u32", "work/qwen35_08b/folded-pq2.gguf",
              "work/qwen35_08b/folded/hadamard_packing.json", *COMPARATORS.values()]
    inputs += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "work/qwen35_08b/base").iterdir()) if p.is_file()]
    p = {
        "date": "2026-10-03",
        "role": "QAT08-MCU: multiple-choice read-out data x trained FP non-projection tensors (x continued training) at 0.8B",
        "spec": "reviews/REVIEW_QAT08_CHAT_opus55.md sections 6-7; docs/superpowers/plans/2026-10-03-qat08-mcu-experiment.md",
        "deviations": DEVIATIONS,
        "diagnostics": {"branch": decision["branch"], **provenance, "amended": decision.get("amended", False),
                        "arms_required": decision["arms_required"], "arms_optional": decision["arms_optional"]},
        "arms": {a: {**ARM_SPECS[a], "steps": training["steps"]} for a in arms},
        "comparators": {a: {"file": f, "sha256": sha(ROOT / f)} for a, f in COMPARATORS.items()},
        "reused_comparator_outputs": "fp, qat_42 and C MMLU-Redux and retrieval scores and GSM8K generations are copied from the chat run after hash checks against results/qat08_chat/results.json; every other set is scored fresh",
        "training": training, "fp_lr_probe": probe,
        "unchanged": "quantizer, rule top86, H512 basis, folded init, fixed ternary embedding, teacher, all-position forward KL, optimizer, schedule, clip threshold, checkpoints, monitor; only the per-arm keys in arms differ (see deviations for joint clipping in U/MU)",
        "data": {"mc_record": mc, "mc_tokens": 4_096_000, "fineweb_tokens_M": 45_056_000, "chat_tokens": 16_384_000,
                 "filters": "keep naturally terminated, non-empty, no special/thinking tokens, no 13-word gram with eval content, transcript <= 1024 tokens; require >=5000 kept, >=1M unique tokens, <=4 passes; else stop"},
        "cases": cases, "splits": {"validation_books": list(VALIDATION),
                                   "heldout_books": [n for n in cases["titles"] if n not in VALIDATION]},
        "pre_freeze_incidents": pre_freeze_incidents(),
        "decision": DECISION, "budget": BUDGET, "prep_gpu_hours": _prep_hours(),
        "implementation_sha256": {n: sha(ROOT / n) for n in IMPLEMENTATION},
        "input_sha256": {n: sha(ROOT / n) for n in inputs},
        "runtime_sha256": {"../../../bin/cuda/" + q.name: sha(q) for q in sorted((ROOT.parents[2] / "bin/cuda").glob("*.so*")) if q.is_file()},
        "environment": current_environment(),
        "approval": "data phase needs design_approval.json bound to this file; training needs protocol_approval.json bound to design and protocol, written only after the user reviews the sealed protocol summary",
    }
    write_new(OUT / "design.json", p)
    print("froze design", sha(OUT / "design.json"), "arms", arms, flush=True)


def verify_design() -> dict:
    d = json.loads((OUT / "design.json").read_text())
    for group in ("implementation_sha256", "input_sha256", "runtime_sha256"):
        for name, digest in d[group].items():
            assert sha(ROOT / name) == digest, f"frozen design dependency changed: {name}"
    for arm, entry in d["comparators"].items():
        assert sha(ROOT / entry["file"]) == entry["sha256"], arm
    check_environment(d["environment"])
    read_cases()
    return d


def contamination(stream: Path) -> dict:
    seen = np.unique(ngram_hashes(np.memmap(stream, dtype=np.uint32, mode="r")))
    out, path = {}, OUT / f"contamination_{stream.stem}.jsonl"
    tmp = path.with_suffix(".partial")  # unsealed intermediate; protocol.json is the exclusive seal
    with tmp.open("w") as f:
        for key, rows in read_cases().items():
            fractions = []
            for case_id, ids in rows:
                hashes = ngram_hashes(ids) if len(ids) >= 13 else np.asarray([], dtype=np.uint64)
                hits = int((seen[np.minimum(np.searchsorted(seen, hashes), len(seen) - 1)] == hashes).sum()) if len(hashes) else 0
                fraction = hits / len(hashes) if len(hashes) else 0.0
                fractions.append(fraction)
                f.write(json.dumps({"set": key, "id": case_id, "ngrams": len(hashes), "hits": hits, "fraction": fraction}) + "\n")
            out[key] = {"cases": len(rows), "cases_with_any_13gram_overlap": sum(v > 0 for v in fractions),
                        "cases_with_majority_13gram_overlap": sum(v > .5 for v in fractions),
                        "maximum_overlap_fraction": max(fractions), "mean_overlap_fraction": float(np.mean(fractions))}
    tmp.replace(path)
    return {"sets": out, "file": path.name, "sha256": sha(path)}


def protocol() -> None:
    if (OUT / "protocol.json").exists():
        raise SystemExit("protocol already frozen; write a versioned amendment instead")
    d = verify_design()
    data = json.loads((DATA / "data_record.json").read_text())
    streams = sorted({a["stream"] for a in d["arms"].values()})
    audit = {s: contamination(ROOT / s) for s in streams}
    assert all(v["cases_with_majority_13gram_overlap"] == 0 for a in audit.values() for v in a["sets"].values()), audit
    timing = json.loads((DATA / "teacher_timing.json").read_text())
    used = d["prep_gpu_hours"] + timing["seconds"] / 3600
    projected = used + len(d["arms"]) * (BUDGET["per_arm_train_hours"] + BUDGET["per_arm_eval_hours"]) + BUDGET["fixed_eval_hours"]
    assert projected <= BUDGET["total_gpu_hours"], f"projected {projected:.1f} GPU-h exceeds the ceiling; stop before training"
    data_files = ["mc_prompts.jsonl", "mc_record.json", "mc_teacher.jsonl", "mc_unique.u32", "kept_mc.json", "data_record.json",
                  "teacher_timing.json", *data["streams"]]
    write_new(OUT / "protocol.json", {**d, "design_sha256": sha(OUT / "design.json"), "data_record": data,
              "data_sha256": {n: sha(DATA / n) for n in data_files}, "contamination_13gram": audit,
              "teacher_timing": timing, "projected_total_gpu_hours": projected})
    print("froze protocol", sha(OUT / "protocol.json"), flush=True)
    print(json.dumps({"arms": list(d["arms"]), "fp_peak_lr": d["training"]["fp_peak_lr"],
                      "mc": {k: data[k] for k in ("mc_kept", "mc_unique_tokens", "mc_passes", "teacher_letter_accuracy")},
                      "contamination": {s: a["sets"] for s, a in audit.items()}, "projected_gpu_hours": projected}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["design", "protocol", "verify-design"])
    a = ap.parse_args()
    {"design": design, "protocol": protocol, "verify-design": verify_design}[a.stage]()

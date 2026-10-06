"""CPU tests for the post-results diagnostics amendment. Run: CUDA_VISIBLE_DEVICES= python test_diag_amendment.py"""
from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

RULE = {"fp_binding_min": 95.0, "fp_letter_pairwise_min": 60.0, "fp_cloze_pairwise_lo_min": 55.0,
        "readout_broken_below": 70.0, "readout_intact_at_least": 90.0, "knowledge_pairwise_lo_above": 52.0,
        "trajectory_rise_points": 2.0}


def _results(fp_raw=99.25, fp_cal=94.0, chat_bind=27.2):
    def arm(bind_raw, bind_cal, letter, cloze_lo):
        return {"binding": {"raw_accuracy": bind_raw, "calibrated_accuracy": bind_cal, "pairwise": 98.0, "pairwise_lo": 97.2},
                "letter": {"pairwise": letter}, "cloze_chat": {"pairwise_lo": cloze_lo}, "cloze_raw": {"pairwise_lo": cloze_lo + 0.7}}
    flat = {"mean": 0.5, "lo": -1.0}
    return {"arms": {"fp": arm(fp_raw, fp_cal, 65.7, 55.0), "qat_chat_42-final": arm(25.5, chat_bind, 51.3, 50.3)},
            "trajectory": {"qat_chat_42": {"final_minus_t8m": {"letter": flat, "cloze_raw": flat}}}}


def test_amend_uses_uncalibrated_fp_floor():
    from diag_amendment import amend
    from diag_metrics import branch
    r = _results()
    assert branch(r, RULE)["branch"] == "harness_invalid"
    out = amend(r, {"rule": RULE})
    assert out["branch"] == "readout_broken" and out["arms_required"] == ["M", "U"] and out["stop"] is False
    assert out["reasons"][0].startswith("FP binding uncalibrated accuracy 99.25%")
    low = amend(_results(fp_raw=93.0), {"rule": RULE})
    assert low["branch"] == "harness_invalid" and low["stop"] is True
    letter_fail = _results()
    letter_fail["arms"]["fp"]["letter"]["pairwise"] = 55.0
    assert amend(letter_fail, {"rule": RULE})["branch"] == "harness_invalid", "the letter floor is unchanged"


def test_write_is_exclusive_and_hash_bound():
    import diag_amendment as d
    from score_17b import sha
    with tempfile.TemporaryDirectory() as name:
        out = Path(name)
        (out / "plan.json").write_text("{}")
        r = _results()
        (out / "results.json").write_text(json.dumps(r))
        (out / "decision.json").write_text(json.dumps({"branch": "harness_invalid", "stop": True, "rule": RULE,
                                                       "results_sha256": sha(out / "results.json"),
                                                       "plan_sha256": sha(out / "plan.json")}))
        d.write(out, "test instruction")
        a = json.loads((out / "decision_amendment.json").read_text())
        assert a["amends_decision_sha256"] == sha(out / "decision.json")
        assert a["results_sha256"] == sha(out / "results.json") and a["written_after_results"] is True
        assert a["original_branch"] == "harness_invalid" and a["amended"]["branch"] == "readout_broken"
        assert a["user_instruction"] == "test instruction"
        try:
            d.write(out, "again")
        except FileExistsError:
            pass
        else:
            raise AssertionError("amendment must be exclusive")


def test_real_results_amend_to_readout_broken():
    from diag_amendment import amend
    out = Path(__file__).resolve().parent / "results/diag_readout"
    if not (out / "results.json").exists():
        return
    a = amend(json.loads((out / "results.json").read_text()), json.loads((out / "decision.json").read_text()))
    assert a["branch"] == "readout_broken" and a["arms_required"] == ["M", "U", "B"] and a["arms_optional"] == ["MU"]


TESTS = [f for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print("PASS", t.__name__, flush=True)

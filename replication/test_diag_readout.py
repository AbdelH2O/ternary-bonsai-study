"""CPU tests for the read-out diagnostics. Run: CUDA_VISIBLE_DEVICES= python test_diag_readout.py [test_name ...]"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _rows(path):
    return {r["id"]: r["values"] for r in map(json.loads, Path(path).read_text().splitlines())}


def test_wilson_matches_existing():
    import diag_metrics as m
    from prepare_qat08_chat import wilson
    assert m.wilson(192, 1319) == wilson(192, 1319)


def test_pairwise_ties_are_half():
    import diag_metrics as m
    assert m.item_pairwise([0.0, 0.0, 0.0, 0.0], 2) == 0.5
    rows = {f"s/{i}/{'ABCD'[i % 4]}": [-1.0, -1.0, -1.0, -1.0] for i in range(40)}
    r = m.letter_metrics(rows)
    assert abs(r["pairwise"] - 50.0) < 1e-9
    assert r["raw_accuracy"] == 0.0  # strict rule: gold must beat every wrong letter


def test_calibration_removes_constant_bias():
    import diag_metrics as m
    rows = {}
    for i in range(80):
        gold = i % 4
        v = [-3.0, -3.0, -3.0, -3.0]
        v[0] += 2.0          # strong constant preference for A
        v[gold] += 1.0       # real but smaller signal for gold
        rows[f"s/{i}/{'ABCD'[gold]}"] = v
    r = m.letter_metrics(rows)
    assert r["raw_accuracy"] == 25.0
    assert r["calibrated_accuracy"] == 100.0
    assert r["pairwise"] == 100.0


def test_cloze_length_normalized():
    import diag_metrics as m
    rows = {
        "s/0/B/0": [2.0], "s/0/B/1": [1.0, 1.0, 1.0], "s/0/B/2": [3.0], "s/0/B/3": [2.5, 2.5],
        "s/1/A/0": [0.5, 0.5], "s/1/A/1": [1.0], "s/1/A/2": [2.0], "s/1/A/3": [0.9],
    }
    r = m.cloze_metrics(rows)
    assert r["items"] == 2 and r["accuracy"] == 100.0 and r["pairwise"] == 100.0


def test_parse_mc_answer():
    import diag_metrics as m
    assert m.parse_mc_answer("so the result follows.\nAnswer: C") == "C"
    assert m.parse_mc_answer("Thus the answer is (B).") == "B"
    assert m.parse_mc_answer("**Answer: D**") == "D"
    assert m.parse_mc_answer("Answer: **A**") == "A"
    assert m.parse_mc_answer("Answer: B ... wait, Answer: D") == "D"
    assert m.parse_mc_answer("the answer is a dog") is None
    assert m.parse_mc_answer("no letter here") is None


def test_parse_mc_answer_markdown_forms():
    import diag_metrics as m
    assert m.parse_mc_answer("**Answer:** C") == "C"
    assert m.parse_mc_answer("Answer: **(D)**") == "D"
    assert m.parse_mc_answer("The correct answer is: **(C)**") == "C"
    assert m.parse_mc_answer("so we get \\boxed{C}") == "C"
    assert m.parse_mc_answer("Answer: A.") == "A"
    assert m.parse_mc_answer("The answer is A dog") is None
    assert m.parse_mc_answer("Answer: B\nthen \\boxed{D}") == "D"


def test_regions():
    import diag_metrics as m
    assert m.regions(20, 3, 9) == {"question": (2, 10), "template": (10, 19), "letter": (19, 20)}


def test_branch_rule():
    import diag_metrics as m
    rule = {"fp_binding_min": 95.0, "fp_letter_pairwise_min": 60.0, "readout_broken_below": 70.0,
            "readout_intact_at_least": 90.0, "knowledge_pairwise_lo_above": 52.0, "trajectory_rise_points": 2.0}
    rule["fp_cloze_pairwise_lo_min"] = 55.0
    def res(fp_bind=99.0, fp_pw=68.0, bind=95.0, cc_lo=50.0, cr_lo=50.0, rise=(0.0, -1.0), fp_cloze_lo=60.0):
        traj = {"mean": rise[0], "lo": rise[1], "hi": rise[0] + 1, "n": 100}
        return {"arms": {"fp": {"binding": {"calibrated_accuracy": fp_bind}, "letter": {"pairwise": fp_pw},
                                "cloze_chat": {"pairwise_lo": fp_cloze_lo}, "cloze_raw": {"pairwise_lo": fp_cloze_lo}},
                         "qat_chat_42-final": {"binding": {"calibrated_accuracy": bind},
                                               "cloze_chat": {"pairwise_lo": cc_lo}, "cloze_raw": {"pairwise_lo": cr_lo}}},
                "trajectory": {"qat_chat_42": {"final_minus_t8m": {"letter": traj, "cloze_raw": traj}}}}
    assert m.branch(res(fp_bind=80.0), rule)["branch"] == "harness_invalid"
    assert m.branch(res(fp_bind=80.0), rule)["stop"] is True
    b = m.branch(res(bind=40.0), rule)
    assert b["branch"] == "readout_broken" and b["arms_required"] == ["M", "U"] and b["arms_optional"] == ["MU"]
    b = m.branch(res(bind=95.0), rule)
    assert b["branch"] == "knowledge_lost" and b["arms_required"] == ["U", "B"]
    b = m.branch(res(bind=95.0, cr_lo=55.0), rule)
    assert b["branch"] == "closed_book_readout" and b["arms_required"] == ["M", "U"]
    b = m.branch(res(bind=80.0), rule)
    assert b["branch"] == "ambiguous" and b["arms_required"] == ["M", "U"]
    b = m.branch(res(bind=40.0, rise=(3.0, 0.5)), rule)
    assert b["arms_required"] == ["M", "U", "B"]
    b = m.branch(res(bind=95.0, fp_cloze_lo=51.0), rule)  # FP itself shows no cloze signal: cloze is uninformative
    assert b["branch"] == "ambiguous" and b["arms_required"] == ["M", "U"]


def test_review_numbers_reproduce():
    import diag_metrics as m
    chat = m.letter_metrics(_rows(ROOT / "work/qat08_chat/scores/qat_chat_42-mmlu.jsonl"))
    fp = m.letter_metrics(_rows(ROOT / "work/qat08_chat/scores/fp-mmlu.jsonl"))
    # Item-level pairwise (mean over items of the share of wrong letters beaten). The review quoted 50.3
    # [49.6, 51.0] from a per-letter average with a bootstrap; this frozen definition gives 50.36 [49.35, 51.38].
    assert abs(chat["raw_accuracy"] - 23.8649) < 1e-3
    assert abs(chat["calibrated_accuracy"] - 25.966) < 0.01
    assert abs(chat["pairwise"] - 50.363) < 0.01
    assert abs(chat["pairwise_lo"] - 49.350) < 0.01 and abs(chat["pairwise_hi"] - 51.375) < 0.01
    assert abs(fp["pairwise"] - 68.649) < 0.01
    assert abs(chat["letter_mass"] - 0.7795) < 1e-3


def _tok():
    from prepare_qat08_chat import tokenizer
    return tokenizer()


def test_case_suffix_and_letters():
    import diag_cases as c
    tok = _tok()
    item = {"subject": "high_school_biology", "question": "Which organelle makes ATP?",
            "choices": ["Nucleus", "Mitochondrion", "Ribosome", "Golgi body"], "answer": 1}
    cid, kind, ids, arg = c.letter_case(tok, 7, item)
    assert cid == "high_school_biology/7/B" and kind == "choice" and arg == "32,33,34,35"
    assert ids[:3] == c.PREFIX and ids[-9:] == c.SUFFIX
    assert [tok.encode(x, add_special_tokens=False) for x in "ABCD"] == [[32], [33], [34], [35]]
    for raw in (False, True):
        rows = c.cloze_cases(tok, 7, item, raw)
        assert [r[0] for r in rows] == [f"high_school_biology/7/B/{k}" for k in range(4)]
        assert all(r[1] == "nll" and 1 <= int(r[3]) < len(r[2]) for r in rows)
        prefix = rows[0][2][:int(rows[0][3])]
        assert all(r[2][:int(r[3])] == prefix for r in rows)  # identical context, only the option differs
    assert c.cloze_cases(tok, 7, item, False)[0][2][:int(c.cloze_cases(tok, 7, item, False)[0][3])][-9:] == c.SUFFIX


def test_keep_item_excludes_banned():
    import diag_cases as c
    from prepare_qat08_chat import normalize
    banned = {normalize("Which organelle makes ATP?")}
    good = {"question": "Which planet is largest?", "choices": ["Mars", "Jupiter", "Venus", "Earth"], "answer": 1}
    assert c.keep_item(good, banned)
    assert not c.keep_item({**good, "question": "Which  organelle makes ATP ?"}, banned)
    assert not c.keep_item({**good, "choices": ["a", "b", "c"]}, banned)
    assert not c.keep_item({**good, "choices": ["a", "b", " ", "d"]}, banned)


def test_binding_cases_balanced_and_answerable():
    import collections, re
    import diag_cases as c
    tok = _tok()
    rows = c.binding_cases(tok, seed=123, n=40, entries=8)
    gold = collections.Counter(r[0].rsplit("/", 1)[1] for r in rows)
    assert gold == {"A": 10, "B": 10, "C": 10, "D": 10}
    queried_positions = set()
    for cid, kind, ids, arg in rows:
        assert kind == "choice" and arg == "32,33,34,35" and ids[-9:] == c.SUFFIX and len(ids) < 2000
        text = tok.decode(ids)
        registry = dict(re.findall(r"^([A-Z][a-z]+ [A-Z][a-z]+): (\d{4})$", text, re.M))
        name = re.search(r"What is the access code for (.+)\?", text)[1]
        options = dict(re.findall(r"^([ABCD])\. (\d{4})\b", text, re.M))  # D is followed by <|im_end|>
        assert sorted(options) == ["A", "B", "C", "D"]
        assert options[cid.rsplit("/", 1)[1]] == registry[name]
        assert list(options.values()).count(registry[name]) == 1
        queried_positions.add(list(registry).index(name))
    assert len(queried_positions) > 4  # query position is not tied to the gold letter


def test_runtime_hashes_match_chat_protocol():
    import diag_cases as c
    hashes = c.runtime_hashes()
    assert any(k.endswith("libggml-cuda.so") for k in hashes) and any(k.endswith("libllama.so") for k in hashes)
    chat = json.loads((ROOT / "results/qat08_chat/protocol.json").read_text())["runtime_sha256"]
    assert hashes == chat


def test_position_kl_chunked_matches_reference():
    import torch
    import diag_run as r
    torch.manual_seed(0)
    hs, ht = torch.randn(37, 16), torch.randn(37, 16)
    es, et = torch.randn(300, 16), torch.randn(300, 16)
    per, letter4 = r.position_kl(hs, ht, es, et, [3, 7, 11, 13], chunk=8)
    ls, lt = (hs @ es.T).log_softmax(-1), (ht @ et.T).log_softmax(-1)
    ref = (lt.exp() * (lt - ls)).sum(-1)
    assert per.shape == (37,) and torch.allclose(per, ref, atol=1e-5)
    s4, t4 = ls[-1, [3, 7, 11, 13]].log_softmax(-1), lt[-1, [3, 7, 11, 13]].log_softmax(-1)
    assert abs(letter4 - (t4.exp() * (t4 - s4)).sum().item()) < 1e-5


def test_checked_rows_rejects_mismatch():
    import tempfile
    import diag_run as r
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.jsonl"
        p.write_text('{"id":"a","values":[1.0]}\n{"id":"b","values":[2.0]}\n')
        assert r.checked_rows(p, ["a", "b"]) == {"a": [1.0], "b": [2.0]}
        for bad in (["a"], ["b", "a"], ["a", "c"]):
            try:
                r.checked_rows(p, bad)
            except AssertionError:
                continue
            raise AssertionError(f"accepted mismatched ids {bad}")
        p.write_text('{"id":"a","values":[NaN]}\n')
        try:
            r.checked_rows(p, ["a"])
        except AssertionError:
            pass
        else:
            raise AssertionError("accepted a non-finite value")


def test_analyze_on_synthetic_scores():
    import tempfile
    import diag_run as r
    import diag_cases as c
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        r.SCORES, r.OUT = base / "scores", base / "out"
        r.SCORES.mkdir(); r.OUT.mkdir()
        (r.OUT / "generative").mkdir(); (r.OUT / "kl").mkdir()
        arms = ["fp", "qat_chat_42-t8m", "qat_chat_42-final", "qat_42-t8m", "qat_42-final"]
        plan = {"arms": {a: {} for a in arms}, "sets": list(c.SETS),
                "trajectory": {"qat_chat_42": ["qat_chat_42-t8m", "qat_chat_42-final"],
                               "qat_42": ["qat_42-t8m", "qat_42-final"]},
                "generative_arms": ["fp"], "kl_arms": {"qat_chat_42-final": {}}, "rule": c.RULE}
        def write(arm, key, rows):
            (r.SCORES / f"{arm}-{key}.jsonl").write_text("".join(json.dumps({"id": k, "values": v}) + "\n" for k, v in rows.items()))
        for arm in arms:
            good = arm == "fp"
            letter = {f"s/{i}/{'ABCD'[i % 4]}": [(-0.1 if (k == i % 4 and good) else -2.0) for k in range(4)] for i in range(40)}
            write(arm, "letter", letter); write(arm, "binding", letter)
            cloze = {f"s/{i}/{'ABCD'[i % 4]}/{k}": [0.5 if (k == i % 4 and good) else 2.0] for i in range(40) for k in range(4)}
            write(arm, "cloze_chat", cloze); write(arm, "cloze_raw", cloze)
        (r.OUT / "generative" / "fp.jsonl").write_text(json.dumps({"id": "s/0/A", "gold": "A", "pred": "A", "correct": True}) + "\n")
        (r.OUT / "kl" / "qat_chat_42-final.jsonl").write_text(json.dumps({"id": "s/0/A", "question": 0.4, "template": 0.2, "letter": 1.5, "letter4": 0.9}) + "\n")
        res = r.analyze(plan=plan, write=False)
        assert res["arms"]["fp"]["binding"]["calibrated_accuracy"] == 100.0
        assert res["arms"]["qat_chat_42-final"]["letter"]["pairwise"] == 50.0
        assert res["trajectory"]["qat_chat_42"]["final_minus_t8m"]["letter"]["mean"] == 0.0
        assert res["generative"]["fp"]["accuracy"]["correct"] == 1
        assert res["kl"]["qat_chat_42-final"]["letter"] == 1.5
        assert r.branch(res, c.RULE)["branch"] == "readout_broken"


TESTS = [f for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]

if __name__ == "__main__":
    wanted = sys.argv[1:]
    for t in TESTS:
        if not wanted or t.__name__ in wanted:
            t()
            print("PASS", t.__name__, flush=True)

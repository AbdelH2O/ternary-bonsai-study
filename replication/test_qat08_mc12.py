"""CPU tests for QAT08-MC12. Run: CUDA_VISIBLE_DEVICES= python test_qat08_mc12.py [test_name ...]

Internal restart modes (used by the crash-restart test): --toy {full,pause,resume} --directory D
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEQ = 1024


def _raises(fn, exc=(AssertionError, SystemExit)) -> None:
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"{fn} should have raised")


# ------------------------------------------------------------------------------------------------ stream
def test_layout_counts_synthetic():
    import numpy as np
    import prepare_qat08_mc12 as p
    n = 48
    with tempfile.TemporaryDirectory() as name:
        d = Path(name)
        chat = (np.arange(n * SEQ, dtype=np.uint32) + 10_000_000).reshape(n, SEQ)
        chat.tofile(d / "chat.u32")
        fine = np.arange(40 * SEQ, dtype=np.uint32) + 20_000_000
        fine.tofile(d / "fine.u32")
        mc = np.arange(2500, dtype=np.uint32) + 30_000_000
        counts = p.stream_mc12(mc, d / "out.u32", d / "chat.u32", d / "fine.u32", n=n)
        out = np.fromfile(d / "out.u32", dtype=np.uint32).reshape(n, SEQ)
    assert counts == {"fineweb_seqs": 30, "mc_seqs": 6, "chat_seqs": 12}, counts
    for block in range(n // 16):
        kinds = [p.layout(s) for s in range(block * 16, block * 16 + 16)]
        assert kinds.count("fine") == 10 and kinds.count("chat") == 4 and kinds.count("mc") == 2
    assert all(np.array_equal(out[s], chat[s]) for s in range(n) if p.layout(s) == "chat")
    fine_rows = [s for s in range(n) if p.layout(s) == "fine"]
    assert np.array_equal(out[fine_rows].ravel(), fine[:30 * SEQ]), "FineWeb rows must be the prefix in order"
    mc_rows = [s for s in range(n) if p.layout(s) == "mc"]
    assert np.array_equal(out[mc_rows].ravel(), mc[np.arange(6 * SEQ) % len(mc)]), "MC chunks contiguous, one counter"
    assert p.MC_SLOTS == (10, 14) and all(s % 4 != 3 for s in p.MC_SLOTS)


def test_real_stream_matches_m_c_and_mc_corpus():
    import numpy as np
    import prepare_qat08_mc12 as p
    from score_17b import sha
    if not p.STREAM.exists():
        return
    rec = json.loads((p.DATA / "stream_record.json").read_text())
    assert sha(p.STREAM) == rec["stream_sha256"]
    s12 = np.memmap(p.STREAM, dtype=np.uint32, mode="r").reshape(-1, SEQ)
    c = np.memmap(p.CHAT_STREAM, dtype=np.uint32, mode="r").reshape(-1, SEQ)
    m = np.memmap(p.M_STREAM, dtype=np.uint32, mode="r").reshape(-1, SEQ)
    fw = np.memmap(p.FINEWEB, dtype=np.uint32, mode="r")
    mc = np.fromfile(p.MCU_DATA / "mc_unique.u32", dtype=np.uint32)
    kinds = np.array([p.layout(s) for s in range(p.TRAIN_SEQS)])
    assert (kinds == "fine").sum() == 40_000 and (kinds == "chat").sum() == 16_000 and (kinds == "mc").sum() == 8_000
    assert np.array_equal(s12[3::4], c[3::4]), "chat rows must be byte-identical to C"
    fine_idx = np.flatnonzero(kinds == "fine")
    assert np.array_equal(s12[fine_idx].ravel(), fw[:40_000 * SEQ]), "FineWeb rows = FineWeb seqs 0-39,999"
    m_fine = np.array([s for s in range(p.TRAIN_SEQS) if s % 4 != 3 and s % 16 != 14])
    assert np.array_equal(s12[fine_idx], m[m_fine[:40_000]]), "the FineWeb prefix of M's 44,000"
    mc_idx = np.flatnonzero(kinds == "mc")
    assert np.array_equal(s12[mc_idx].ravel(), mc[np.arange(8000 * SEQ) % len(mc)]), "MC rows = contiguous mc_unique chunks"
    m_mc = np.arange(14, p.TRAIN_SEQS, 16)
    assert np.array_equal(s12[mc_idx[:4000]], m[m_mc]), "the first 4,000 MC chunks are M's"
    assert abs(rec["mc_passes"] - 8000 * SEQ / len(mc)) < 1e-12 and 3.07 < rec["mc_passes"] <= 4.0


# ------------------------------------------------------------------------------------------------ robustness
def test_variants_are_novel_wordings():
    import diag_cases as dc
    import prepare_qat08_mc12 as p
    import prepare_qat08_mcu as mcu
    assert p.VARIANTS["original"] == dc.INSTRUCTION and len(set(p.VARIANTS.values())) == 4
    prompts = (p.MCU_DATA / "mc_prompts.jsonl").read_text()
    for name, text in p.VARIANTS.items():
        if name == "original":
            continue
        assert text not in mcu.PARAPHRASES and text not in prompts, name


def test_robust_cases_re_render_exactly():
    import prepare_qat08_mc12 as p
    from score_17b import sha
    path = p.OUT / "robust_record.json"
    if not path.exists():
        return
    rec = json.loads(path.read_text())
    for key in ("letter", "binding"):
        orig = rec["files"][f"original-{key}"]
        assert orig["sha256"] == orig["source_sha256"], "the original wording must reproduce the dev file byte for byte"
        ids = [l.split("\t", 1)[0] for l in (p.OUT / orig["file"]).read_text().splitlines()]
        for v in p.VARIANTS:
            entry = rec["files"][f"{v}-{key}"]
            assert sha(p.OUT / entry["file"]) == entry["sha256"]
            assert [l.split("\t", 1)[0] for l in (p.OUT / entry["file"]).read_text().splitlines()] == ids, "case order fixed"


def _metrics(fp_binding_raw=99.0, fp_pairwise=66.0, m_items=1.0, c_items=0.5, n=40):
    letter = lambda pw, items: {"raw_accuracy": 40.0, "calibrated_accuracy": 41.0, "bias": [0, 0, 0, 0], "pairwise": pw,
                                "pairwise_lo": pw - 1, "pairwise_hi": pw + 1, "letter_mass": .9, "predicted": {},
                                "pairwise_items": items}
    binding = lambda raw: {"raw_accuracy": raw, "raw_wilson": {}, "calibrated_accuracy": raw, "calibrated_wilson": {},
                           "pairwise": raw, "pairwise_lo": raw, "pairwise_hi": raw, "predicted": {}}
    return {"fp": {"letter": letter(fp_pairwise, [.7] * n), "binding": binding(fp_binding_raw)},
            "C": {"letter": letter(50.0, [c_items + (i % 2) * .02 for i in range(n)]), "binding": binding(25.0)},
            "M": {"letter": letter(58.0, [m_items + (i % 2) * .02 for i in range(n)]), "binding": binding(80.0)}}


def test_robustness_criterion_is_three_valued_and_non_vacuous():
    import freeze_qat08_mc12 as fz
    import robust_qat08_mc12 as r
    vs = ["original", "give_letter", "pick_choice", "output_single"]
    good = {v: _metrics() for v in vs}
    flip = lambda d: {m: {v: d[v][m] for v in vs} for m in ("fp", "C", "M")}
    assert r.evaluate(flip(good), fz.ROBUSTNESS, vs)["status"] == "wording_robust"
    weak = {**good, "pick_choice": _metrics(m_items=.52)}
    assert r.evaluate(flip(weak), fz.ROBUSTNESS, vs)["status"] == "not_robust"
    two_bad = {**good, "give_letter": _metrics(fp_binding_raw=90.0), "pick_choice": _metrics(fp_pairwise=55.0)}
    out = r.evaluate(flip(two_bad), fz.ROBUSTNESS, vs)
    assert out["status"] == "inconclusive" and set(out["table"]) == set(vs), "all rows kept, even FP-invalid ones"
    assert out["excluded_from_criterion_only"] == ["give_letter", "pick_choice"] and out["fp_sanity"]["give_letter"]["reasons"]
    one_bad = {**good, "give_letter": _metrics(fp_binding_raw=90.0)}
    assert r.evaluate(flip(one_bad), fz.ROBUSTNESS, vs)["status"] == "wording_robust", "2 of 3 sane novel wordings suffice"
    assert r.evaluate(flip({**good, "original": _metrics(fp_binding_raw=94.9)}), fz.ROBUSTNESS, vs)["status"] == "harness_invalid"
    no_gain = {**good, "original": _metrics(m_items=.5)}
    assert r.evaluate(flip(no_gain), fz.ROBUSTNESS, vs)["status"] == "inconclusive"


# ------------------------------------------------------------------------------------------------ decision
def _summary(**over):
    ci = lambda m, lo, hi: {"mean": m, "lo": lo, "hi": hi, "n": 10}
    a = {"mmlu_pairwise_vs_M": ci(3.0, 1.5, 4.5), "mmlu_fresh2_pairwise_vs_M": ci(2.5, .5, 4.5),
         "books_vs_M": ci(.02, .01, .03), "books_vs_C": ci(.05, .03, .07), "gsm8k_vs_M": ci(0, -2, 2),
         "retrieval_accuracy": 93.0, "mmlu": ci(38, 36.5, 39.5), "mmlu_fresh2": ci(37, 35, 39), "gsm8k": ci(16, 14, 18),
         "gsm8k_strict": ci(16, 14, 18), "binding_raw": 88.0, "binding_raw_vs_M": ci(8, 5, 11)}
    a.update(over)
    return {"arms": {"MC12": a}}


def test_decide_mc12_rules():
    import freeze_qat08_mc12 as fz
    from score_qat08_mc12 import decide_mc12
    ci = lambda m, lo, hi: {"mean": m, "lo": lo, "hi": hi, "n": 10}
    rules = fz.DECISION
    assert decide_mc12(_summary(), rules)["recommendation"] == "increase_mc_to_12.5"
    assert not decide_mc12(_summary(), rules)["binding_raw_target_met"]
    assert decide_mc12(_summary(mmlu_pairwise_vs_M=ci(1.9, .5, 3.3)), rules)["recommendation"] == "keep_m_6.25_inconclusive"
    assert decide_mc12(_summary(mmlu_fresh2_pairwise_vs_M=ci(1, -.5, 2.5)), rules)["recommendation"] == "keep_m_6.25_inconclusive"
    assert decide_mc12(_summary(books_vs_M=ci(.04, .03, .051)), rules)["recommendation"] == "keep_m_6.25_inconclusive"
    assert decide_mc12(_summary(books_vs_C=ci(.08, .06, .101)), rules)["recommendation"] == "keep_m_6.25_inconclusive"
    assert decide_mc12(_summary(gsm8k_vs_M=ci(-1, -3.1, 1)), rules)["recommendation"] == "keep_m_6.25_inconclusive"
    assert decide_mc12(_summary(retrieval_accuracy=89.5), rules)["recommendation"] == "keep_m_6.25_inconclusive"
    assert decide_mc12(_summary(gsm8k_strict=ci(11, 9.9, 13)), rules)["recommendation"] == "keep_m_6.25_inconclusive"
    assert decide_mc12(_summary(mmlu_pairwise_vs_M=ci(-2, -3, -.1)), rules)["recommendation"] == "keep_m_6.25_mc12_worse"
    assert decide_mc12(_summary(binding_raw=90.0), rules)["binding_raw_target_met"]


# ------------------------------------------------------------------------------------------------ accounting
def test_conservative_accounting():
    import mc12_control as c
    ev = [{"event": "sv_start", "stage": "robust-score", "time": 0.0},
          {"event": "sv_finish", "stage": "robust-score", "time": 100.0, "seconds": 100.0},
          {"event": "sv_start", "stage": "train-MC12", "time": 1000.0},
          {"event": "sv_heartbeat", "stage": "train-MC12", "time": 1600.0},
          {"event": "sv_heartbeat", "stage": "train-MC12", "time": 2200.0}]
    assert abs(c.used(events=ev, now=1e9) * 3600 - (100 + 1800)) < 1e-6, "unfinished stage charged to last heartbeat + interval"
    ev2 = ev + [{"event": "sv_start", "stage": "train-MC12", "time": 2500.0},
                {"event": "sv_finish", "stage": "train-MC12", "time": 2700.0, "seconds": 200.0}]
    assert abs(c.used(events=ev2, now=1e9) * 3600 - (100 + 1500 + 200)) < 1e-6, "capped at the next supervisor start"
    paused = [{"event": "sv_start", "stage": "train-MC12", "time": 0.0},
              {"event": "sv_finish", "stage": "train-MC12", "time": 300.0, "seconds": 300.0, "outcome": "exited", "rc": 78},
              {"event": "sv_start", "stage": "train-MC12", "time": 50_000.0},
              {"event": "sv_finish", "stage": "train-MC12", "time": 50_200.0, "seconds": 200.0}]
    assert abs(c.used(events=paused, now=1e9) * 3600 - 500) < 1e-6, "paused downtime is not charged"
    running = [{"event": "sv_start", "stage": "train-MC12", "time": 0.0}, {"event": "sv_heartbeat", "stage": "train-MC12", "time": 600.0}]
    assert abs(c.used(events=running, now=700.0) * 3600 - 700) < 1e-6, "a running stage is never charged future time"
    assert abs(c.used(grp="train", events=ev2, now=1e9) * 3600 - 1700) < 1e-6 and abs(c.used(grp="robust", events=ev2, now=1e9) * 3600 - 100) < 1e-6


def test_limits_reserve_grace_and_ceilings():
    import inspect
    import freeze_qat08_mc12 as fz
    import mc12_control as c
    assert c.CEILING == {"robust": fz.BUDGET["robust_gpu_hours"], "train": fz.BUDGET["train_gpu_hours"],
                         "eval": fz.BUDGET["eval_gpu_hours"]} and c.TOTAL == fz.BUDGET["total_gpu_hours"] == 8.0
    assert c.GRACE_S == fz.BUDGET["grace_s"] and c.HEARTBEAT_S == fz.BUDGET["heartbeat_s"]
    ev = [{"event": "sv_start", "stage": "train-MC12", "time": 0.0},
          {"event": "sv_finish", "stage": "train-MC12", "time": 6.4 * 3600, "seconds": 6.4 * 3600}]
    assert abs(c.limit_hours("train-MC12", ev, now=1e9) - 0.1) < 1e-9
    assert abs(c.limit_hours("curve", ev, now=1e9) - 0.75) < 1e-9
    over = [{"event": "sv_start", "stage": "train-MC12", "time": 0.0},
            {"event": "sv_finish", "stage": "train-MC12", "time": 6.6 * 3600, "seconds": 6.6 * 3600},
            {"event": "sv_start", "stage": "robust-score", "time": 1e6},
            {"event": "sv_finish", "stage": "robust-score", "time": 1e6 + 0.75 * 3600, "seconds": 0.75 * 3600}]
    assert abs(c.limit_hours("curve", over, now=1e9) - (8.0 - 6.6 - 0.75)) < 1e-9, "the shared total binds after an overrun"
    src = inspect.getsource(c.supervise)
    assert "limit * 3600 - GRACE_S" in src and "limit * 3600 <= GRACE_S + 60" in src


def test_wait_group_kills_descendants():
    import mc12_control as c
    with tempfile.TemporaryDirectory() as name:
        pidfile = Path(name) / "grandchild.pid"
        code = ("import signal, subprocess, sys, time\n"
                "signal.signal(signal.SIGUSR1, signal.SIG_IGN)\n"
                f"g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
                f"open({str(pidfile)!r}, 'w').write(str(g.pid))\n"
                "time.sleep(120)\n")
        child = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
        while not pidfile.exists():
            time.sleep(.05)
        outcome, _ = c.wait_group(child, deadline=time.time() + .5, grace=.5, interval=.2)
        grandchild = int(pidfile.read_text())
        time.sleep(.3)
        alive = True
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            alive = False
        if alive:  # a zombie still answers kill(0); check its state
            alive = Path(f"/proc/{grandchild}/stat").exists() and Path(f"/proc/{grandchild}/stat").read_text().split()[2] != "Z"
    assert outcome == "budget" and not alive, "the budget hard stop must kill the worker's whole process group"


def test_launch_rejects_concurrent_phase():
    import mc12_control as c
    real_run, real_guard = c.subprocess.run, c.guard

    class R:
        def __init__(self, rc):
            self.returncode = rc
    c.guard = lambda: None
    c.subprocess.run = lambda args, **kw: R(0 if args[-1] == "qat08-mc12-run.service" else 3)
    try:
        _raises(lambda: c.launch("robust"), SystemExit)
    finally:
        c.subprocess.run, c.guard = real_run, real_guard


def test_worker_pause_signal_marks_training_and_runtime():
    import inspect
    import mc12_control as c
    src = inspect.getsource(c.worker)
    assert "mcu.request_pause(signum, frame)" in src and "rt._requested = True" in src
    assert "start_new_session=True" in inspect.getsource(c.supervise)
    assert "durable_json(WORK / \"stages\"" in src


def test_events_recover_truncated_tail():
    import mc12_control as c
    with tempfile.TemporaryDirectory() as name:
        path = Path(name) / "execution.jsonl"
        path.write_text('{"event": "sv_start", "stage": "a", "time": 1}\n{"event": "sv_heartbeat", "stage": "a", "time": 2}\n{"event": "sv_fin')
        out = c._events(path)
        assert len(out) == 3 and out[-1]["event"] == "artifact_preserved"
        assert [json.loads(l)["event"] for l in path.read_text().splitlines()] == ["sv_start", "sv_heartbeat", "artifact_preserved"]
        assert len(list(Path(name).glob("execution.jsonl.interrupted-*"))) == 1
        path.write_text('{"event": "sv_start", "time": 1}\nnot json\n{"event": "x", "time": 2}\n')
        _raises(lambda: c._events(path), SystemExit)


def test_repair_log_cases():
    import torch
    import mc12_control as c
    with tempfile.TemporaryDirectory() as name:
        run, inc = Path(name) / "work/MC12", Path(name) / "work/incidents.json"
        run.mkdir(parents=True)
        torch.save({"step": 2}, run / "resume.pt")
        lines = [{"step": 0, "monitor_kl": 1.0}, *[{"step": i, "train_kl": 1.0 / i} for i in range(1, 5)]]
        (run / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines) + '{"step": 5, "tr')
        entry = c.repair_log(run, inc)
        kept = [json.loads(l) for l in (run / "train.jsonl").read_text().splitlines()]
        assert [r["step"] for r in kept] == [0, 1, 2] and entry["records_dropped"] == 2 and entry["partial_line_recovered"]
        assert len(list(run.glob("train.jsonl.before-restart-*"))) == 1 and json.loads(inc.read_text())[0]["resume_step"] == 2
        assert c.repair_log(run, inc) is None, "an already consistent log is left alone"
        (run / "train.jsonl").write_text('{"step": 1}\n{"step": 2}\n{"event": "paused", "step": 2}\n')
        assert c.repair_log(run, inc) is None, "a graceful pause needs no repair"
        (run / "train.jsonl").write_text('{"step": 1}\nbroken\n{"step": 2}\n')
        _raises(lambda: c.repair_log(run, inc), SystemExit)
        (run / "resume.pt").unlink()
        (run / "train.jsonl").write_text('{"step": 0, "monitor_kl": 1}\n{"step": 1, "train_kl": 1}\n')
        entry = c.repair_log(run, inc)
        assert (run / "train.jsonl").read_text() == "" and entry["resume_step"] == 0, "no checkpoint: restart from step 0"
        assert len(list(run.glob("train.jsonl.before-restart-*"))) == 2, "archives are never overwritten"


def _toy_setup(directory: Path) -> None:
    import test_qat08_mcu as t
    from score_17b import sha
    t._toy_setup(directory, False)
    p = json.loads((directory / "results/protocol.json").read_text())
    p["arms"]["toy"]["seed"] = 7
    (directory / "results/design.json").write_text("{}")
    (directory / "results/protocol.json").write_text(json.dumps(p))
    (directory / "results/protocol_approval.json").write_text(json.dumps({
        "approved_by_user": True, "design_sha256": sha(directory / "results/design.json"),
        "protocol_sha256": sha(directory / "results/protocol.json")}))


def _toy_main(mode: str, directory: Path) -> None:
    import qat_08b as qat
    import qat_08b_mcu as mcu
    import test_qat08_mcu as t
    import mc12_control as c
    t._toy_install(qat, mcu, directory)
    original = qat.AdamWBF16.step

    def step(opt, lr):
        original(opt, lr)
        if mode == "pause" and opt.params and opt.t == 2:
            (directory / "work/PAUSE_REQUESTED").write_text("{}")
    qat.AdamWBF16.step = step
    if mode == "resume":
        (directory / "work/PAUSE_REQUESTED").unlink()
    try:
        c.train("toy", out=directory / "results", work=directory / "work")
    except mcu.Paused:
        assert mode == "pause"


def test_toy_crash_restart_no_duplicate_steps():
    import torch
    with tempfile.TemporaryDirectory() as name:
        full, split = Path(name) / "full", Path(name) / "split"
        full.mkdir(); split.mkdir(); _toy_setup(full); _toy_setup(split)
        run = lambda mode, d: subprocess.run([sys.executable, str(Path(__file__)), "--toy", mode, "--directory", str(d)], check=True)
        run("full", full)
        run("pause", split)
        assert torch.load(split / "work/toy/resume.pt", weights_only=False)["step"] == 2
        log = split / "work/toy/train.jsonl"   # simulate a crash after the checkpoint: later records and a torn line
        log.write_text(log.read_text() + '{"step": 3, "train_kl": 9.0}\n{"step": 4, "train_kl": 9.0}\n{"step": 5, "tra')
        run("resume", split)
        steps = [json.loads(l)["step"] for l in log.read_text().splitlines() if "train_kl" in json.loads(l)]
        assert steps == [1, 2, 3, 4, 5], steps
        kl = lambda d: [json.loads(l)["train_kl"] for l in (d / "work/toy/train.jsonl").read_text().splitlines() if "train_kl" in json.loads(l)]
        assert kl(full) == kl(split), "a resumed run reproduces the uninterrupted one"
        x = torch.load(full / "work/toy/resume.pt", weights_only=False)
        y = torch.load(split / "work/toy/resume.pt", weights_only=False)
        assert x["step"] == y["step"] == 5 and torch.equal(x["latents"]["projection"], y["latents"]["projection"])
        assert json.loads((split / "work/incidents.json").read_text())[0]["records_dropped"] == 2


# ------------------------------------------------------------------------------------------------ provenance
def _mini_protocol(root: Path) -> None:
    from score_17b import sha
    out = root / "out"
    (out / "cases").mkdir(parents=True)
    (out / "robust").mkdir()
    for n in ("a.py", "b.u32", "stream.u32", "m.gguf", "reused.jsonl", "gsm.jsonl"):
        (root / n).write_text(n)
    for n in ("validation_books.tsv", "heldout_books.tsv", "mmlu.tsv", "mmlu_fresh2.tsv", "binding.tsv", "gsm8k.jsonl", "retrieval.tsv"):
        (out / "cases" / n).write_text(n)
    (out / "robust/original-letter.tsv").write_text("r")
    case = lambda n: {"file": f"cases/{n}", "sha256": sha(out / "cases" / n)}
    (out / "design.json").write_text("{}")
    p = {"design_sha256": sha(out / "design.json"), "implementation_sha256": {"a.py": sha(root / "a.py")},
         "input_sha256": {"b.u32": sha(root / "b.u32")}, "runtime_sha256": {},
         "comparators": {"M": {"file": "m.gguf", "sha256": sha(root / "m.gguf")}},
         "reused_outputs": {"scores": {"reused.jsonl": sha(root / "reused.jsonl")}, "gsm8k": {"gsm.jsonl": sha(root / "gsm.jsonl")}},
         "cases": {"book_cases": {"validation": case("validation_books.tsv"), "heldout": case("heldout_books.tsv")},
                   "mmlu": case("mmlu.tsv"), "mmlu_fresh2": case("mmlu_fresh2.tsv"), "binding": case("binding.tsv"),
                   "gsm8k": case("gsm8k.jsonl"), "retrieval": case("retrieval.tsv")},
         "robustness": {"record": {"files": {"original-letter": {"file": "robust/original-letter.tsv", "sha256": sha(out / "robust/original-letter.tsv")}}}},
         "stream": {"stream": "stream.u32", "stream_sha256": sha(root / "stream.u32")}, "environment": {}}
    (out / "protocol.json").write_text(json.dumps(p))


def test_verify_rejects_every_mutation():
    import freeze_qat08_mc12 as fz
    targets = ["a.py", "b.u32", "stream.u32", "m.gguf", "reused.jsonl", "gsm.jsonl", "out/cases/mmlu_fresh2.tsv",
               "out/cases/heldout_books.tsv", "out/cases/gsm8k.jsonl", "out/robust/original-letter.tsv", "out/design.json"]
    for target in targets:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _mini_protocol(root)
            fz.verify(root, root / "out", check_env=False)
            (root / target).write_text("mutated")
            _raises(lambda: fz.verify(root, root / "out", check_env=False))


def test_approval_binding():
    import mc12_control as c
    from score_17b import sha
    with tempfile.TemporaryDirectory() as name:
        out = Path(name)
        (out / "design.json").write_text("{}")
        (out / "protocol.json").write_text('{"x": 1}')
        saved = c.OUT
        c.OUT = out
        try:
            (out / "protocol_approval.json").write_text(json.dumps({"approved_by_user": True, "design_sha256": sha(out / "design.json"),
                                                                    "protocol_sha256": "0" * 64}))
            _raises(c.check_approval)
            (out / "protocol_approval.json").write_text(json.dumps({"approved_by_user": True, "design_sha256": sha(out / "design.json"),
                                                                    "protocol_sha256": sha(out / "protocol.json")}))
            c.check_approval()
        finally:
            c.OUT = saved


def test_gsm8k_validator():
    import copy
    import gsm8k_17b as gen
    import score_qat08_mc12 as s
    items = [{"id": str(i), "gold": str(i * 3), "prompt_ids": [1]} for i in range(4)]
    rows = []
    for it in items:
        text = f"So the answer is {it['gold']}."
        pred = gen.parse(text)
        rows.append({"id": it["id"], "text": text, "gold": it["gold"], "pred": pred, "correct": gen.correct(pred, it["gold"])})
    s.check_gsm8k(rows, items)
    _raises(lambda: s.check_gsm8k([rows[1], rows[0], *rows[2:]], items))
    _raises(lambda: s.check_gsm8k(rows[:3], items))
    bad = copy.deepcopy(rows); bad[2]["correct"] = not bad[2]["correct"]
    _raises(lambda: s.check_gsm8k(bad, items))
    bad = copy.deepcopy(rows); bad[1]["gold"] = "999"
    _raises(lambda: s.check_gsm8k(bad, items))
    bad = copy.deepcopy(rows); bad[3]["text"] = "I think it is 7."
    _raises(lambda: s.check_gsm8k(bad, items))
    with tempfile.TemporaryDirectory() as name:
        out = Path(name)
        (out / "gsm8k").mkdir()
        (out / "cases").mkdir()
        (out / "cases/gsm8k.jsonl").write_text("".join(json.dumps(it) + "\n" for it in items))
        (out / "gsm8k/fp.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        from score_17b import sha
        p = {"comparators": {"fp": {}}, "cases": {"gsm8k": {"file": "cases/gsm8k.jsonl", "sha256": sha(out / "cases/gsm8k.jsonl")}},
             "reused_outputs": {"gsm8k": {"results/qat08_mcu/gsm8k/fp.jsonl": sha(out / "gsm8k/fp.jsonl")}}}
        saved = s.OUT
        s.OUT = out
        try:
            s.load_gsm8k("fp", p)
            (out / "gsm8k/fp.jsonl").write_text((out / "gsm8k/fp.jsonl").read_text() + "\n")
            _raises(lambda: s.load_gsm8k("fp", p))
        finally:
            s.OUT = saved


def test_old_records_unchanged():
    from score_17b import sha
    expected = {"results/qat08_mcu/protocol.json": "b99f4bddfd2177c1f52cb83a747eb1173ea030944a02b1e32d145f89b82a783e",
                "results/qat08_mcu/design.json": "8247ef1ee2e87b1b0d55a857225587def0790f9a6f68f2e5e50b2fab295a44a9",
                "results/qat08_mcu/versions/v1/MANIFEST.json": "df6d4cedf1d1cacecb1a30b1b00b598823f368b73d8ca8e3e365d7a00b9a9dc6",
                "results/diag_readout/plan.json": "8c08c1fefc15b782b61cc1d2c983504b396d4b7ec880fe6510749459c6e5b41a",
                "results/diag_readout/decision.json": "90d5b083f419df1ad858a548297ee2cbbb191db3c54a038d99e3fe4618d6b70c",
                "results/diag_readout/decision_amendment.json": "44dd024b6a11ea1adf65bfbdc161cab30ecdd8e911b192b85e0f245fd639c5d6"}
    for name, digest in expected.items():
        assert sha(ROOT / name) == digest, name
    mcu = json.loads((ROOT / "results/qat08_mcu/protocol.json").read_text())
    for name, digest in mcu["implementation_sha256"].items():
        assert sha(ROOT / name) == digest, f"frozen MCU code changed: {name}"
    res = json.loads((ROOT / "results/qat08_mcu/results.json").read_text())
    for name, digest in res["raw_output_sha256"].items():
        assert sha(ROOT / name) == digest, name


def test_namespaces_are_isolated():
    import freeze_qat08_mc12 as fz
    import mc12_control as c
    import prepare_qat08_mc12 as p
    import report_qat08_mc12 as rp
    import robust_qat08_mc12 as r
    import score_qat08_mc12 as s
    written = [p.OUT, p.WORK, p.DATA, p.STREAM, fz.DESIGN, fz.PROTOCOL, fz.APPROVAL, c.OUT, c.WORK, c.PAUSE, r.OUT, r.SCORES,
               s.OUT, s.WORK, s.SCORES]
    assert all("qat08_mc12" in str(x) for x in written), written
    assert rp.REPORT.name == "QAT08_MC12.md"
    assert all(n not in str(x) for x in written for n in ("qat08_mcu/", "qat08_chat/", "diag_readout/"))


def test_book_scan_scoped_to_mc12_only():
    import prepare_qat08_mc12 as p
    import prepare_qat08_mcu as mcu
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        secret = root / "bonsai2/replication/results/qat08_mcu/books/secretnovelzz.raw.txt"
        secret.parent.mkdir(parents=True)
        secret.write_text("x")
        saved = mcu.OWN_DIRS
        found = p.book_inventory(root)
        assert mcu.OWN_DIRS is saved, "the scoped exclusion must be restored"
        assert mcu.book_clashes({"secretnovelzz": (999999, "Zz Unlikely Title")}, found), "an MCU-only book must be found"
        mcu.OWN_DIRS = (secret.parent.parent,)
        try:
            hidden = mcu.book_inventory(root, exclude=set())
        finally:
            mcu.OWN_DIRS = saved
        assert not mcu.book_clashes({"secretnovelzz": (999999, "Zz Unlikely Title")}, hidden), "(MCU's own scope would skip it)"


def test_new_books_unused_including_mcu():
    import prepare_qat08_mc12 as p
    import prepare_qat08_mcu as mcu
    inv = p.book_inventory()
    assert not mcu.book_clashes(p.GUTENBERG, inv) and not mcu.book_clashes(p.RESERVE, inv)
    assert set(mcu.book_clashes(mcu.GUTENBERG, inv)) == set(mcu.GUTENBERG), "every MCU book is seen as used"
    assert set(p.VALIDATION) <= set(p.GUTENBERG) and len(p.GUTENBERG) == 15


def test_content_screen_uses_prompts_and_responses():
    import prepare_qat08_mc12 as p
    choices = ["red", "green", "blue", "yellow"]
    response = "Chlorophyll absorbs mostly red and blue light in leaves. Options: red, green, blue, yellow"
    cands = [(0, {"question": response, "choices": choices}),
             (1, {"question": "Chlorophyll absorbs mostly red and blue light in green leaves", "choices": choices}),
             (2, {"question": "Which planet has the largest mass in the solar system", "choices": ["Mars", "Venus", "Jupiter", "Earth"]})]
    kept, audit = p.content_screen(cands, ["What is the capital of France?\n\nA. Paris"], [response])
    assert [k[0] for k in kept] == [2], (kept, audit)
    assert audit["dropped"].get("mc_exact_question") == 1 and audit["dropped"].get("mc_tfidf_near_duplicate") == 1
    kept, audit = p.content_screen([(9, {"question": "Large winter storms bring heavy snowfall to the mountain.",
                                         "choices": ["x", "y", "z", "w"]})],
                                   ["Multiply two numbers to obtain their product."],
                                   ["Large winter storms bring heavy snowfall to the mountains."])
    assert not kept and audit["dropped"] == {"mc_tfidf_near_duplicate": 1}, "a response-only near duplicate is dropped"


def test_durable_publication_scope_and_export_order():
    import mc12_control as c
    import qat_08b_mcu as mcu
    rt = c._runtime()
    assert rt.managed(c.WORK / "scores/x.jsonl") and rt.managed(c.OUT / "results.json")
    assert not rt.managed(ROOT / "work/qat08_mcu/scores/M-mmlu.jsonl") and not rt.managed(ROOT / "results/qat08_mcu/results.json")
    with tempfile.TemporaryDirectory() as name:
        out, work = Path(name) / "out", Path(name) / "work"
        out.mkdir(); work.mkdir()
        order, real_export, real_fsync = [], mcu.export, c.fsync_paths

        def fake_export(arm, ckpt):
            (work / f"{arm}-{ckpt}.gguf").write_bytes(b"g")
            (out / f"export_{arm}-{ckpt}.json").write_text("{}")
            order.append("exported")
        mcu.export, c.fsync_paths = fake_export, lambda *paths: (order.append(tuple(p.name for p in paths)), real_fsync(*paths))
        try:
            c.export("MC12", "final", out=out, work=work)
        finally:
            mcu.export, c.fsync_paths = real_export, real_fsync
    assert order == ["exported", ("MC12-final.gguf", "export_MC12-final.json")], order
    import inspect
    src = inspect.getsource(c.worker)
    assert src.index("export(arm, ckpt)") < src.index('durable_json(WORK / "stages"'), "outputs before the .done marker"
    assert "rt.install_file_controls()" in src


def test_fresh2_real_exclusions():
    import prepare_qat08_mc12 as p
    from prepare_qat08_chat import tokenizer
    rec_path = p.OUT / "cases_record.json"
    if not rec_path.exists():
        return
    rec = json.loads(rec_path.read_text())["mmlu_fresh2"]
    tok = tokenizer()
    q, g, o, idx = p.tsv_signatures(tok, p.OUT / rec["file"])
    hq, hg, ho, hidx = p.tsv_signatures(tok, p.MCU_V1_FRESH)
    rq, rg, ro = __import__("prepare_qat08_mcu").redux_signatures(tok)
    assert rec["items"] == 3000 and len(idx) == 3000
    assert not idx & hidx and not q & hq and not g & hg and not o & ho, "disjoint from every archived MCU-fresh item"
    assert {7642, 7932} <= hidx and not {7642, 7932} & idx, "the two Hurston items stay excluded"
    assert not q & rq and not o & ro, "disjoint from MMLU-Redux cases"


def test_freeze_pins_reproduction_inputs():
    import inspect
    import freeze_qat08_mc12 as fz
    src = inspect.getsource(fz.inputs)
    assert "results/diag_readout/results.json" in src and "work/diag_readout/scores/" in src
    assert "case_entries(cases)" in src


TESTS = [f for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tests", nargs="*")
    ap.add_argument("--toy", choices=["full", "pause", "resume"])
    ap.add_argument("--directory")
    a = ap.parse_args()
    if a.toy:
        _toy_main(a.toy, Path(a.directory))
    else:
        for t in TESTS:
            if not a.tests or t.__name__ in a.tests:
                t()
                print("PASS", t.__name__, flush=True)

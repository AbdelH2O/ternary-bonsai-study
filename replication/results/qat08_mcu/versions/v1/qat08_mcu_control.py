"""Start, pause, resume and inspect QAT08-MCU phases as systemd user units (one phase at a time).

    python qat08_mcu_control.py start --phase prep|data|run    # each needs its own user approval file
    python qat08_mcu_control.py pause | resume | status
Durable state lives in work/qat08_mcu: run_state.json, execution.jsonl, stages/<stage>.done, PAUSE_REQUESTED.
"""
from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

from score_17b import sha

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/qat08_mcu"
WORK = ROOT / "work/qat08_mcu"
DIAG = ROOT / "results/diag_readout"
PYTHON = ROOT.parents[2] / ".venv/bin/python"
PAUSE = WORK / "PAUSE_REQUESTED"
PROBE_LRS = (0.0, 1e-5, 1e-4, 1e-3)
CEILING = {"prep": 2.0, "data": 1.0}
TOTAL, PER_ARM_TRAIN = 25.0, 6.5


def diagnostics(diag: Path | None = None) -> tuple[dict, dict]:
    """Effective diagnostics decision: decision.json, or its hash-bound post-results amendment when one exists.

    Both paths require decision.json to be bound to the current results.json and plan.json; an amendment must also be
    bound to the current decision.json, results.json and plan.json."""
    diag = diag or DIAG
    decision = json.loads((diag / "decision.json").read_text())
    current = {"decision": sha(diag / "decision.json"), "results": sha(diag / "results.json"), "plan": sha(diag / "plan.json")}
    assert decision["results_sha256"] == current["results"], "decision bound to different results than results.json"
    assert decision["plan_sha256"] == current["plan"], "decision bound to a different plan than plan.json"
    provenance = {"decision_sha256": current["decision"], "amendment_sha256": None}
    path = diag / "decision_amendment.json"
    if not path.exists():
        return decision, provenance
    a = json.loads(path.read_text())
    assert a["amends_decision_sha256"] == current["decision"], "amendment bound to a different decision"
    assert a["results_sha256"] == current["results"], "amendment bound to different results"
    assert a["plan_sha256"] == current["plan"], "amendment bound to a different plan"
    provenance["amendment_sha256"] = sha(path)
    return {**decision, **a["amended"], "amended": True}, provenance


def stages(phase: str, p: dict | None = None, decision: dict | None = None) -> list[str]:
    if phase == "prep":
        decision = decision or diagnostics()[0]
        arms = decision["arms_required"] + decision["arms_optional"]
        return ["selftest", "pilot"] + ([f"probe-{lr:g}" for lr in PROBE_LRS] if {"U", "MU"} & set(arms) else [])
    if phase == "data":
        return ["generate", "pack", "protocol"]
    p = p or json.loads((OUT / "protocol.json").read_text())
    tags = [t for _, t in p["training"]["checkpoints"]]
    exports = [f"export-{a}-{c}" for a, s in p["arms"].items()
               for c in ((["init"] if s["init"] == "folded" else []) + tags)]
    return [f"train-{a}" for a in p["arms"]] + exports + ["curve", "heldout", "gsm8k", "analyze", "report"]


def check_approval(phase: str) -> None:
    if phase == "prep":
        a = json.loads((OUT / "prep_approval.json").read_text())
        assert a["cases_record_sha256"] == sha(OUT / "cases_record.json")
        assert a["mc_record_sha256"] == sha(WORK / "data/mc_record.json")
        provenance = diagnostics()[1]
        assert a["decision_sha256"] == provenance["decision_sha256"]
        assert a.get("amendment_sha256") == provenance["amendment_sha256"], "prep approval must bind the diagnostics amendment"
    elif phase == "data":
        a = json.loads((OUT / "design_approval.json").read_text())
        assert a["design_sha256"] == sha(OUT / "design.json")
    else:
        a = json.loads((OUT / "protocol_approval.json").read_text())
        assert a["design_sha256"] == sha(OUT / "design.json"), "approval bound to a different design"
        assert a["protocol_sha256"] == sha(OUT / "protocol.json"), "approval bound to a different protocol"
    assert a["approved_by_user"] is True


def _runtime():
    import qat08_chat_runtime as rt
    rt.WORK, rt.OUT, rt.PAUSE = WORK, OUT, PAUSE
    return rt


def _events() -> list[dict]:
    path = WORK / "execution.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def _used(phase: str | None = None, stage: str | None = None) -> float:
    """GPU-hours from the supervisor's own sv_* events. A stage with no sv_finish (supervisor killed, reboot)
    counts up to its last heartbeat, so at most one heartbeat interval is lost."""
    ev = [e for e in _events() if str(e.get("event", "")).startswith("sv_")]
    total = 0.0
    for i, e in enumerate(ev):
        if e["event"] != "sv_start" or (phase and e["phase"] != phase) or (stage and e["stage"] != stage):
            continue
        end = None
        for later in ev[i + 1:]:
            if later["stage"] != e["stage"]:
                continue
            if later["event"] == "sv_start":
                break
            if later["event"] == "sv_finish":
                end = e["time"] + later["seconds"]
                break
            end = later["time"]
        total += (end - e["time"]) if end is not None else 0.0
    return total / 3600


def _wait(child, deadline: float, grace: float = 300.0, beat=None, interval: float = 600.0) -> tuple[str, int]:
    """Wait for a stage; heartbeat every interval; at the deadline request a pause, then hard-kill after grace."""
    while True:
        try:
            return "exited", child.wait(timeout=max(0.01, min(interval, deadline - time.time())))
        except subprocess.TimeoutExpired:
            if time.time() < deadline:
                beat()
                continue
            child.send_signal(signal.SIGUSR1)
            try:
                child.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            return "budget", child.returncode


def install_scoring_pause(rt) -> None:
    """Check for a pause before every scorer launch (shard granularity), as the chat runtime did."""
    original = subprocess.run

    def run(*args, **kwargs):
        rt.check_pause()
        return original(*args, **kwargs)
    subprocess.run = run


def _write(path: Path, obj: dict) -> None:
    _runtime().durable_json(path, obj)


def launch(phase: str) -> None:
    check_approval(phase)
    if (WORK / "budget_stop.json").exists():
        raise SystemExit("compute ceiling reached; a versioned budget amendment is required")
    unit = f"qat08-mcu-{phase}"
    if subprocess.run(["systemctl", "--user", "is-active", "--quiet", unit + ".service"]).returncode == 0:
        raise SystemExit(f"{unit} is already running")
    WORK.mkdir(parents=True, exist_ok=True)
    if PAUSE.exists():
        PAUSE.unlink()
    subprocess.run(["systemd-run", "--user", f"--unit={unit}", "--collect", "-p", f"WorkingDirectory={ROOT}",
                    "-p", "KillMode=mixed", "-p", "TimeoutStopSec=300",
                    "-p", f"StandardOutput=append:{WORK / 'run.log'}", "-p", "StandardError=inherit",
                    "/usr/bin/env", "HF_HUB_DISABLE_XET=1", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                    str(PYTHON), "-u", str(Path(__file__)), "supervise", "--phase", phase], check=True)
    print(f"{unit} started. Pause with: python qat08_mcu_control.py pause")


def pause() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    _write(PAUSE, {"requested_at": time.time()})
    for phase in ("prep", "data", "run"):
        subprocess.run(["systemctl", "--user", "kill", "--kill-whom=main", "--signal=SIGUSR1", f"qat08-mcu-{phase}.service"],
                       capture_output=True)
    print("Pause requested; a training step finishes and saves. Check status before rebooting.")


def resume() -> None:
    state = json.loads((WORK / "run_state.json").read_text())
    launch(state["phase"])


def status() -> None:
    for phase in ("prep", "data", "run"):
        subprocess.run(["systemctl", "--user", "show", f"qat08-mcu-{phase}.service", "-p", "ActiveState", "-p", "SubState"])
    path = WORK / "run_state.json"
    print(path.read_text() if path.exists() else "not started")
    print("pause requested:", PAUSE.exists(), "| GPU-h used:", round(_used(), 2), "of", TOTAL)


def supervise(phase: str) -> None:
    check_approval(phase)
    child = None

    def stop(signum, frame):
        _write(PAUSE, {"requested_at": time.time(), "signal": signum})
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGUSR1)
    signal.signal(signal.SIGUSR1, stop)
    signal.signal(signal.SIGTERM, stop)
    (WORK / "stages").mkdir(parents=True, exist_ok=True)
    for name in stages(phase):
        if (WORK / "stages" / f"{name}.done").exists():
            continue
        if PAUSE.exists():
            _write(WORK / "run_state.json", {"state": "paused", "phase": phase, "next_stage": name, "time": time.time()})
            return
        limit = (CEILING[phase] - _used(phase)) if phase in CEILING else (TOTAL - _used())
        if name.startswith("train-"):
            limit = min(limit, PER_ARM_TRAIN - _used(stage=name))
        if limit <= 0:
            _write(WORK / "budget_stop.json", {"phase": phase, "stage": name, "time": time.time()})
            raise SystemExit("compute ceiling reached before " + name)
        _write(WORK / "run_state.json", {"state": "running", "phase": phase, "stage": name, "time": time.time()})
        rt = _runtime()
        started = time.time()
        rt.append_event({"event": "sv_start", "phase": phase, "stage": name, "time": started})
        child = subprocess.Popen([str(PYTHON), "-u", str(Path(__file__)), "worker", "--phase", phase, "--stage", name])
        outcome, rc = _wait(child, deadline=started + limit * 3600,
                            beat=lambda: rt.append_event({"event": "sv_heartbeat", "phase": phase, "stage": name}))
        rt.append_event({"event": "sv_finish", "phase": phase, "stage": name, "seconds": time.time() - started,
                         "outcome": outcome, "rc": rc})
        child = None
        if outcome == "budget":
            _write(WORK / "budget_stop.json", {"phase": phase, "stage": name, "time": time.time()})
            _write(WORK / "run_state.json", {"state": "budget_stop", "phase": phase, "stage": name, "time": time.time()})
            raise SystemExit("compute ceiling reached during " + name)
        if rc == 78 or PAUSE.exists():
            _write(WORK / "run_state.json", {"state": "paused", "phase": phase, "next_stage": name, "time": time.time()})
            return
        if rc:
            _write(WORK / "run_state.json", {"state": "failed", "phase": phase, "stage": name, "exit_code": rc, "time": time.time()})
            raise SystemExit(rc)
    _write(WORK / "run_state.json", {"state": "complete", "phase": phase, "time": time.time()})


def worker(phase: str, name: str) -> None:
    rt = _runtime()
    import qat_08b_mcu as mcu
    signal.signal(signal.SIGUSR1, mcu.request_pause)
    rt.install_generation_controls()
    started = time.time()
    rt.append_event({"event": "stage_started", "phase": phase, "stage": name})
    outcome = "failed"
    try:
        if phase in ("data", "run"):
            import freeze_qat08_mcu as freeze
            freeze.verify_design()
        if name == "selftest":  # GPU self-test, counted in the prep ceiling like every other prep stage
            mcu.selftest()
        elif name == "pilot" or name == "generate":
            import prepare_qat08_mcu as prep
            rt.recover_journal(WORK / "data/mc_teacher.jsonl")
            prep.generate(pilot=name == "pilot")
        elif name.startswith("probe-"):
            mcu.probe(float(name.split("-", 1)[1]))
        elif name == "pack":
            import prepare_qat08_mcu as prep
            prep.pack()
        elif name == "protocol":
            import freeze_qat08_mcu as freeze
            freeze.protocol()
        elif name.startswith("train-"):
            mcu.approved()
            mcu.train(name.split("-", 1)[1])
        elif name.startswith("export-"):
            _, arm, ckpt = name.split("-", 2)
            mcu.export(arm, ckpt)
        elif name == "report":
            import report_qat08_mcu
            report_qat08_mcu.write_report()
        else:
            import score_qat08_mcu as score
            for journal in (OUT / "gsm8k").glob("*.partial.jsonl"):
                rt.recover_journal(journal)
            install_scoring_pause(rt)
            getattr(score, name)()
        outcome = "complete"
        (WORK / "stages" / f"{name}.done").write_text(json.dumps({"time": time.time()}) + "\n")
    except (mcu.Paused, rt.Paused) as e:
        outcome = "paused"
        print(str(e), flush=True)
        raise SystemExit(78)
    finally:
        rt.append_event({"event": "stage_finished", "phase": phase, "stage": name, "outcome": outcome,
                         "seconds": time.time() - started})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["start", "pause", "resume", "status", "supervise", "worker"])
    ap.add_argument("--phase", choices=["prep", "data", "run"])
    ap.add_argument("--stage")
    a = ap.parse_args()
    {"start": lambda: launch(a.phase), "pause": pause, "resume": resume, "status": status,
     "supervise": lambda: supervise(a.phase), "worker": lambda: worker(a.phase, a.stage)}[a.cmd]()

"""Start, pause, resume and inspect QAT08-MC12 phases as systemd user units (one phase at a time).

    python mc12_control.py start --phase robust   # prompt robustness of FP, C and M (<= 0.75 GPU-h)
    python mc12_control.py start --phase run      # train MC12, export, evaluate, analyze, report
    python mc12_control.py pause | resume | status

Every launch, resume and stage re-verifies the frozen protocol (freeze_qat08_mc12.verify) and the approval binding.
Accounting is conservative (see BUDGET in freeze_qat08_mc12.py). The frozen MCU trainer and exporter are reused
unchanged; this worker points them at the MC12 namespace and pins the MC12 seed.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from qat08_mcu_control import install_scoring_pause
from score_17b import sha

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/qat08_mc12"
WORK = ROOT / "work/qat08_mc12"
PAUSE = WORK / "PAUSE_REQUESTED"
PYTHON = ROOT.parents[2] / ".venv/bin/python"
CEILING = {"robust": 0.75, "train": 6.5, "eval": 0.75}
TOTAL, GRACE_S, HEARTBEAT_S = 8.0, 300.0, 600.0
CKPTS = ("init", "t8m", "t16m", "t33m", "final")
STAGES = {"robust": ["robust-score", "robust-analyze"],
          "run": ["train-MC12", *[f"export-MC12-{c}" for c in CKPTS], "curve", "heldout", "gsm8k", "analyze", "report"]}


def group(stage: str) -> str:
    return "robust" if stage.startswith("robust-") else "train" if stage.startswith("train-") else "eval"


def _runtime():
    import qat08_chat_runtime as rt
    rt.WORK, rt.OUT, rt.PAUSE = WORK, OUT, PAUSE
    return rt


def _archive_copy(path: Path, reason: str) -> Path:
    """Exclusive, durable copy of an artifact before repair (time_ns name; never overwrites earlier evidence)."""
    target = path.with_name(f"{path.name}.{reason}-{time.time_ns()}")
    with target.open("xb") as f:
        f.write(path.read_bytes())
        f.flush()
        os.fsync(f.fileno())
    return target


def _events(path: Path | None = None) -> list[dict]:
    """Supervisor events. An incomplete final record (crash mid-write) is archived and removed, with a recovery event
    appended; a corrupt interior record is a stop. (rt.recover_journal would lose its own event for this file, since
    it appends the event to the journal it then rewrites.)"""
    path = path or WORK / "execution.jsonl"
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.split(b"\n")
    out = []
    for i, line in enumerate(lines[:-1]):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"interior corrupt execution record: {path}:{i + 1}") from e
    if lines[-1]:
        try:
            out.append(json.loads(lines[-1]))
        except json.JSONDecodeError:
            archive = _archive_copy(path, "interrupted")
            event = {"time": time.time(), "event": "artifact_preserved", "file": str(path), "archive": str(archive),
                     "reason": "incomplete final execution record after interruption"}
            tmp = path.with_name(path.name + ".repair")
            with tmp.open("w") as f:
                f.write("".join(json.dumps(e) + "\n" for e in out) + json.dumps(event) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            _runtime().fsync_dir(path.parent)
            out.append(event)
    return out


def used(grp: str | None = None, stage: str | None = None, events: list[dict] | None = None, now: float | None = None) -> float:
    """GPU-hours from the supervisor's sv_* events. A finished stage counts its recorded seconds (paused downtime is
    outside any stage). An unfinished stage (crash, kill, reboot) counts to its last heartbeat plus one full heartbeat
    interval, capped at the next supervisor stage start and at the present (a running stage is never charged future
    time), so time lost to a crash is charged rather than dropped."""
    now = time.time() if now is None else now
    ev = [e for e in (events if events is not None else _events()) if str(e.get("event", "")).startswith("sv_")]
    total = 0.0
    for i, e in enumerate(ev):
        if e["event"] != "sv_start" or (stage and e["stage"] != stage) or (grp and group(e["stage"]) != grp):
            continue
        end, last, cap = None, e["time"], None
        for later in ev[i + 1:]:
            if later["event"] == "sv_start":
                cap = later["time"]
                break
            if later["stage"] != e["stage"]:
                continue
            if later["event"] == "sv_finish":
                end = e["time"] + later["seconds"]
                break
            last = later["time"]
        if end is None:
            end = min(last + HEARTBEAT_S, now, cap if cap is not None else now)
        total += end - e["time"]
    return total / 3600


def limit_hours(stage: str, events: list[dict] | None = None, now: float | None = None) -> float:
    """Hours available to a stage: the smaller of the total and its group's remaining ceiling."""
    return min(TOTAL - used(events=events, now=now), CEILING[group(stage)] - used(grp=group(stage), events=events, now=now))


def wait_group(child, deadline: float, grace: float = GRACE_S, beat=None, interval: float = HEARTBEAT_S) -> tuple[str, int]:
    """Wait for a worker started in its own process group; heartbeat every interval. At the deadline request a pause
    (SIGUSR1 to the worker), then hard-kill the whole group (scorer, llama-server, any descendant) after grace."""
    while True:
        try:
            return "exited", child.wait(timeout=max(0.01, min(interval, deadline - time.time())))
        except subprocess.TimeoutExpired:
            if time.time() < deadline:
                if beat:
                    beat()
                continue
            child.send_signal(signal.SIGUSR1)
            try:
                child.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                pass
            kill_group(child.pid)
            child.wait()
            return "budget", child.returncode


def kill_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def check_approval() -> dict:
    a = json.loads((OUT / "protocol_approval.json").read_text())
    assert a["approved_by_user"] is True, "not approved"
    assert a["design_sha256"] == sha(OUT / "design.json"), "approval bound to a different design"
    assert a["protocol_sha256"] == sha(OUT / "protocol.json"), "approval bound to a different protocol"
    return a


def guard() -> dict:
    import freeze_qat08_mc12 as fz12
    p = fz12.verify()
    check_approval()
    return p


def _write(path: Path, obj: dict) -> None:
    _runtime().durable_json(path, obj)


def launch(phase: str) -> None:
    guard()
    if (WORK / "budget_stop.json").exists():
        raise SystemExit("compute ceiling reached; a versioned budget amendment is required")
    if phase == "run" and not (WORK / "stages/robust-analyze.done").exists():
        raise SystemExit("the robustness phase must complete before training")
    unit = f"qat08-mc12-{phase}"
    active = [ph for ph in STAGES if subprocess.run(["systemctl", "--user", "is-active", "--quiet",
                                                       f"qat08-mc12-{ph}.service"]).returncode == 0]
    if active:
        raise SystemExit(f"an MC12 phase is already running: {active}; one phase at a time")
    WORK.mkdir(parents=True, exist_ok=True)
    if PAUSE.exists():
        PAUSE.unlink()
    subprocess.run(["systemd-run", "--user", f"--unit={unit}", "--collect", "-p", f"WorkingDirectory={ROOT}",
                    "-p", "KillMode=mixed", "-p", "TimeoutStopSec=300",
                    "-p", f"StandardOutput=append:{WORK / 'run.log'}", "-p", "StandardError=inherit",
                    "/usr/bin/env", "HF_HUB_DISABLE_XET=1", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                    str(PYTHON), "-u", str(Path(__file__)), "supervise", "--phase", phase], check=True)
    print(f"{unit} started. Pause with: python mc12_control.py pause")


def pause() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    _write(PAUSE, {"requested_at": time.time()})
    for phase in STAGES:
        subprocess.run(["systemctl", "--user", "kill", "--kill-whom=main", "--signal=SIGUSR1", f"qat08-mc12-{phase}.service"],
                       capture_output=True)
    print("Pause requested; a training step finishes and saves. Check status before rebooting.")


def resume() -> None:
    state = json.loads((WORK / "run_state.json").read_text())
    launch(state["phase"])


def status() -> None:
    for phase in STAGES:
        subprocess.run(["systemctl", "--user", "show", f"qat08-mc12-{phase}.service", "-p", "ActiveState", "-p", "SubState"])
    path = WORK / "run_state.json"
    print(path.read_text() if path.exists() else "not started")
    print("pause requested:", PAUSE.exists(), "| GPU-h used:", round(used(), 3), "of", TOTAL,
          {g: round(used(grp=g), 3) for g in CEILING})


def supervise(phase: str) -> None:
    guard()
    child = None

    def stop(signum, frame):
        _write(PAUSE, {"requested_at": time.time(), "signal": signum})
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGUSR1)
    signal.signal(signal.SIGUSR1, stop)
    signal.signal(signal.SIGTERM, stop)
    (WORK / "stages").mkdir(parents=True, exist_ok=True)
    rt = _runtime()
    for name in STAGES[phase]:
        if (WORK / "stages" / f"{name}.done").exists():
            continue
        if PAUSE.exists():
            _write(WORK / "run_state.json", {"state": "paused", "phase": phase, "next_stage": name, "time": time.time()})
            return
        guard()
        limit = limit_hours(name)
        if limit * 3600 <= GRACE_S + 60:
            _write(WORK / "budget_stop.json", {"phase": phase, "stage": name, "limit_hours": limit, "time": time.time()})
            raise SystemExit("compute ceiling reached before " + name)
        _write(WORK / "run_state.json", {"state": "running", "phase": phase, "stage": name, "time": time.time()})
        started = time.time()
        rt.append_event({"event": "sv_start", "phase": phase, "stage": name, "time": started})
        child = subprocess.Popen([str(PYTHON), "-u", str(Path(__file__)), "worker", "--phase", phase, "--stage", name],
                                 start_new_session=True)
        outcome, rc = wait_group(child, deadline=started + limit * 3600 - GRACE_S, grace=GRACE_S, interval=HEARTBEAT_S,
                                 beat=lambda: rt.append_event({"event": "sv_heartbeat", "phase": phase, "stage": name}))
        kill_group(child.pid)  # nothing of a finished stage may outlive it
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


def fsync_run(directory: Path) -> None:
    """Durability after a pause or completion: every file of the run (checkpoint, log, tags, DONE) and the directory."""
    for path in sorted(directory.iterdir()):
        if path.is_file():
            with path.open("rb") as f:
                os.fsync(f.fileno())
    fd = os.open(directory, os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)


def repair_log(run: Path, incidents: Path) -> dict | None:
    """Before a resume: drop training-log records beyond the step saved in resume.pt (a crash after the last periodic
    checkpoint), so the resumed trainer does not log duplicate steps. The original log is archived first; an incomplete
    last line is recovered; corruption anywhere else is a stop. A graceful pause needs no change."""
    import torch
    resume, log = run / "resume.pt", run / "train.jsonl"
    if not log.exists():
        return None
    # no checkpoint: a crash before the first periodic save restarts from step 0, so the old log is reset, never appended
    step = torch.load(resume, map_location="cpu", mmap=True, weights_only=False)["step"] if resume.exists() else -1
    raw = log.read_bytes()
    lines = raw.split(b"\n")
    body, tail = lines[:-1], lines[-1]
    records = []
    for i, line in enumerate(body):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"training log corrupted at line {i + 1}; refusing to resume") from e
    partial = b""
    if tail:
        try:
            records.append(json.loads(tail))
        except json.JSONDecodeError:
            partial = tail
    keep = [r for r in records if r.get("step", 0) <= step]
    if len(keep) == len(records) and not partial and (raw.endswith(b"\n") or not raw):
        return None
    archive = _archive_copy(log, "before-restart")
    tmp = run / "train.jsonl.repair"
    with tmp.open("w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in keep))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, log)
    fsync_run(run)
    entry = {"time": time.time(), "incident": "restart after a crash: training-log records beyond the saved checkpoint removed"
             if step >= 0 else "restart after a crash before the first checkpoint: training log reset to restart at step 0",
             "resume_step": max(step, 0), "records_dropped": len(records) - len(keep), "partial_line_recovered": bool(partial),
             "archived_log": str(archive.relative_to(run.parents[1]) if len(run.parents) > 1 else archive)}
    past = json.loads(incidents.read_text()) if incidents.exists() else []
    _runtime().durable_json(incidents, past + [entry])
    return entry


def train(arm: str = "MC12", out: Path = OUT, work: Path = WORK) -> None:
    """The frozen MCU trainer, pointed at the MC12 namespace, with the MC12 seed pinned before start-up."""
    import random
    import numpy as np
    import torch
    import qat_08b_mcu as mcu
    mcu.OUT, mcu.WORK = out, work
    mcu.approved()
    seed = json.loads((out / "protocol.json").read_text())["arms"][arm]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if (work / arm).exists():
        repair_log(work / arm, work / "incidents.json")
    try:
        mcu.train(arm)
    except mcu.Paused:
        fsync_run(work / arm)
        raise
    fsync_run(work / arm)


def fsync_paths(*paths: Path) -> None:
    """fsync files, then every distinct parent directory."""
    for path in paths:
        with path.open("rb") as f:
            os.fsync(f.fileno())
    for directory in sorted({p.parent for p in paths}):
        fd = os.open(directory, os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)


def export(arm: str, ckpt: str, out: Path = OUT, work: Path = WORK) -> None:
    """The frozen MCU exporter; the GGUF and its record reach disk before the stage's .done marker is published."""
    import qat_08b_mcu as mcu
    mcu.OUT, mcu.WORK = out, work
    mcu.export(arm, ckpt)
    fsync_paths(work / f"{arm}-{ckpt}.gguf", out / f"export_{arm}-{ckpt}.json")


def worker(phase: str, name: str) -> None:
    rt = _runtime()
    import qat_08b_mcu as mcu

    def request(signum, frame):  # budget or user pause: training, scoring and generation all see it
        mcu.request_pause(signum, frame)
        rt._requested = True
    signal.signal(signal.SIGUSR1, request)
    rt.install_file_controls()      # durable appends/text/replace inside the MC12 namespace only (rt.WORK/OUT)
    rt.install_generation_controls()
    started = time.time()
    rt.append_event({"event": "stage_started", "phase": phase, "stage": name})
    outcome = "failed"
    try:
        guard()
        if name == "robust-score":
            import robust_qat08_mc12 as robust
            install_scoring_pause(rt)
            robust.score()
        elif name == "robust-analyze":
            import robust_qat08_mc12 as robust
            if robust.analyze()["status"] == "harness_invalid":
                raise SystemExit("robustness harness invalid: FP fails the floor on the original wording (incident)")
        elif name.startswith("train-"):
            train(name.split("-", 1)[1])
        elif name.startswith("export-"):
            _, arm, ckpt = name.split("-", 2)
            export(arm, ckpt)
        elif name == "report":
            import report_qat08_mc12
            report_qat08_mc12.write_report()
        else:
            import score_qat08_mc12 as score
            for journal in (OUT / "gsm8k").glob("*.partial.jsonl"):
                rt.recover_journal(journal)
            install_scoring_pause(rt)
            getattr(score, name)()
        outcome = "complete"
        rt.durable_json(WORK / "stages" / f"{name}.done", {"time": time.time()})
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
    ap.add_argument("--phase", choices=list(STAGES))
    ap.add_argument("--stage")
    a = ap.parse_args()
    {"start": lambda: launch(a.phase), "pause": pause, "resume": resume, "status": status,
     "supervise": lambda: supervise(a.phase), "worker": lambda: worker(a.phase, a.stage)}[a.cmd]()

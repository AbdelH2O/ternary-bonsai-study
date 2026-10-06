"""Start/status/pause/resume the chat experiment, including after a host reboot.

Transient systemd units need no boot configuration: `resume` recreates the unit.
The authoritative state and checkpoints live on disk in this repository.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work/qat08_chat"
OUT = ROOT / "results/qat08_chat"
UNIT = "qat08-chat-run"
PYTHON = ROOT.parents[2] / ".venv/bin/python"
PAUSE = WORK / "PAUSE_REQUESTED"
CONTROL_FILES = ("qat08_chat_control.py", "qat08_chat_runtime.py", "test_qat08_chat_resume.py", "report_qat08_chat.py")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    from qat08_chat_runtime import durable_json
    durable_json(path, obj)


def seal():
    path = OUT / "resume_amendment_v1.json"
    if path.exists():
        raise SystemExit("execution amendment already sealed")
    tests = json.loads((OUT / "resume_test_results.json").read_text())
    assert tests["passed"] is True
    record = {"date": "2026-10-02", "design_sha256": sha(OUT / "design.json"),
        "authorization": "User requested reboot-safe pause/resume after approving the experiment; this adds lifecycle and durable storage only.",
        "change": "No frozen source, recipe, evaluation case, data-selection rule or decision is changed. A separate wrapper adds pause checks at optimizer boundaries, atomic/fsynced writes, RNG restoration, incomplete-tail journal repair, resumable intermediate publication, and a manual restart controller.",
        "implementation_sha256": {name: sha(ROOT / name) for name in CONTROL_FILES},
        "tests_sha256": sha(OUT / "resume_test_results.json"), "tests": tests,
        "hard_restart_limit": "On sudden power loss, student resumes from its most recent periodic checkpoint (original 1200-second cadence); a graceful pause writes a fresh checkpoint. Interrupted teacher/evaluation work is replayed only after the last durable record/shard.",
        "post_reboot": "Explicit resume is required; paused jobs do not automatically restart at login or boot."}
    with path.open("x") as f:
        json.dump(record, f, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
    from qat08_chat_runtime import fsync_dir
    fsync_dir(path.parent)
    print("sealed execution amendment", sha(path))


def verify_amendment():
    p = OUT / "resume_amendment_v1.json"
    record = json.loads(p.read_text())
    assert record["design_sha256"] == sha(OUT / "design.json")
    for name, digest in record["implementation_sha256"].items():
        assert sha(ROOT / name) == digest, f"execution wrapper changed: {name}"
    assert sha(OUT / "resume_test_results.json") == record["tests_sha256"]
    return record


def launch():
    verify_amendment()
    if (WORK / "budget_stop.json").exists():
        raise SystemExit("Frozen compute ceiling reached; a new budget amendment is required before resuming.")
    approval = json.loads((OUT / "design_approval.json").read_text())
    assert approval["approved_by_user"] is True
    assert approval["design_sha256"] == sha(OUT / "design.json")
    active = subprocess.run(["systemctl", "--user", "is-active", "--quiet", UNIT+".service"])
    if active.returncode == 0:
        raise SystemExit("Experiment is already running; use status or pause.")
    if PAUSE.exists():
        PAUSE.unlink()
        from qat08_chat_runtime import fsync_dir
        fsync_dir(WORK)
    result = subprocess.run(["systemd-run", "--user", f"--unit={UNIT}", "--collect",
        "-p", f"WorkingDirectory={ROOT}", "-p", "KillMode=mixed", "-p", "TimeoutStopSec=180",
        "-p", f"StandardOutput=append:{WORK / 'run.log'}", "-p", "StandardError=inherit",
        "/usr/bin/env", "HF_HUB_DISABLE_XET=1", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        str(PYTHON), "-u", str(Path(__file__)), "supervise"], check=True)
    print("Experiment started/resumed. Pause with: python qat08_chat_control.py pause")


def pause():
    write(PAUSE, {"requested_at": time.time()})
    # Exact unit/main process; never match a command-line pattern.
    result = subprocess.run(["systemctl", "--user", "kill", "--kill-whom=main", "--signal=SIGUSR1", UNIT+".service"],
                            capture_output=True, text=True)
    print("Pause requested. A running training step will finish and save; check status before rebooting.")
    if result.returncode:
        print("No running unit was signalled; the persistent pause request is saved.")


def status():
    subprocess.run(["systemctl", "--user", "show", UNIT+".service", "-p", "ActiveState", "-p", "SubState", "-p", "MainPID"])
    path = WORK / "run_state.json"
    print(path.read_text() if path.exists() else "Experiment has not started.")
    print("Persistent pause requested:", PAUSE.exists())
    checkpoint = WORK / "qat_chat_42/resume.pt"
    if checkpoint.exists():
        print("Training checkpoint:", checkpoint, "modified", time.ctime(checkpoint.stat().st_mtime))
    journal = WORK / "data/teacher.jsonl"
    if journal.exists():
        print("Durable teacher responses:", sum(1 for _ in journal.open()))


def supervise():
    verify_amendment()
    WORK.mkdir(parents=True, exist_ok=True)
    child = None
    def stop(signum, frame):
        write(PAUSE, {"requested_at": time.time(), "signal": signum})
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGUSR1)
    signal.signal(signal.SIGUSR1, stop)
    signal.signal(signal.SIGTERM, stop)
    with (WORK / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stages = ["generate", "pack", "protocol", "train"] + ["export-"+c for c in ("init", "t8m", "t16m", "t33m", "final")] + ["curve", "heldout", "gsm8k", "analyze", "report"]
        for name in stages:
            if PAUSE.exists():
                write(WORK / "run_state.json", {"state": "paused", "next_stage": name, "time": time.time()})
                return
            skip = ((name in ("generate", "pack") and (WORK / "data/data_record.json").exists()) or
                    (name == "protocol" and (OUT / "protocol.json").exists()) or
                    (name == "train" and (WORK / "qat_chat_42/DONE").exists()))
            if skip:
                continue
            if name == "train":
                approval = json.loads((OUT / "design_approval.json").read_text())
                assert approval["approved_by_user"] is True and approval["design_sha256"] == sha(OUT / "design.json")
                write(OUT / "training_approval.json", {**approval, "protocol_sha256": sha(OUT / "protocol.json"),
                                                       "execution_amendment_sha256": sha(OUT / "resume_amendment_v1.json")})
            write(WORK / "run_state.json", {"state": "running", "stage": name, "time": time.time()})
            child = subprocess.Popen([str(PYTHON), "-u", str(Path(__file__)), "worker", "--stage", name])
            events_path = WORK / "execution.jsonl"
            events = [json.loads(l) for l in events_path.read_text().splitlines()] if events_path.exists() else []
            finished = [r for r in events if r.get("event") == "stage_finished"]
            used = sum(r["seconds"] for r in finished)
            limit = 12*3600 - used
            if name == "train":
                training_used = sum(r["seconds"] for r in finished if r["stage"] == "train")
                limit = min(limit, 6.5*3600-training_used)
            try:
                rc = child.wait(timeout=max(1,limit))
            except subprocess.TimeoutExpired:
                stop(signal.SIGTERM, None)
                rc = child.wait(timeout=180)
                write(WORK / "budget_stop.json", {"stage": name, "time": time.time(), "reason": "frozen compute ceiling reached"})
            child = None
            if rc == 78 or PAUSE.exists():
                write(WORK / "run_state.json", {"state": "paused", "next_stage": name, "time": time.time()})
                return
            if rc:
                write(WORK / "run_state.json", {"state": "failed", "stage": name, "exit_code": rc, "time": time.time()})
                raise SystemExit(rc)
        write(WORK / "run_state.json", {"state": "complete", "time": time.time(), "report": str(ROOT / "QAT08_CHAT.md")})


def worker(name):
    verify_amendment()
    import qat08_chat_runtime as runtime
    runtime.install_signals()
    runtime.install_file_controls()
    try:
        # Original frozen code and exact inputs are checked before any generation/training.
        import freeze_qat08_chat as freeze
        freeze.verify_design()
        runtime.check_pause()
        if name == "report":
            import report_qat08_chat
            report_qat08_chat.write_report()
        else:
            runtime.stage(name)
    except runtime.Paused as e:
        print(str(e), flush=True)
        raise SystemExit(78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["seal", "start", "resume", "pause", "status", "supervise", "worker"])
    parser.add_argument("--stage")
    args = parser.parse_args()
    {"seal": seal, "start": launch, "resume": launch, "pause": pause, "status": status,
     "supervise": supervise, "worker": lambda: worker(args.stage)}[args.cmd]()

"""Durable execution controls around the original, hash-frozen research functions.

This layer changes file publication and process lifecycle, not model arithmetic.
No frozen source file is edited. SIGUSR1/SIGTERM request a checkpoint and pause.
"""
from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import os
import random
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work/qat08_chat"
OUT = ROOT / "results/qat08_chat"
PAUSE = WORK / "PAUSE_REQUESTED"
_requested = False
_real_open = Path.open
_real_write_text = Path.write_text
_real_replace = Path.replace


class Paused(Exception):
    pass


def pause_requested():
    return _requested or PAUSE.exists()


def check_pause():
    if pause_requested():
        raise Paused("pause requested; durable progress retained")


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".pending")
    with _real_open(tmp, "w") as f:
        json.dump(value, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def append_event(event):
    WORK.mkdir(parents=True, exist_ok=True)
    with _real_open(WORK / "execution.jsonl", "a") as f:
        f.write(json.dumps({"time": time.time(), **event}) + "\n")
        f.flush()
        os.fsync(f.fileno())


def managed(path):
    p = path.absolute()
    return p.is_relative_to(WORK) or p.is_relative_to(OUT)


def archive(path, reason):
    target = path.with_name(path.name + f".interrupted-{time.time_ns()}")
    os.replace(path, target)
    fsync_dir(path.parent)
    append_event({"event": "artifact_preserved", "file": str(path), "archive": str(target), "reason": reason})


def recover_journal(path):
    """Only discard an incomplete final record, preserving the original as evidence."""
    if not path.exists():
        return
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    valid = []
    for i, line in enumerate(lines):
        try:
            json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if i != len(lines)-1:
                raise RuntimeError(f"interior corrupt journal record: {path}:{i+1}")
            archive(path, "incomplete final journal record after interruption")
            with _real_open(path, "wb") as f:
                f.write(b"".join(valid))
                f.flush()
                os.fsync(f.fileno())
            fsync_dir(path.parent)
            return
        valid.append(line if line.endswith(b"\n") else line+b"\n")
    if lines and not raw.endswith(b"\n"):
        with _real_open(path, "ab") as f:
            f.write(b"\n"); f.flush(); os.fsync(f.fileno())


class _DurableAppend:
    def __init__(self, file):
        self.file = file

    def __getattr__(self, name):
        return getattr(self.file, name)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.file.close()

    def write(self, text):
        result = self.file.write(text)
        self.file.flush()
        os.fsync(self.file.fileno())
        return result


class _AtomicNew:
    def __init__(self, path, args, kwargs):
        self.path = path
        self.tmp = path.with_name(path.name + ".pending")
        self.file = _real_open(self.tmp, "w", *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.file, name)

    def __enter__(self):
        return self.file

    def __exit__(self, typ, value, tb):
        try:
            if typ is None:
                self.file.flush(); os.fsync(self.file.fileno())
            self.file.close()
            if typ is None:
                if self.path.exists():
                    # Packing may have published this deterministic intermediate before a reboot.
                    if self.path.name != "kept_chats.json" or self.path.read_bytes() != self.tmp.read_bytes():
                        raise FileExistsError(self.path)
                    self.tmp.unlink()
                else:
                    if self.path.name == "data_record.json":
                        for name in ("train.u32", "monitor.u32", "chat_unique.u32"):
                            with _real_open(self.path.parent / name, "rb") as f:
                                os.fsync(f.fileno())
                    os.link(self.tmp, self.path)  # exclusive publication; never overwrite a freeze
                    self.tmp.unlink()
                fsync_dir(self.path.parent)
        finally:
            if not self.file.closed:
                self.file.close()


def install_file_controls():
    def open_file(path, mode="r", *args, **kwargs):
        if managed(path):
            if mode == "a" and path.suffix == ".jsonl":
                recover_journal(path)
                return _DurableAppend(_real_open(path, mode, *args, **kwargs))
            if mode == "x":
                return _AtomicNew(path, args, kwargs)
        return _real_open(path, mode, *args, **kwargs)

    def write_text(path, text, *args, **kwargs):
        if not managed(path):
            return _real_write_text(path, text, *args, **kwargs)
        tmp = path.with_name(path.name + ".pending")
        with _real_open(tmp, "w", *args, **kwargs) as f:
            n = f.write(text); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
        fsync_dir(path.parent)
        return n

    def replace(path, target):
        result = _real_replace(path, target)
        if managed(Path(target)):
            # Files such as memmaps/scorer shards are closed before publication.
            with _real_open(Path(target), "rb") as f:
                os.fsync(f.fileno())
            fsync_dir(Path(target).parent)
        return result

    Path.open, Path.write_text, Path.replace = open_file, write_text, replace


def install_signals():
    def request(signum, frame):
        global _requested
        _requested = True
    signal.signal(signal.SIGUSR1, request)
    signal.signal(signal.SIGTERM, request)


def install_generation_controls():
    import gsm8k_17b as gen
    original = gen.complete

    def complete(ids):
        check_pause()  # queued tasks skip inference; at most eight in-flight requests drain
        return original(ids)
    gen.complete = complete


def install_training_controls(qat):
    """Check for pause before the next microbatch-0; save exactly completed steps."""
    import numpy as np
    import torch
    original_batch = qat.Stream.batch
    original_save, original_load = torch.save, torch.load
    original_opt_load = qat.AdamWBF16.load
    pending_rng = {}

    def torch_save(obj, path, *args, **kwargs):
        path = Path(path)
        if not path.absolute().is_relative_to(WORK):
            return original_save(obj, path, *args, **kwargs)
        if path.name == "resume.tmp":
            obj = {**obj, "runtime_rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}}
        tmp = path.with_name(path.name + ".pending")
        original_save(obj, tmp, *args, **kwargs)
        with _real_open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
        fsync_dir(path.parent)

    def torch_load(path, *args, **kwargs):
        obj = original_load(path, *args, **kwargs)
        if isinstance(path, (str, Path)) and Path(path).name == "resume.pt" and isinstance(obj, dict):
            pending_rng.update(obj.get("runtime_rng", {}))
        return obj

    def opt_load(opt, state):
        original_opt_load(opt, state)
        if pending_rng:
            random.setstate(pending_rng["python"])
            np.random.set_state(pending_rng["numpy"])
            torch.set_rng_state(pending_rng["torch"])
            if pending_rng["cuda"]:
                torch.cuda.set_rng_state_all(pending_rng["cuda"])
            pending_rng.clear()

    def batch(stream, start, count):
        frame = inspect.currentframe().f_back
        # Only a direct call from the original train loop, not the monitor.
        if frame.f_code is qat.train.__code__ and frame.f_locals.get("k") == 0 and pause_requested():
            frame.f_locals["save_resume"]()
            checkpoint = qat.QAT / frame.f_locals["arm"] / "resume.pt"
            fsync_dir(checkpoint.parent)
            append_event({"event": "training_checkpoint_on_pause", "step": frame.f_locals["step"],
                          "file": str(checkpoint)})
            raise Paused("training checkpoint saved at an optimizer boundary")
        return original_batch(stream, start, count)

    torch.save, torch.load = torch_save, torch_load
    qat.AdamWBF16.load, qat.Stream.batch = opt_load, batch


def fix_intermediates(stage):
    recover_journal(WORK / "execution.jsonl")
    recover_journal(WORK / "data/teacher.jsonl")
    for path in (OUT / "gsm8k").glob("*.partial.jsonl"):
        recover_journal(path)
    if stage == "pack" and not (WORK / "data/data_record.json").exists():
        for name in ("kept_chats.json",):
            path = WORK / "data" / name
            if path.exists():
                try:
                    json.loads(path.read_text())
                except ValueError:
                    archive(path, "incomplete unsealed packing intermediate")
    if stage == "protocol" and not (OUT / "protocol.json").exists():
        audit = OUT / "contamination_cases.jsonl"
        if audit.exists():
            archive(audit, "interrupted before protocol publication; recompute audit against exact stream")
    if stage.startswith("export"):
        for path in WORK.glob("qat_chat_42-*.gguf"):
            record = OUT / f"export_{path.stem}.json"
            try:
                valid = json.loads(record.read_text())
            except (FileNotFoundError, ValueError):
                archive(path, "export interrupted before complete manifest")
                if record.exists():
                    archive(record, "incomplete export manifest")


def stage(name):
    check_pause()
    fix_intermediates(name)
    started = time.time()
    append_event({"event": "stage_started", "stage": name})
    outcome = "failed"
    try:
        if name in ("generate", "pack"):
            import prepare_qat08_chat as prep
            install_generation_controls()
            getattr(prep, name)()
        elif name == "protocol":
            import freeze_qat08_chat as freeze
            freeze.freeze_protocol()
        elif name == "train" or name.startswith("export-"):
            import qat_08b as qat
            import qat_08b_chat as entry
            if name == "train":
                install_training_controls(qat)
                sys.argv = ["qat_08b_chat.py", "train"]
            else:
                sys.argv = ["qat_08b_chat.py", "export", "--ckpt", name.split("-", 1)[1]]
            entry.main()
        else:
            import score_qat08_chat as score
            score.verify(check_data=False)
            install_generation_controls()
            if name in ("curve", "heldout"):
                import subprocess
                original = subprocess.run
                def run(*args, **kwargs):
                    check_pause()
                    return original(*args, **kwargs)
                subprocess.run = run
            getattr(score, name)()
        outcome = "complete"
    except Paused:
        outcome = "paused"
        raise
    finally:
        elapsed = time.time()-started
        append_event({"event": "stage_finished", "stage": name, "outcome": outcome, "seconds": elapsed})
        if name == "generate" and outcome == "complete":
            # Frozen builder expects total generation duration, including prior paused sessions.
            path = WORK / "data/teacher_timing.json"
            if path.exists():
                record = json.loads(path.read_text())
            else:
                # Last response could have been fsynced just before an abrupt reboot.
                from score_17b import sha
                record = {"responses_this_session": 0, "seconds": elapsed, "tokens": 0,
                    "pilot": False, "model_sha256": sha(ROOT / "work/qwen35_08b/base-f32.gguf"),
                    "prompt_sha256": sha(WORK / "data/prompts.jsonl"),
                    "thinking": "disabled in template (empty think block)",
                    "generation": "F32 FP GGUF, pinned llama-server, greedy top_k=1, temperature=0, 8 slots, 768 token limit",
                    "incident": "timing manifest reconstructed after response journal finished before publication"}
            events = [json.loads(l) for l in (WORK / "execution.jsonl").read_text().splitlines()]
            sessions = [r for r in events if r.get("event") == "stage_finished" and r.get("stage") == "generate"]
            record["seconds_last_session"] = record["seconds"]
            record["seconds"] = sum(r["seconds"] for r in sessions)
            record["execution_sessions"] = sessions
            durable_json(path, record)

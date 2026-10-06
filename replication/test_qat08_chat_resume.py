"""CPU lifecycle tests: original training loop, separate processes, exact restart parity.

Uses a tiny synthetic model and temporary data, never the research student/data.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def toy(mode, directory):
    import numpy as np
    import torch
    import qat_08b as qat
    import qat08_chat_runtime as runtime
    directory = Path(directory)
    runtime.WORK = directory / "work"
    runtime.OUT = directory / "results"
    runtime.PAUSE = directory / "PAUSE_REQUESTED"
    qat.QAT, qat.OUT, qat.DEV, qat.SEQ = runtime.WORK, runtime.OUT, "cpu", 8
    random.seed(321); np.random.seed(321); torch.manual_seed(321)

    class Student(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 2, bias=False)
            with torch.no_grad():
                self.proj.weight.copy_(torch.arange(8).reshape(2,4)*.03)
        def gradient_checkpointing_enable(self, **kwargs):
            pass

    def build(rule, latents=None):
        model = Student()
        if latents:
            model.proj.weight.data.copy_(latents["projection"])
        return model, ["projection"], None

    def loss(student, teacher, ids):
        # Deliberately consume three RNGs to exercise restart state, not only weights.
        x = ids.float().reshape(-1,4) / 100
        noise = torch.rand_like(x)*.01 + random.random()*.001 + np.random.random()*.001
        return (student.proj(x+noise)-.2).square().mean()

    qat.load_teacher = lambda: None
    qat.build_student, qat.distill_loss, qat.hf_name = build, loss, lambda name: "proj"
    qat.monitor_kl = lambda *args, **kwargs: 0.0
    runtime.install_file_controls()
    runtime.install_training_controls(qat)
    step = qat.AdamWBF16.step
    def opt_step(opt, lr):
        step(opt, lr)
        if mode == "pause" and opt.t == 2:
            runtime._requested = True
    qat.AdamWBF16.step = opt_step
    try:
        qat.train("toy")
    except runtime.Paused:
        return
    assert mode != "pause", "expected an optimizer-boundary pause"


def setup(directory):
    import numpy as np
    (directory / "work/data").mkdir(parents=True)
    (directory / "results").mkdir()
    np.arange(256, dtype=np.uint32).tofile(directory / "work/data/train.u32")
    np.arange(256, dtype=np.uint32).tofile(directory / "work/data/monitor.u32")
    cfg = {"steps": 5, "micro_batch": 2, "grad_accum": 2, "peak_lr": .01, "warmup_steps": 2,
           "betas": [.9,.95], "grad_clip": 1., "checkpoints": [[1.,"final"]],
           "monitor_every": 5, "resume_every_s": 1200}
    (directory / "results/protocol.json").write_text(json.dumps({"training": cfg, "arms": {"toy": {"rule": "top86"}}}))


def tests():
    import numpy as np
    import torch
    import qat08_chat_runtime as runtime
    with tempfile.TemporaryDirectory(prefix="qat08-chat-resume-", dir="/tmp") as name:
        base = Path(name)
        full, split = base / "full", base / "split"
        full.mkdir(); split.mkdir(); setup(full); setup(split)
        def run(mode, directory):
            subprocess.run([sys.executable, str(Path(__file__)), "--toy", mode, "--directory", str(directory)], check=True)
        run("full", full)
        run("pause", split)
        paused = torch.load(split / "work/toy/resume.pt", weights_only=False)
        assert paused["step"] == 2 and paused["opt"]["t"] == 2
        assert not (split / "work/toy/DONE").exists()
        run("resume", split)  # fresh process; no Python/GPU state carried over
        a = torch.load(full / "work/toy/resume.pt", weights_only=False)
        b = torch.load(split / "work/toy/resume.pt", weights_only=False)
        assert a["step"] == b["step"] == a["opt"]["t"] == b["opt"]["t"] == 5
        assert all(torch.equal(a["latents"][k],b["latents"][k]) for k in a["latents"])
        assert all(torch.equal(x,y) for key in ("m","v") for x,y in zip(a["opt"][key],b["opt"][key]))
        assert torch.equal(a["runtime_rng"]["torch"], b["runtime_rng"]["torch"])
        assert a["runtime_rng"]["python"] == b["runtime_rng"]["python"]
        assert np.array_equal(a["runtime_rng"]["numpy"][1],b["runtime_rng"]["numpy"][1])
        rows = lambda d: [json.loads(l) for l in (d / "work/toy/train.jsonl").read_text().splitlines() if "train_kl" in json.loads(l)]
        assert [r["train_kl"] for r in rows(full)] == [r["train_kl"] for r in rows(split)]

        runtime.WORK, runtime.OUT = base / "journals", base / "out"
        runtime.WORK.mkdir(); runtime.OUT.mkdir()
        journal = runtime.WORK / "teacher.jsonl"
        journal.write_bytes(b'{"id":1}\n{"id":2')
        runtime.recover_journal(journal)
        assert journal.read_bytes() == b'{"id":1}\n'
        assert list(journal.parent.glob("teacher.jsonl.interrupted-*"))
        journal.write_bytes(b'{bad}\n{"id":2}\n')
        try:
            runtime.recover_journal(journal)
        except RuntimeError:
            pass
        else:
            raise AssertionError("must reject interior corruption")
        runtime.install_file_controls()
        frozen = runtime.OUT / "freeze.json"
        with frozen.open("x") as f:
            f.write('{"fixed":true}\n')
        try:
            with frozen.open("x") as f:
                f.write('{"fixed":false}\n')
        except FileExistsError:
            pass
        else:
            raise AssertionError("must not overwrite an existing freeze")
        assert frozen.read_text() == '{"fixed":true}\n'
    result = {"passed": True, "training": "original QAT08 loop on a tiny CPU fixture; 5 steps uninterrupted equals 2 steps + fresh-process restart + 3 steps bit-for-bit in weights, BF16 moments, losses, optimizer step and RNG state",
              "journal": "incomplete final record repaired with original preserved; interior corruption rejected",
              "freeze": "exclusive atomic publication refuses overwrite and preserves original bytes",
              "scope": "synthetic CPU lifecycle test only; no research model training or held-out scoring"}
    path = ROOT / "results/qat08_chat/resume_test_results.json"
    if path.exists():
        raise SystemExit("test record already exists; preserve it and write a new version if retesting")
    runtime.durable_json(path, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--toy", choices=["full", "pause", "resume"])
    parser.add_argument("--directory")
    args = parser.parse_args()
    toy(args.toy, args.directory) if args.toy else tests()

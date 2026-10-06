"""Isolated entrypoint for the unchanged QAT08 trainer and exporter.

No changes to quantization, initialization, loss, optimization or checkpoints.
Training additionally requires a recorded user go-ahead bound to the frozen hash.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import qat_08b as qat
from score_17b import sha

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/qat08_chat"
WORK = ROOT / "work/qat08_chat"
ARM = "qat_chat_42"


def verify(check_data=True):
    p = json.loads((OUT / "protocol.json").read_text())
    assert platform.python_version() == p["environment"]["python"]
    for name, version in p["environment"]["packages"].items():
        assert importlib.metadata.version(name) == version, f"frozen package version changed: {name}"
    assert p["training"] == json.loads((ROOT / "results/qat08/protocol.json").read_text())["training"]
    for name, digest in p["implementation_sha256"].items():
        assert sha(ROOT / name) == digest, f"frozen dependency changed: {name}"
    for name, digest in p["input_sha256"].items():
        assert sha(ROOT / name) == digest, f"frozen input changed: {name}"
    for name, digest in p["runtime_sha256"].items():
        assert sha(ROOT / name) == digest, f"frozen runtime changed: {name}"
    if check_data:
        for name, digest in p["data_sha256"].items():
            assert sha(WORK / "data" / name) == digest, f"frozen data changed: {name}"
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "export", "selftest", "verify"])
    ap.add_argument("--ckpt", default="final")
    a = ap.parse_args()
    if a.cmd == "selftest":
        qat.selftest()
        return
    verify(check_data=a.cmd in ("train", "verify"))
    qat.OUT, qat.QAT = OUT, WORK
    if a.cmd == "verify":
        print("frozen recipe, code, models and data verified")
    elif a.cmd == "train":
        approval = json.loads((OUT / "training_approval.json").read_text())
        assert approval["protocol_sha256"] == sha(OUT / "protocol.json")
        assert approval["design_sha256"] == sha(OUT / "design.json")
        assert approval["approved_by_user"] is True
        run = WORK / ARM
        run.mkdir(parents=True, exist_ok=True)
        binding = run / "binding.json"
        expected = {"protocol_sha256": sha(OUT / "protocol.json"), "arm": ARM}
        if binding.exists():
            assert json.loads(binding.read_text()) == expected, "resume belongs to a different experiment"
        else:
            assert not (run / "resume.pt").exists()
            with binding.open("x") as f:
                json.dump(expected, f)
        if (run / "DONE").exists():
            print("training already complete")
            return
        qat.train(ARM)
    else:
        target = WORK / f"{ARM}-{a.ckpt}.gguf"
        if target.exists():
            record = json.loads((OUT / f"export_{ARM}-{a.ckpt}.json").read_text())
            assert sha(target) == record["sha256"]
            return
        qat.export(ARM, a.ckpt)


if __name__ == "__main__":
    main()

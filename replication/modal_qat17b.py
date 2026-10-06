"""Run the 1.7B profile (qat_17b.py) on a Modal GPU. Measurement only; no training run.

    ../../../.venv-modal/bin/modal run modal_qat17b.py              # upload, fetch base, profile on H100
    ../../../.venv-modal/bin/modal run modal_qat17b.py --gpu A100-80GB
    ../../../.venv-modal/bin/modal run modal_qat17b.py --fetch      # download saved results only

The pinned Qwen3-1.7B is downloaded inside Modal and checked file-by-file against the local
checksums from `qat_17b.py prep`; the profile result lands in work/qat17b/ locally.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
LOCAL_DATA = HERE / "work/qat17b"
LOCAL_WORK = HERE / "work/qwen3_17b"
REMOTE = "/root/analysis/bonsai2/replication"
VOL = "/vol"
CODE = ["qat_08b.py", "qat_17b.py", "make_gptq_pilot.py", "codec_probe.py", "phase0.py"]

volume = modal.Volume.from_name("bonsai-qat17b", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "torch==2.14.0", "transformers==5.5.4", "numpy==2.4.6", "safetensors==0.8.0", "huggingface_hub==1.8.0")
for name in CODE:
    image = image.add_local_file(HERE / name, f"{REMOTE}/{name}")
image = image.add_local_file(HERE.parent / "inspect_gguf.py", "/root/analysis/bonsai2/inspect_gguf.py")
env = {"QAT17_WORK": f"{VOL}/qwen3_17b", "QAT17_DATA": f"{VOL}/qat17b", "PYTHONPATH": REMOTE}
app = modal.App("bonsai-qat17b", image=image)


@app.function(volumes={VOL: volume}, timeout=1800, cpu=4, memory=8192)
def fetch_base(revision: str, expected: dict[str, str]) -> dict:
    import hashlib

    from huggingface_hub import snapshot_download

    base = Path(VOL) / "qwen3_17b/base"
    snapshot_download("Qwen/Qwen3-1.7B", revision=revision, local_dir=base, allow_patterns=list(expected))
    got = {}
    for name in expected:
        h = hashlib.sha256()
        with (base / name).open("rb") as f:
            for block in iter(lambda: f.read(8 << 20), b""):
                h.update(block)
        got[name] = h.hexdigest()
    bad = {n for n in expected if got[n] != expected[n]}
    if bad:
        raise RuntimeError(f"checksum mismatch: {sorted(bad)}")
    volume.commit()
    return {"files": len(got), "verified": True}


def _run_profile(steps: int) -> str:
    """Return JSON text, never torch objects or exceptions, so the local client needs no torch."""
    import os
    import sys
    import traceback

    os.environ.update(env)
    sys.path.insert(0, REMOTE)
    os.chdir(REMOTE)
    try:
        import qat_17b

        result = qat_17b.profile("top86", qat_17b.DEFAULT_CONFIGS, steps, 32)
    except BaseException:
        raise RuntimeError(traceback.format_exc()) from None
    text = json.dumps(result, indent=2) + "\n"
    (Path(env["QAT17_DATA"]) / f"profile_{result['gpu'].replace(' ', '_')}.json").write_text(text)
    volume.commit()
    return text


@app.function(volumes={VOL: volume}, gpu="H100", timeout=1800, memory=65536)
def profile_h100(steps: int) -> str:
    return _run_profile(steps)


@app.function(volumes={VOL: volume}, gpu="A100-80GB", timeout=1800, memory=65536)
def profile_a100(steps: int) -> str:
    return _run_profile(steps)


@app.local_entrypoint()
def main(gpu: str = "H100", steps: int = 3, fetch: bool = False) -> None:
    if fetch:  # download profile results already on the volume; no GPU
        for entry in volume.listdir("/qat17b"):
            name = Path(entry.path).name
            if name.startswith("profile_") and name.endswith(".json"):
                (LOCAL_DATA / name).write_bytes(b"".join(volume.read_file(entry.path)))
                print("saved", LOCAL_DATA / name)
        return
    record = json.loads((LOCAL_DATA / "prep_record.json").read_text())
    with volume.batch_upload(force=True) as up:
        for name in ("names.json", "profile.u32", "prep_record.json"):
            up.put_file(LOCAL_DATA / name, f"/qat17b/{name}")
        up.put_file(LOCAL_WORK / "folded/hadamard_packing.json", "/qwen3_17b/folded/hadamard_packing.json")
    print("uploaded profile inputs")
    print("base:", fetch_base.remote(record["ancestor"].split("@")[1], record["base_sha256"]))
    fn = {"H100": profile_h100, "A100-80GB": profile_a100}[gpu]
    result = json.loads(fn.remote(steps))
    path = LOCAL_DATA / f"profile_{result['gpu'].replace(' ', '_')}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "configs"}, indent=2))
    print("saved", path)

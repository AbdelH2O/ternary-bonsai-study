"""Run the frozen exploratory matrix, one CUDA model and server slot at a time."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
URL = "http://127.0.0.1:18082"
ORIGINAL = ROOT / "models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf"


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_freeze(freeze):
    files = {
        ORIGINAL: freeze["source_model_sha256"],
        BASE / "prompts/calibration_v2.jsonl": freeze["calibration_v2_sha256"],
        BASE / "prompts/primary.jsonl": freeze["primary_prompts_sha256"],
        BASE / "benchmark/selected.jsonl": freeze["longbench_selected_sha256"],
    }
    for arm in freeze["arms"]:
        if arm["name"] != "baseline":
            files[BASE / "variants" / f"bonsai2-{arm['name']}.json"] = arm["patch_manifest_sha256"]
    for path, expected in files.items():
        if sha256(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")


def loaded_model():
    try:
        with urllib.request.urlopen(URL + "/props", timeout=2) as response:
            return json.load(response).get("model_path")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def output_path(stage: str, arm: str):
    suffix = "-logprobs" if stage == "primary" else ""
    return BASE / "runs" / f"{stage}{suffix}-{arm}.jsonl"


def completed_time(arm):
    total = 0.0
    for stage in ("primary", "longbench"):
        path = output_path(stage, arm)
        if path.exists():
            total += sum(json.loads(line).get("elapsed_seconds", 0) for line in path.open())
    return total


def run_arm(arm: str, freeze):
    model = ORIGINAL if arm == "baseline" else BASE / "variants" / f"bonsai2-{arm}.gguf"
    model = model.resolve()
    if loaded_model():
        raise RuntimeError(f"Port 18082 already serves {loaded_model()}; stop it first")
    env = dict(os.environ, BONSAI_FAMILY="bonsai2", BONSAI_GGUF=str(model),
               BONSAI_NGL="99", BONSAI_CTX="16384", BONSAI_SPECULATIVE="0",
               BONSAI_KV4="0", PORT="18082")
    launch = [str(ROOT / "scripts/start_llama_server.sh"), "--parallel", "1",
              "-b", "512", "-ub", "512", "--reasoning", "off"]
    log_path = BASE / "runs" / f"server-matrix-{arm}.log"
    with log_path.open("w") as log:
        server = subprocess.Popen(launch, cwd=ROOT, env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited {server.returncode}; see {log_path}")
                if loaded_model() == str(model):
                    break
                time.sleep(0.5)
            else:
                raise TimeoutError(f"Server load timed out; see {log_path}")
            for stage in ("primary", "longbench"):
                remaining = freeze["run_limit_seconds_per_arm"] - completed_time(arm)
                if remaining <= 0:
                    print(f"Declared time limit reached for {arm}", flush=True)
                    break
                command = [sys.executable, str(BASE / "run_experiment.py"), stage, arm,
                           "--expected-model", str(model), "--url", URL,
                           "--max-seconds", str(max(1, int(remaining)))]
                if stage == "primary":
                    command.append("--logprobs")
                print(f"Running {stage} {arm}, {remaining:.0f}s remaining", flush=True)
                subprocess.run(command, cwd=ROOT, check=True)
                path = output_path(stage, arm)
                rows = [json.loads(line) for line in path.open()]
                if any("out of memory" in str(row.get("error", "")).lower() or
                       "out of memory" in str(row.get("error_body", "")).lower() for row in rows):
                    raise MemoryError(f"OOM during {stage} {arm}; see {path}")
        finally:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


def main():
    freeze = json.loads((BASE / "PRIMARY_FREEZE.json").read_text())
    verify_freeze(freeze)
    (BASE / "runs").mkdir(exist_ok=True)
    for arm in freeze["arms"]:
        name = arm["name"]
        print(f"Starting frozen arm {name}", flush=True)
        run_arm(name, freeze)


if __name__ == "__main__":
    main()

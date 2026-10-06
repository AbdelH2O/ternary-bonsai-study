"""Sequentially run the frozen short calibration set on all six GPU candidates."""

from __future__ import annotations

import argparse
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


def loaded_model():
    try:
        with urllib.request.urlopen(URL + "/props", timeout=2) as response:
            return json.load(response).get("model_path")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def run_one(arm: str, delta: int, logprobs: bool = False, stage: str = "calibration"):
    path = (BASE / "variants" / f"bonsai2-{arm}-d{delta:02d}.gguf").resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if loaded_model():
        raise RuntimeError(f"Port 18082 already serves {loaded_model()}; stop it first")
    env = dict(os.environ, BONSAI_FAMILY="bonsai2", BONSAI_GGUF=str(path),
               BONSAI_NGL="99", BONSAI_CTX="16384", BONSAI_SPECULATIVE="0",
               BONSAI_KV4="0", PORT="18082")
    command = [str(ROOT / "scripts/start_llama_server.sh"), "--parallel", "1",
               "-b", "512", "-ub", "512", "--reasoning", "off"]
    log_path = BASE / "runs" / f"server-{stage}-{arm}-d{delta:02d}.log"
    with log_path.open("w") as log:
        server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited {server.returncode}; see {log_path}")
                if loaded_model() == str(path):
                    break
                time.sleep(0.5)
            else:
                raise TimeoutError(f"Server load timed out; see {log_path}")
            arm_name = f"{arm}-d{delta:02d}"
            run_command = [sys.executable, str(BASE / "run_experiment.py"),
                           stage, arm_name, "--expected-model", str(path),
                           "--url", URL]
            if logprobs:
                run_command.append("--logprobs")
            subprocess.run(run_command, cwd=ROOT, check=True)
        finally:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deltas", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--arms", choices=["state", "mlp"], nargs="+", default=["state", "mlp"])
    parser.add_argument("--logprobs", action="store_true")
    parser.add_argument("--stage", choices=["calibration", "calibration_v2"], default="calibration")
    args = parser.parse_args()
    (BASE / "runs").mkdir(exist_ok=True)
    for delta in args.deltas:
        for arm in args.arms:
            print(f"Calibrating {arm}-d{delta:02d}", flush=True)
            run_one(arm, delta, args.logprobs, args.stage)


if __name__ == "__main__":
    main()

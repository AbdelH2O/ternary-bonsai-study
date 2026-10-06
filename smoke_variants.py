"""CPU-only one-question loading smoke test for locally patched checkpoints."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


BASE = Path(__file__).resolve().parent
BIN = Path("bin/cuda/llama-server").resolve()
URL = "http://127.0.0.1:18081"


def get_json(path: str, payload=None, timeout=10):
    req = urllib.request.Request(URL + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def smoke(path: Path):
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(BIN.parent) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    command = [str(BIN), "-m", str(path.resolve()), "--host", "127.0.0.1", "--port", "18081",
               "-ngl", "0", "-c", "1024", "--parallel", "1", "-b", "256", "-ub", "256",
               "-fa", "on", "--jinja", "--reasoning", "off"]
    log_path = BASE / "variants" / f"{path.stem}.smoke.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=Path.cwd(), env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited {process.returncode}; see {log_path}")
                try:
                    props = get_json("/props", timeout=2)
                    if props.get("model_path") == str(path.resolve()):
                        break
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                    pass
                time.sleep(0.5)
            else:
                raise TimeoutError(f"Server load timed out; see {log_path}")
            response = get_json("/v1/chat/completions", {
                "model": path.name, "messages": [{"role": "user", "content":
                    "What is 2 + 2? Reply with only the numeral."}],
                "temperature": 0, "max_tokens": 16, "stream": False}, timeout=180)
            answer = response["choices"][0]["message"]["content"].strip()
            return {"candidate": str(path), "answer": answer, "sensible": answer == "4",
                    "timings": response.get("timings"), "log": str(log_path)}
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main():
    paths = sorted((BASE / "variants").glob("*.gguf"))
    if not paths:
        raise FileNotFoundError("No candidate GGUFs")
    output = BASE / "variants/smoke.jsonl"
    for path in paths:
        result = smoke(path)
        with output.open("a") as file:
            file.write(json.dumps(result) + "\n")
        print(path.name, result["answer"], "PASS" if result["sensible"] else "FAIL", flush=True)
        if not result["sensible"]:
            raise ValueError(f"Smoke test did not produce expected answer: {path}")


if __name__ == "__main__":
    main()

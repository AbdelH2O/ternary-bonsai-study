"""Run the frozen corrected-retrieval subset against the original model only."""

import hashlib
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
URL = "http://127.0.0.1:18083"
MODEL = ROOT / "models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request(path, payload=None, timeout=120):
    req = urllib.request.Request(URL + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def main():
    frozen = json.loads((HERE / "DOSE_FREEZE.json").read_text())
    pilot = frozen["retrieval_pilot"]
    prompts_path = ROOT / pilot["prompt_file"]
    if sha(prompts_path) != pilot["prompt_file_sha256"] or sha(MODEL) != frozen["baseline_model_sha256"]:
        raise ValueError("Frozen prompt or baseline model changed")
    by_id = {r["id"]: r for r in (json.loads(line) for line in prompts_path.read_text().splitlines())}
    if any(item not in by_id for item in pilot["item_ids"]):
        raise ValueError("Frozen retrieval ID missing")
    out = HERE / "retrieval_baseline_pilot.jsonl"
    done = {r["id"] for r in (json.loads(line) for line in out.read_text().splitlines())} if out.exists() else set()
    try:
        request("/props", timeout=2)
        raise RuntimeError(f"Port {URL} already has a server")
    except (urllib.error.URLError, TimeoutError):
        pass
    env = dict(os.environ, BONSAI_FAMILY="bonsai2", BONSAI_GGUF=str(MODEL),
               BONSAI_NGL="99", BONSAI_CTX="16384", BONSAI_SPECULATIVE="0",
               BONSAI_KV4="0", PORT="18083")
    log_path = HERE / "retrieval_baseline_server.log"
    with log_path.open("w") as server_log:
        server = subprocess.Popen([str(ROOT / "scripts/start_llama_server.sh"),
                                   "--parallel", "1", "-b", "512", "-ub", "512", "--reasoning", "off"],
                                  cwd=ROOT, env=env, stdout=server_log, stderr=server_log)
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited {server.returncode}: {log_path}")
                try:
                    props = request("/props", timeout=2)
                    if props.get("model_path") == str(MODEL):
                        break
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                    pass
                time.sleep(0.5)
            else:
                raise TimeoutError(f"Baseline model load timed out: {log_path}")
            for index, item in enumerate(pilot["item_ids"], 1):
                if item in done:
                    continue
                row = by_id[item]
                body = {"model": MODEL.name, "messages": [{"role": "user", "content": row["prompt"]}],
                        "temperature": 0, "top_p": 1, "max_tokens": 32, "seed": 20260929,
                        "cache_prompt": False, "stream": False, "thinking_budget_tokens": 0}
                result = {"id": item, "registry_id": row["registry_id"],
                    "question_id": row["question_id"], "layout": row["layout"], "swap": row["swap"],
                    "prompt_sha256": row["prompt_sha256"], "chat_sha256": row["chat_sha256"],
                    "model_sha256": frozen["baseline_model_sha256"],
                    "expected_prompt_tokens": row["prompt_tokens"], "gold": row["answer"]}
                began = time.monotonic()
                try:
                    response = request("/v1/chat/completions", body)
                    choice = response["choices"][0]
                    answer = choice["message"].get("content") or ""
                    codes = re.findall(r"K-\d{6}", answer)
                    result.update({"response": answer, "extracted_code": codes[0] if len(codes) == 1 else None,
                        "exact_correct": answer.strip() == row["answer"],
                        "extracted_correct": len(codes) == 1 and codes[0] == row["answer"],
                        "finish_reason": choice.get("finish_reason"),
                        "reasoning_content": choice["message"].get("reasoning_content"),
                        "prompt_tokens": response.get("usage", {}).get("prompt_tokens"),
                        "completion_tokens": response.get("usage", {}).get("completion_tokens"),
                        "timings": response.get("timings"), "error": None})
                    if result["prompt_tokens"] is not None and result["prompt_tokens"] > 14384:
                        raise ValueError("Prompt exceeded reserved input budget")
                except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
                    result.update({"error": str(exc), "exact_correct": None, "extracted_correct": None})
                    if isinstance(exc, urllib.error.HTTPError):
                        result["error_body"] = exc.read(2000).decode(errors="replace")
                result["elapsed_seconds"] = time.monotonic() - began
                with out.open("a") as stream:
                    stream.write(json.dumps(result) + "\n")
                done.add(item)
                print(f"{index}/{len(pilot['item_ids'])} {item}: exact={result['exact_correct']} error={result['error']}", flush=True)
                if result["error"]:
                    raise RuntimeError(f"Pilot request failed: {item}: {result['error']}")
        finally:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()

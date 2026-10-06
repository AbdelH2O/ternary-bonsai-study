"""Run frozen retrieval IDs against each matched-dose perturbation on host CUDA."""

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
URL = "http://127.0.0.1:18084"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request(path, payload=None, timeout=120):
    req = urllib.request.Request(URL + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def run_variant(variant, ids, rows, done, freeze):
    name = variant["name"]
    model = ROOT / variant["model_path"]
    if sha(model) != variant["model_sha256"]:
        raise ValueError(f"Model hash mismatch: {name}")
    patch = ROOT / variant["patch_manifest_path"]
    if sha(patch) != variant["patch_manifest_sha256"]:
        raise ValueError(f"Patch manifest hash mismatch: {name}")
    try:
        request("/props", timeout=2)
        raise RuntimeError(f"Port {URL} already has a server")
    except (urllib.error.URLError, TimeoutError):
        pass
    env = dict(os.environ, BONSAI_FAMILY="bonsai2", BONSAI_GGUF=str(model),
               BONSAI_NGL="99", BONSAI_CTX="16384", BONSAI_SPECULATIVE="0",
               BONSAI_KV4="0", PORT="18084")
    log_path = HERE / f"retrieval_variant_{name}.log"
    with log_path.open("w") as server_log:
        server = subprocess.Popen([str(ROOT / "scripts/start_llama_server.sh"),
                                   "--parallel", "1", "-b", "512", "-ub", "512",
                                   "--reasoning", "off"], cwd=ROOT, env=env,
                                  stdout=server_log, stderr=server_log)
        try:
            deadline = time.monotonic() + 150
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited {server.returncode}: {log_path}")
                try:
                    if request("/props", timeout=2).get("model_path") == str(model):
                        break
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                    pass
                time.sleep(0.5)
            else:
                raise TimeoutError(f"Model load timed out: {log_path}")
            for index, item in enumerate(ids, 1):
                if (name, item) in done:
                    continue
                row = rows[item]
                body = {"model": model.name,
                        "messages": [{"role": "user", "content": row["prompt"]}],
                        "temperature": 0, "top_p": 1, "max_tokens": 32,
                        "seed": freeze["generation"]["seed"],
                        "cache_prompt": False, "stream": False,
                        "thinking_budget_tokens": 0}
                result = {"variant": name, "arm": variant["arm"], "seed": variant["seed"],
                    "id": item, "registry_id": row["registry_id"],
                    "question_id": row["question_id"], "layout": row["layout"],
                    "swap": row["swap"], "prompt_sha256": row["prompt_sha256"],
                    "chat_sha256": row["chat_sha256"],
                    "model_sha256": variant["model_sha256"],
                    "patch_manifest_sha256": variant["patch_manifest_sha256"],
                    "expected_prompt_tokens": row["prompt_tokens"], "gold": row["answer"]}
                began = time.monotonic()
                try:
                    response = request("/v1/chat/completions", body)
                    choice = response["choices"][0]
                    answer = choice["message"].get("content") or ""
                    codes = re.findall(r"K-\d{6}", answer)
                    result.update({"response": answer,
                        "extracted_code": codes[0] if len(codes) == 1 else None,
                        "exact_correct": answer.strip() == row["answer"],
                        "extracted_correct": len(codes) == 1 and codes[0] == row["answer"],
                        "finish_reason": choice.get("finish_reason"),
                        "reasoning_content": choice["message"].get("reasoning_content"),
                        "prompt_tokens": response.get("usage", {}).get("prompt_tokens"),
                        "completion_tokens": response.get("usage", {}).get("completion_tokens"),
                        "timings": response.get("timings"), "error": None})
                    if result["prompt_tokens"] != row["prompt_tokens"]:
                        raise ValueError("Server/frozen chat token count mismatch")
                    if result["prompt_tokens"] > 14384:
                        raise ValueError("Prompt exceeded reserved input budget")
                except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
                        ValueError, KeyError) as exc:
                    result.update({"error": str(exc), "exact_correct": None,
                                   "extracted_correct": None})
                    if isinstance(exc, urllib.error.HTTPError):
                        result["error_body"] = exc.read(2000).decode(errors="replace")
                result["elapsed_seconds"] = time.monotonic() - began
                with (HERE / "retrieval_perturbed_pilot.jsonl").open("a") as stream:
                    stream.write(json.dumps(result) + "\n")
                done.add((name, item))
                print(f"{name} {index}/{len(ids)} {item}: exact={result['exact_correct']} error={result['error']}", flush=True)
                if result["error"]:
                    raise RuntimeError(f"Pilot request failed: {name} {item}: {result['error']}")
        finally:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


def main():
    freeze = json.loads((HERE / "PERTURBED_RETRIEVAL_PILOT_FREEZE.json").read_text())
    if sha(HERE / "DOSE_FREEZE.json") != freeze["source_dose_freeze_sha256"]:
        raise ValueError("Dose freeze changed")
    if sha(HERE / "retrieval_baseline_pilot.jsonl") != freeze["source_baseline_responses_sha256"]:
        raise ValueError("Baseline pilot changed")
    prompts = ROOT / freeze["prompt_file"]
    if sha(prompts) != freeze["prompt_file_sha256"]:
        raise ValueError("Prompt file changed")
    rows = {r["id"]: r for r in map(json.loads, prompts.read_text().splitlines())}
    ids = freeze["item_ids"]
    if len(ids) != len(set(ids)) or any(item not in rows for item in ids):
        raise ValueError("Missing or duplicate frozen ID")
    out = HERE / "retrieval_perturbed_pilot.jsonl"
    prior = [json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []
    done = {(r["variant"], r["id"]) for r in prior}
    if len(done) != len(prior):
        raise ValueError("Duplicate saved output")
    for variant in freeze["variants"]:
        if all((variant["name"], item) in done for item in ids):
            continue
        run_variant(variant, ids, rows, done, freeze)


if __name__ == "__main__":
    main()

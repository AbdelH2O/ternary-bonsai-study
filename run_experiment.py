"""Run frozen items against one explicitly launched Prism llama-server arm.

Start a single-slot, text-only server first. This script records every response
and failure as JSONL and resumes without repeating completed item IDs.
"""

from __future__ import annotations

import argparse
import json
import re
import string
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "runs"
SEED = 20260929
MAX_PROMPT_TOKENS = 14384


def read_json(url: str, payload: dict | None = None, timeout: int = 60):
    request = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def normalize(text: str) -> list[str]:
    # Matches LongBench/LongBench/metrics.py's English QA normalization.
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split()).split()


def qa_f1(prediction: str, answer: str) -> float:
    pred, gold = normalize(prediction), normalize(answer)
    common = Counter(pred) & Counter(gold)
    matched = sum(common.values())
    if not matched:
        return 0.0
    precision, recall = matched / len(pred), matched / len(gold)
    return 2 * precision * recall / (precision + recall)


def cases(stage: str):
    if stage == "longbench":
        path = BASE / "benchmark/selected.jsonl"
    elif stage == "calibration_v2":
        path = BASE / "prompts/calibration_v2.jsonl"
    elif stage == "calibration":
        path = BASE / "prompts/calibration.jsonl"
    else:
        path = BASE / "prompts/primary.jsonl"
    rows = [json.loads(line) for line in path.open()]
    if stage == "pilot":
        rows = [row for row in rows if row["pair_id"] in ("primary-00", "primary-01")]
    return rows


def run(args):
    props = read_json(args.url + "/props")
    if props.get("model_path") != str(args.expected_model.resolve()):
        raise ValueError(f"Server loaded {props.get('model_path')}, expected {args.expected_model.resolve()}")
    rows = cases(args.stage)
    RESULTS.mkdir(exist_ok=True)
    suffix = "-logprobs" if args.logprobs else ""
    out = RESULTS / f"{args.stage}{suffix}-{args.arm}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in out.open()}
    started = time.monotonic()
    for row in rows:
        if row["id"] in done:
            continue
        if args.max_items and len(done) >= args.max_items:
            break
        if args.max_seconds and time.monotonic() - started >= args.max_seconds:
            break
        max_tokens = (128 if row.get("task") == "qasper_e" else
                      32 if row.get("task") == "2wikimqa_e" else 64)
        body = {"model": args.expected_model.name, "messages": [{"role": "user", "content": row["prompt"]}],
                "temperature": 0, "top_p": 1, "max_tokens": max_tokens, "seed": SEED,
                "cache_prompt": False, "stream": False, "thinking_budget_tokens": 0}
        if args.logprobs:
            body["logprobs"] = True
        result = {"id": row["id"], "stage": args.stage, "variant": args.arm,
                  "seed": SEED, "expected_model": str(args.expected_model.resolve()),
                  "answer": row.get("answer", row.get("answers")),
                  "length": row.get("length", row.get("length_bin")),
                  "expected_prompt_tokens": row.get("prompt_tokens")}
        began = time.monotonic()
        try:
            response = read_json(args.url + "/v1/chat/completions", body, args.timeout)
            choice = response["choices"][0]
            answer = choice["message"].get("content") or ""
            result.update({"response": answer, "finish_reason": choice.get("finish_reason"),
                           "reasoning_content": choice["message"].get("reasoning_content"),
                           "prompt_tokens": response.get("usage", {}).get("prompt_tokens"),
                           "completion_tokens": response.get("usage", {}).get("completion_tokens"),
                           "timings": response.get("timings"), "elapsed_seconds": time.monotonic() - began,
                           "score": (float(answer.strip() == row["answer"]) if "answer" in row else
                                     max(qa_f1(answer, gold) for gold in row["answers"])),
                           "error": None})
            if args.logprobs:
                token_rows = (choice.get("logprobs") or {}).get("content") or []
                answer_tokens = [token for token in token_rows if token.get("bytes")]
                reconstructed = bytes(byte for token in answer_tokens for byte in token["bytes"]).decode("utf-8", errors="replace")
                result["answer_logprob"] = (sum(token["logprob"] for token in answer_tokens)
                                            if reconstructed == row.get("answer") else None)
                result["answer_token_count"] = len(answer_tokens)
                result["logprob_reconstructed"] = reconstructed
                result["logprob_valid"] = reconstructed == row.get("answer")
            if result["prompt_tokens"] is not None and result["prompt_tokens"] > MAX_PROMPT_TOKENS:
                raise ValueError(f"Prompt exceeded token budget: {result['prompt_tokens']}")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
            result.update({"error": str(exc), "elapsed_seconds": time.monotonic() - began,
                           "score": None})
            if isinstance(exc, urllib.error.HTTPError):
                result["error_body"] = exc.read(2000).decode(errors="replace")
        with out.open("a") as stream:
            stream.write(json.dumps(result) + "\n")
        done.add(row["id"])
        print(f"{row['id']} score={result['score']} error={result['error']}", flush=True)
        if result["error"] and ("out of memory" in result["error"].lower() or
                                "out of memory" in result.get("error_body", "").lower()):
            break
    print(f"Wrote {out}; completed {len(done)}/{len(rows)} items")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["calibration", "calibration_v2", "pilot", "primary", "longbench"])
    parser.add_argument("arm", help="e.g. baseline, state-d03, mlp-d03")
    parser.add_argument("--expected-model", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--logprobs", action="store_true", help="Record correct generated-answer log probability when verifiable")
    parser.add_argument("--max-items", type=int, default=0, help="Declared run limit; zero means all")
    parser.add_argument("--max-seconds", type=int, default=0, help="Declared run limit; zero means unlimited")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

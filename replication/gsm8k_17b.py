"""GSM8K greedy generation through the pinned llama-server, for the frozen 1.7B comparison.

    python3 gsm8k_17b.py timing MODEL.gguf [none]    # pre-freeze throughput check on 64 train items
"""

from __future__ import annotations

import contextlib
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SERVER = REPO / "bin/cuda/llama-server"
PORT = 8091
SLOTS = 8
N_PREDICT = 1024


@contextlib.contextmanager
def serve(model: Path, rope: str, log: Path):
    cmd = [str(SERVER), "-m", str(model), "-ngl", "99", "-c", str(SLOTS * 2048), "-np", str(SLOTS),
           "--host", "127.0.0.1", "--port", str(PORT), "--no-webui"]
    if rope == "none":
        cmd += ["--rope-scaling", "none"]
    with log.open("w") as err:
        proc = subprocess.Popen(cmd, stdout=err, stderr=subprocess.STDOUT)
        try:
            for _ in range(600):
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                        if json.loads(r.read()).get("status") == "ok":
                            break
                except Exception:
                    if proc.poll() is not None:
                        raise RuntimeError(f"llama-server exited; see {log}")
                    time.sleep(0.5)
            else:
                raise RuntimeError("llama-server did not become healthy")
            yield
        finally:
            proc.terminate()
            proc.wait(timeout=60)


def _post(prompt_ids: list[int], stream: bool):
    body = json.dumps({"prompt": prompt_ids, "n_predict": N_PREDICT, "temperature": 0.0, "top_k": 1,
                       "cache_prompt": False, "stream": stream}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", body, {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=1800)


def complete(prompt_ids: list[int]) -> dict:
    try:
        with _post(prompt_ids, False) as r:
            out = json.loads(r.read())
        return {"text": out["content"], "tokens": out.get("tokens_predicted"), "stop_type": out.get("stop_type")}
    except urllib.error.HTTPError as e:
        if e.code != 500 or b"does not match the expected" not in e.read():
            raise
    # The server parses the finished text into a chat message even for raw /completion and throws when
    # it does not fit; stream chunks carry the raw token text, so regenerate and keep that instead.
    text, tokens, stop_type = [], None, "parse_error"
    with _post(prompt_ids, True) as r:
        for line in r:
            line = line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            chunk = json.loads(line[6:])
            if "error" in chunk:
                break
            text.append(chunk.get("content", ""))
            if chunk.get("stop"):
                tokens, stop_type = chunk.get("tokens_predicted"), chunk.get("stop_type")
    return {"text": "".join(text), "tokens": tokens, "stop_type": stop_type, "parse_fallback": True}


def generate(items: list[dict]) -> list[dict]:
    with ThreadPoolExecutor(SLOTS) as pool:
        return list(pool.map(lambda it: {"id": it["id"], **complete(it["prompt_ids"])}, items))


def number(text: str) -> str | None:
    text = text.replace(",", "").replace("$", "").strip()
    m = re.findall(r"-?\d+(?:\.\d+)?", text)
    return m[-1] if m else None


def parse(text: str) -> str | None:
    boxed = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text)
    return number(boxed[-1]) if boxed and number(boxed[-1]) is not None else number(text)


def correct(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred.rstrip(".")) - float(gold)) < 1e-6
    except ValueError:
        return False


def timing(model: str, rope: str) -> None:
    import io
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(Path(__file__).resolve().parent / "work/qwen3_17b/base")
    url = ("https://huggingface.co/datasets/openai/gsm8k/resolve/740312add88f781978c0658806c59bc2815b9866/"
           "main/train-00000-of-00001.parquet")
    with urllib.request.urlopen(url) as r:
        rows = pq.read_table(io.BytesIO(r.read())).to_pylist()[:64]
    items = []
    for i, row in enumerate(rows):
        text = f"{row['question']}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        prompt = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        items.append({"id": i, "gold": row["answer"].split("####")[-1].strip().replace(",", ""),
                      "prompt_ids": tok(prompt, add_special_tokens=False)["input_ids"]})
    with serve(Path(model), rope, Path(model).with_suffix(".timing.log")):
        t0 = time.time()
        outs = generate(items)
        dt = time.time() - t0
    acc = sum(correct(parse(o["text"]), it["gold"]) for o, it in zip(outs, items)) / len(items)
    toks = sum(o["tokens"] for o in outs)
    print(f"{len(items)} train items in {dt:.1f}s; {toks} tokens ({toks / dt:.0f} tok/s); "
          f"train accuracy {acc:.3f}; truncated {sum(o['stop_type'] == 'limit' for o in outs)}")
    print(outs[0]["text"][-300:])


if __name__ == "__main__":
    if sys.argv[1] == "timing":
        timing(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "model")

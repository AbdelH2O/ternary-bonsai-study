"""Which Qwen3 does Prism's Ternary-Bonsai-1.7B answer like? Descriptive, CPU only (GPU hidden).

Greedy openings (first 16 tokens) and first-token probabilities on 100 GSM8K *train* questions (items
100-199; the test set is untouched), with the Phase 1 prompt and the Qwen3 template, thinking off, for
Qwen3-1.7B (F32), Prism (PQ2_0, rope scaling off as in Phase 1's reference), Qwen3-4B and Qwen3-8B (Q8_0).
If Prism was distilled from Qwen3-1.7B alone, its openings should resemble Qwen3-1.7B's; a closer match to
a larger sibling would point to a different teacher or training data.

    python3 teacher_style_probe.py
"""

from __future__ import annotations

import collections
import io
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gsm8k_17b as g

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/teacher_style_probe.json"
MODELS = {
    "qwen3_1.7b": (ROOT / "work/qwen3_17b/base_tied-f32.gguf", "model"),
    "prism_1.7b": (ROOT / "work/qwen3_17b/prism_gguf/Ternary-Bonsai-1.7B-PQ2_0.gguf", "none"),
    "qwen3_4b": (ROOT / "work/qwen3_teachers/Qwen3-4B-Q8_0.gguf", "model"),
    "qwen3_8b": (ROOT / "work/qwen3_teachers/Qwen3-8B-Q8_0.gguf", "model"),
}
N_PREDICT = 16


def prompts() -> list[dict]:
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ROOT / "work/qwen3_17b/base")
    url = ("https://huggingface.co/datasets/openai/gsm8k/resolve/740312add88f781978c0658806c59bc2815b9866/"
           "main/train-00000-of-00001.parquet")
    with urllib.request.urlopen(url) as r:
        rows = pq.read_table(io.BytesIO(r.read())).to_pylist()[100:200]
    items = []
    for i, row in enumerate(rows, start=100):
        text = f"{row['question']}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        prompt = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        items.append({"id": i, "prompt_ids": tok(prompt, add_special_tokens=False)["input_ids"]})
    return items


def complete(ids: list[int]) -> dict:
    body = json.dumps({"prompt": ids, "n_predict": N_PREDICT, "temperature": 0.0, "top_k": 1, "n_probs": 8,
                       "cache_prompt": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{g.PORT}/completion", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.loads(r.read())
    first = out.get("completion_probabilities", [{}])[0]
    top = first.get("top_probs") or first.get("probs") or []
    return {"text": out["content"],
            "first_token_top": [(t.get("token", t.get("tok_str")), t.get("prob", t.get("logprob"))) for t in top]}


def opening(text: str) -> str:
    return text.strip().split("\n")[0][:48]


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # another job owns the GPU; keep this probe on the CPU
    g.SLOTS = 4
    items = prompts()
    result = {"items": [it["id"] for it in items], "models": {}}
    for name, (path, rope) in MODELS.items():
        cmd_log = ROOT / f"work/qwen3_teachers/{name}.style.log"
        g.SERVER = g.REPO / "bin/cuda/llama-server"
        with g.serve(path, rope, cmd_log):
            with ThreadPoolExecutor(4) as pool:
                outs = list(pool.map(lambda it: complete(it["prompt_ids"]), items))
        openings = collections.Counter(opening(o["text"]) for o in outs)
        result["models"][name] = {"file": str(path.relative_to(ROOT)), "rope": rope,
                                  "top_openings": openings.most_common(8),
                                  "outputs": [{"id": it["id"], **o} for it, o in zip(items, outs)]}
        print(name, openings.most_common(4), flush=True)
    names = list(MODELS)
    per = {n: [opening(o["text"]) for o in result["models"][n]["outputs"]] for n in names}
    result["opening_agreement"] = {f"{a}~{b}": sum(x == y for x, y in zip(per[a], per[b])) / len(items)
                                   for i, a in enumerate(names) for b in names[i + 1:]}
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["opening_agreement"], indent=1))


if __name__ == "__main__":
    main()

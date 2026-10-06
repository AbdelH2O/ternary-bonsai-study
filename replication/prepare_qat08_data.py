"""Tokenize one pinned FineWeb-Edu shard into packed training tokens for the 0.8B training test.

Documents are tokenized with the Qwen3.5 tokenizer, each followed by <|endoftext|>, and concatenated
in file order. The first 2,097,152 tokens are a training-domain monitoring slice (never evaluation);
the next TRAIN_TOKENS are the training stream. Output: uint32 arrays in work/qat08/data.

    .venv/bin/python prepare_qat08_data.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "work/qat08/data"
SHARD = DATA / "fineweb_edu_000_00000.parquet"
SOURCE = "HuggingFaceFW/fineweb-edu@87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 sample/10BT/000_00000.parquet"
MONITOR_TOKENS = 2_097_152
TRAIN_TOKENS = 400_000_000


def main() -> None:
    tok = AutoTokenizer.from_pretrained(ROOT / "work/qwen35_08b/base")
    eos = tok.convert_tokens_to_ids("<|endoftext|>")
    target = MONITOR_TOKENS + TRAIN_TOKENS
    out = np.empty(target, dtype=np.uint32)
    filled, docs = 0, 0
    pf = pq.ParquetFile(SHARD)
    for batch in pf.iter_batches(batch_size=2048, columns=["text"]):
        texts = batch.column("text").to_pylist()
        for ids in tok(texts, add_special_tokens=False)["input_ids"]:
            ids.append(eos)
            take = min(len(ids), target - filled)
            out[filled:filled + take] = ids[:take]
            filled += take
            docs += 1
            if filled == target:
                break
        if filled == target:
            break
        print(f"{filled / 1e6:.0f}M tokens from {docs} documents", flush=True)
    assert filled == target, f"shard too small: {filled}"
    out[:MONITOR_TOKENS].tofile(DATA / "monitor.u32")
    out[MONITOR_TOKENS:].tofile(DATA / "train.u32")
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    record = {"source": SOURCE, "source_sha256": digest(SHARD), "documents": docs, "eos_id": int(eos),
              "monitor_tokens": MONITOR_TOKENS, "train_tokens": TRAIN_TOKENS,
              "monitor_sha256": digest(DATA / "monitor.u32"), "train_sha256": digest(DATA / "train.u32")}
    (DATA / "data_record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

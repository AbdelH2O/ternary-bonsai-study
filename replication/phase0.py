"""Bounded, pinned Bonsai 2 / Qwen3.8 weight comparison.

Only requested safetensors byte ranges are read. A response without a valid 206
Content-Range is rejected before its body is consumed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from inspect_gguf import inspect  # noqa: E402

MODEL = ROOT / "models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf"
INDEX = HERE / "metadata/qwen38-index.json"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
REPO = "Qwen/Qwen3.8-27B"
OUT = HERE / "results"

SUFFIX = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_a": "linear_attn.A_log",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
}


def source_name(gguf_name: str) -> str:
    if gguf_name == "output.weight":
        return "lm_head.weight"
    if gguf_name == "output_norm.weight":
        return "model.language_model.norm.weight"
    if gguf_name == "token_embd.weight":
        return "model.language_model.embed_tokens.weight"
    match = re.fullmatch(r"blk\.(\d+)\.(.*)", gguf_name)
    if not match or match[2] not in SUFFIX:
        raise ValueError(f"unmapped GGUF tensor: {gguf_name}")
    return f"model.language_model.layers.{match[1]}.{SUFFIX[match[2]]}"


def tensor_shape(row: dict) -> tuple[int, ...]:
    return tuple(reversed(tuple(map(int, row["dimensions_gguf_order"].split("x")))))


def align(n: int, to: int = 32) -> int:
    return (n + to - 1) // to * to


def local_header():
    _, metadata, rows, header_bytes = inspect(MODEL)
    return metadata, rows, align(header_bytes)


class RangeClient:
    def __init__(self, repo: str = REPO, revision: str = REVISION):
        self.repo = repo
        self.revision = revision
        self.urls: dict[str, str] = {}
        self.bytes_read = 0
        self.requests = 0

    def read(self, filename: str, start: int, length: int) -> bytes:
        if length <= 0 or length > 16 * 1024 * 1024:
            raise ValueError(f"range length out of bounds: {length}")
        end = start + length - 1
        url = self.urls.get(filename) or f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{filename}"
        for attempt in range(4):
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "bonsai-phase0/1"})
            try:
                with urllib.request.urlopen(req, timeout=60) as response:
                    cr = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", cr)
                    if response.status != 206 or not match or (int(match[1]), int(match[2])) != (start, end):
                        raise ValueError(f"unsafe HTTP range response: status={response.status}, Content-Range={cr!r}")
                    data = response.read(length + 1)
                    if len(data) != length:
                        raise ValueError(f"short/overlong range: {len(data)} versus {length}")
                    self.urls[filename] = response.url
                    self.bytes_read += length
                    self.requests += 1
                    return data
            except (TimeoutError, OSError) as exc:
                if attempt == 3:
                    raise
                self.urls.pop(filename, None)
                url = f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{filename}"
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")


def shard_headers(index: dict, client: RangeClient, filenames=None) -> dict:
    result = {}
    for filename in sorted(filenames or set(index["weight_map"].values())):
        size = int.from_bytes(client.read(filename, 0, 8), "little")
        if size < 2 or size > 8 * 1024 * 1024:
            raise ValueError(f"implausible safetensors header in {filename}: {size}")
        header = json.loads(client.read(filename, 8, size))
        result[filename] = {"data_offset": 8 + size, "tensors": header}
        print(f"header {filename}: {len(header)-('_metadata' in header)} tensors, {size} bytes", flush=True)
    return result


def bf16_to_f32(data: bytes) -> np.ndarray:
    bits = np.frombuffer(data, dtype="<u2").astype("<u4") << 16
    return bits.view("<f4")


def decode_source(data: bytes, dtype: str) -> np.ndarray:
    if dtype == "BF16":
        return bf16_to_f32(data)
    if dtype == "F32":
        return np.frombuffer(data, dtype="<f4")
    if dtype == "F16":
        return np.frombuffer(data, dtype="<f2").astype("<f4")
    raise ValueError(dtype)


def reorder_v(x: np.ndarray, axis: int, head_dim: int) -> np.ndarray:
    # Qwen3.8: 16 K heads, three V heads per K head, grouped -> tiled.
    shape = list(x.shape)
    if shape[axis] != 16 * 3 * head_dim:
        raise ValueError(f"V axis mismatch: {shape}, axis={axis}, dim={head_dim}")
    new_shape = shape[:axis] + [16, 3, head_dim] + shape[axis + 1:]
    y = x.reshape(new_shape)
    return np.swapaxes(y, axis, axis + 1).copy().reshape(shape)


def transform_source(name: str, x: np.ndarray) -> np.ndarray:
    if name.endswith(".linear_attn.A_log"):
        x = -np.exp(x)
    elif name.endswith(".linear_attn.conv1d.weight"):
        x = np.squeeze(x)
    elif name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
        x = x + np.float32(1)

    if ".linear_attn." in name:
        if name.endswith((".A_log", ".dt_bias")):
            x = reorder_v(x, 0, 1)
        elif name.endswith(".conv1d.weight"):
            x = np.concatenate([x[:4096], reorder_v(x[4096:], 0, 128)], axis=0)
        elif name.endswith((".in_proj_a.weight", ".in_proj_b.weight")):
            x = reorder_v(x, 0, 1)
    return x


def inventory():
    metadata, rows, _ = local_header()
    index = json.loads(INDEX.read_text())
    names = set(index["weight_map"])
    mapped = []
    for row in rows:
        name = source_name(row["tensor"])
        if name not in names:
            raise ValueError(f"missing Qwen source tensor: {name}")
        mapped.append({**row, "source": name, "shard": index["weight_map"][name]})
    assert len(mapped) == 851 and len({r["source"] for r in mapped}) == 851
    assert Counter(r["storage"] for r in mapped) == {"PQ2_0": 402, "BF16": 96, "F32": 353}
    folded = set(metadata["prism.hadamard.weight_names"])
    source_folded = [r["source"] for r in mapped if r["tensor"] in folded]
    assert len(source_folded) == 401
    gdn_outputs = {r["source"] for r in mapped if r["tensor"].endswith("ssm_out.weight")}
    assert len(gdn_outputs) == 48 and gdn_outputs <= set(source_folded)
    OUT.mkdir(exist_ok=True)
    with (OUT / "mapping.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(mapped[0]))
        writer.writeheader()
        writer.writerows(mapped)
    (OUT / "source_folded_names.json").write_text(json.dumps(source_folded, indent=2) + "\n")
    print(f"mapped {len(mapped)} tensors; {len(source_folded)} projection source names; {len(names)-len(mapped)} HF-only tensors")
    return mapped


def validate_remote_layout():
    mapped = inventory()
    index = json.loads(INDEX.read_text())
    client = RangeClient()
    headers = shard_headers(index, client)
    for row in mapped:
        desc = headers[row["shard"]]["tensors"][row["source"]]
        source_shape = tuple(desc["shape"])
        if row["tensor"].endswith("ssm_conv1d.weight"):
            source_shape = tuple(n for n in source_shape if n != 1)
        if source_shape != tensor_shape(row) or desc["dtype"] != "BF16":
            raise ValueError(f"source layout mismatch for {row['tensor']}: {desc}")
        if desc["data_offsets"][1] - desc["data_offsets"][0] != row["elements"] * 2:
            raise ValueError(f"source byte count mismatch for {row['tensor']}")
    (OUT / "layout_validation.json").write_text(json.dumps({
        "qwen_repo": REPO, "revision": REVISION, "checked_tensors": len(mapped),
        "checked_projection_source_names": 401, "checked_gdn_folded_out_projections": 48,
        "shard_headers": len(headers), "range_requests": client.requests,
        "range_bytes": client.bytes_read,
    }, indent=2) + "\n")
    print(f"Validated source dtype, shape, and byte count for {len(mapped)} tensors")


def t1(limit: int | None = None):
    mapped = inventory()
    protected = [r for r in mapped if r["storage"] != "PQ2_0"]
    if limit:
        protected = protected[:limit]
    index = json.loads(INDEX.read_text())
    client = RangeClient()
    headers = shard_headers(index, client)
    _, _, data_start = local_header()
    results = []
    with MODEL.open("rb") as gguf:
        for i, row in enumerate(protected, 1):
            name, shard = row["source"], row["shard"]
            descriptor = headers[shard]["tensors"][name]
            offset0, offset1 = descriptor["data_offsets"]
            length = offset1 - offset0
            expected = row["elements"] * 2
            if length != expected or descriptor["dtype"] != "BF16":
                raise ValueError(f"unexpected Qwen tensor dtype/size for {name}: {descriptor}")
            remote = client.read(shard, headers[shard]["data_offset"] + offset0, length)
            source = decode_source(remote, descriptor["dtype"]).reshape(descriptor["shape"])
            source = transform_source(name, source)
            if source.shape != tensor_shape(row):
                raise ValueError(f"shape mismatch: {name}: {source.shape} versus {tensor_shape(row)}")
            gguf.seek(data_start + row["offset"])
            local_bytes = gguf.read(row["bytes"])
            if len(local_bytes) != row["bytes"]:
                raise ValueError(f"truncated local tensor: {row['tensor']}")
            target = decode_source(local_bytes, row["storage"]).reshape(source.shape)
            a = source.astype(np.float64).ravel()
            b = target.astype(np.float64).ravel()
            delta = b - a
            abs_delta = np.abs(delta)
            exact = np.count_nonzero(source.ravel().view("<u4") == target.ravel().view("<u4"))
            corr = float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 and np.std(a) and np.std(b) else None
            result = {
                "gguf": row["tensor"], "source": name, "block": row["block"],
                "kind": row["tensor"].split(".")[-2] if row["block"] is not None else row["tensor"],
                "storage": row["storage"], "elements": row["elements"],
                "exact": int(exact), "changed_fraction": float(1 - exact / len(a)),
                "max_abs_delta": float(np.max(abs_delta)), "mean_abs_delta": float(np.mean(abs_delta)),
                "p50_abs_delta": float(np.quantile(abs_delta, .5)), "p95_abs_delta": float(np.quantile(abs_delta, .95)),
                "p99_abs_delta": float(np.quantile(abs_delta, .99)),
                "relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(a)) if np.linalg.norm(a) else None,
                "correlation": corr,
            }
            results.append(result)
            if i % 25 == 0 or i == len(protected):
                print(f"T1 {i}/{len(protected)}; {client.bytes_read/1e6:.1f} MB useful ranges", flush=True)
    (OUT / ("t1-probe.json" if limit else "t1.json")).write_text(json.dumps({
        "qwen_repo": REPO, "revision": REVISION, "local_model": str(MODEL.relative_to(ROOT)),
        "local_bytes": MODEL.stat().st_size, "range_requests": client.requests,
        "range_bytes": client.bytes_read, "complete": not bool(limit), "tensors": results,
    }, indent=2) + "\n")
    print(f"T1 done: {len(results)} tensors; {sum(r['exact']==r['elements'] for r in results)} exactly equal")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["inventory", "validate", "t1"])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.command == "inventory":
        inventory()
    elif args.command == "validate":
        validate_remote_layout()
    elif args.command == "t1":
        t1(args.limit)


if __name__ == "__main__":
    main()

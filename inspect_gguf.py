"""Read a GGUF header and map Bonsai 2's storage by architectural region.

This reads metadata and tensor descriptors only; it does not load model weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from collections import defaultdict
from pathlib import Path


SCALAR_FORMATS = {
    0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
    7: "?", 10: "Q", 11: "q", 12: "d",
}
STORAGE_TYPES = {0: "F32", 30: "BF16", 142: "PQ2_0"}


class HeaderReader:
    def __init__(self, file):
        self.file = file

    def scalar(self, fmt):
        size = struct.calcsize("<" + fmt)
        data = self.file.read(size)
        if len(data) != size:
            raise ValueError("Truncated GGUF header")
        return struct.unpack("<" + fmt, data)[0]

    def string(self, keep=True):
        length = self.scalar("Q")
        if length > 20_000_000:
            raise ValueError(f"Implausible GGUF string length: {length}")
        if not keep:
            self.file.seek(length, 1)
            return None
        data = self.file.read(length)
        if len(data) != length:
            raise ValueError("Truncated GGUF string")
        return data.decode("utf-8")

    def value(self, type_id, keep=True):
        if type_id == 8:
            return self.string(keep)
        if type_id == 9:
            item_type, count = self.scalar("I"), self.scalar("Q")
            if count > 10_000_000:
                raise ValueError(f"Implausible GGUF array length: {count}")
            if not keep or count > 1024:
                if item_type in SCALAR_FORMATS:
                    self.file.seek(struct.calcsize("<" + SCALAR_FORMATS[item_type]) * count, 1)
                else:
                    for _ in range(count):
                        self.value(item_type, False)
                return {"array_length": count} if keep else None
            return [self.value(item_type) for _ in range(count)]
        fmt = SCALAR_FORMATS.get(type_id)
        if fmt is None:
            raise ValueError(f"Unknown GGUF metadata type: {type_id}")
        if keep:
            return self.scalar(fmt)
        self.file.seek(struct.calcsize("<" + fmt), 1)
        return None


def tensor_region(name):
    if name == "token_embd.weight":
        return "embedding", "embedding", None
    if name == "output.weight":
        return "lm_head", "lm_head", None
    if name == "output_norm.weight":
        return "final_norm", "normalization", None
    if not name.startswith("blk."):
        return "other", "other", None
    parts = name.split(".", 2)
    block = int(parts[1])
    kind = "full_attention" if block % 4 == 3 else "linear_attention"
    suffix = parts[2]
    if suffix.startswith("ffn_"):
        region = "mlp"
    elif suffix.startswith("ssm_"):
        region = "recurrent_state"
    elif "norm" in suffix:
        region = "normalization"
    else:
        region = "attention_projection"
    return kind, region, block


def tensor_bytes(type_id, n_elements):
    if type_id == 0:
        return n_elements * 4
    if type_id == 30:
        return n_elements * 2
    if type_id == 142:
        if n_elements % 128:
            raise ValueError("PQ2_0 tensor is not divisible into 128-element groups")
        return n_elements // 128 * 34
    raise ValueError(f"Unsupported tensor storage type {type_id}; inspect manually")


def inspect(path):
    with path.open("rb") as file:
        reader = HeaderReader(file)
        if file.read(4) != b"GGUF":
            raise ValueError("Not a GGUF file")
        version = reader.scalar("I")
        tensor_count, metadata_count = reader.scalar("Q"), reader.scalar("Q")
        metadata = {}
        wanted = {
            "general.architecture", "general.alignment", "qwen35.block_count", "qwen35.full_attention_interval",
            "prism.hadamard.block_size", "prism.hadamard.weight_names",
            "prism.hadamard.inverse_weight_names",
        }
        for _ in range(metadata_count):
            key = reader.string()
            value_type = reader.scalar("I")
            if key in wanted:
                metadata[key] = reader.value(value_type)
            else:
                reader.value(value_type, False)
        rows = []
        for _ in range(tensor_count):
            name = reader.string()
            dims = [reader.scalar("Q") for _ in range(reader.scalar("I"))]
            type_id, offset = reader.scalar("I"), reader.scalar("Q")
            elements = math.prod(dims)
            block_kind, region, block = tensor_region(name)
            rows.append({
                "tensor": name, "block": block, "block_kind": block_kind,
                "region": region, "storage": STORAGE_TYPES.get(type_id, str(type_id)),
                "elements": elements, "bytes": tensor_bytes(type_id, elements),
                "dimensions_gguf_order": "x".join(map(str, dims)), "offset": offset,
            })
        header_bytes = file.tell()
    return version, metadata, rows, header_bytes


def summarize(rows):
    groups = defaultdict(lambda: {"tensors": 0, "elements": 0, "bytes": 0})
    blocks = defaultdict(lambda: {"kind": "", "tensors": 0, "elements": 0, "bytes": 0,
                                  "full_precision_elements": 0, "full_precision_bytes": 0})
    for row in rows:
        key = (row["block_kind"], row["region"], row["storage"])
        for field, value in (("tensors", 1), ("elements", row["elements"]), ("bytes", row["bytes"])):
            groups[key][field] += value
        if row["block"] is not None:
            block = blocks[row["block"]]
            block["kind"] = row["block_kind"]
            block["tensors"] += 1
            block["elements"] += row["elements"]
            block["bytes"] += row["bytes"]
            if row["storage"] != "PQ2_0":
                block["full_precision_elements"] += row["elements"]
                block["full_precision_bytes"] += row["bytes"]
    return groups, blocks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Path to Bonsai 2 PQ2_0 GGUF")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    version, metadata, rows, header_bytes = inspect(args.model)
    if metadata.get("general.architecture") != "qwen35" or metadata.get("qwen35.block_count") != 64:
        raise ValueError("Expected a 64-block qwen35 GGUF; cannot apply this Bonsai 2 region map")
    args.output.mkdir(parents=True, exist_ok=True)
    groups, blocks = summarize(rows)
    rotated = set(metadata.get("prism.hadamard.weight_names", []))
    expected_rotated = {row["tensor"] for row in rows
                        if row["storage"] == "PQ2_0" and row["tensor"] != "token_embd.weight"}
    if rotated != expected_rotated:
        raise ValueError("Rotated weight-name map differs from expected ternary projection set")
    with (args.output / "tensors.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "regions.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["block_kind", "region", "storage", "tensors", "elements", "bytes"])
        for key, values in sorted(groups.items()):
            writer.writerow([*key, values["tensors"], values["elements"], values["bytes"]])
    with (args.output / "blocks.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["block", "kind", "tensors", "elements", "bytes", "full_precision_elements", "full_precision_bytes"])
        for index, values in sorted(blocks.items()):
            writer.writerow([index, *values.values()])
    summary_metadata = dict(metadata)
    summary_metadata["prism.hadamard.weight_names"] = {
        "count": len(metadata.get("prism.hadamard.weight_names", []))
    }
    summary = {
        "model": str(args.model), "gguf_version": version, "header_bytes": header_bytes,
        "tensor_count": len(rows), "file_bytes": args.model.stat().st_size,
        "metadata": summary_metadata,
        "storage_elements": {kind: sum(r["elements"] for r in rows if r["storage"] == kind)
                             for kind in sorted({r["storage"] for r in rows})},
        "storage_bytes": {kind: sum(r["bytes"] for r in rows if r["storage"] == kind)
                          for kind in sorted({r["storage"] for r in rows})},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Read {len(rows)} tensor descriptors and {header_bytes:,} header bytes; no weights loaded.")
    for kind, count in summary["storage_elements"].items():
        print(f"{kind}: {count:,} elements, {summary['storage_bytes'][kind]:,} bytes")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

"""Sample packed ternary groups from Bonsai 2 PQ2_0 without loading the model."""

from __future__ import annotations

import argparse
import csv
import struct
from collections import defaultdict
from pathlib import Path

from inspect_gguf import inspect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--groups-per-tensor", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    if args.groups_per_tensor < 1 or args.groups_per_tensor > 1024:
        parser.error("groups-per-tensor must be 1 to 1024")
    _, metadata, tensors, header_bytes = inspect(args.model)
    if metadata.get("general.architecture") != "qwen35" or metadata.get("prism.hadamard.block_size") != 1024:
        raise ValueError("Expected Bonsai 2 GGUF with Hadamard metadata")
    alignment = metadata.get("general.alignment", 32)
    data_start = (header_bytes + alignment - 1) // alignment * alignment
    tensor_rows = []
    totals = defaultdict(lambda: {"groups": 0, "exactly_42_zeros": 0, "zero": 0,
                                  "negative": 0, "positive": 0, "invalid": 0,
                                  "scale_sum": 0.0})
    with args.model.open("rb") as file:
        for tensor in tensors:
            if tensor["storage"] != "PQ2_0":
                continue
            group_count = tensor["elements"] // 128
            count = min(args.groups_per_tensor, group_count)
            indices = [i * (group_count - 1) // max(count - 1, 1) for i in range(count)]
            result = {"groups": 0, "exactly_42_zeros": 0, "zero": 0,
                      "negative": 0, "positive": 0, "invalid": 0,
                      "scale_sum": 0.0}
            for index in indices:
                file.seek(data_start + tensor["offset"] + 34 * index)
                block = file.read(34)
                if len(block) != 34:
                    raise ValueError(f"Truncated tensor {tensor['tensor']}")
                result["groups"] += 1
                result["scale_sum"] += abs(struct.unpack("<e", block[:2])[0])
                zeros_in_group = 0
                for byte in block[2:]:
                    for shift in (0, 2, 4, 6):
                        code = (byte >> shift) & 3
                        result[{0: "negative", 1: "zero", 2: "positive", 3: "invalid"}[code]] += 1
                        zeros_in_group += code == 1
                result["exactly_42_zeros"] += zeros_in_group == 42
            key = (tensor["block_kind"], tensor["region"])
            for field in result:
                totals[key][field] += result[field]
            tensor_rows.append({
                "tensor": tensor["tensor"], "block": tensor["block"],
                "block_kind": key[0], "region": key[1],
                "sampled_groups": count,
                "groups_with_42_zeros": result["exactly_42_zeros"],
                "zero_fraction": result["zero"] / (128 * count),
                "mean_abs_scale": result["scale_sum"] / count,
                "invalid_codes": result["invalid"],
            })
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "sampled_tensors.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(tensor_rows[0]))
        writer.writeheader()
        writer.writerows(tensor_rows)
    with (args.output / "sampled_regions.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["block_kind", "region", "sampled_groups", "groups_with_42_zeros", "zero_fraction",
                         "mean_abs_scale", "invalid_codes"])
        for (kind, region), item in sorted(totals.items()):
            writer.writerow([kind, region, item["groups"], item["exactly_42_zeros"],
                             item["zero"] / (128 * item["groups"]),
                             item["scale_sum"] / item["groups"], item["invalid"]])
    invalid = sum(item["invalid"] for item in totals.values())
    if invalid:
        raise ValueError(f"Found {invalid} invalid PQ2_0 codes; check format/version")
    print(f"Sampled {sum(item['groups'] for item in totals.values()):,} groups from "
          f"{len(tensor_rows)} tensors; wrote {args.output}.")


if __name__ == "__main__":
    main()

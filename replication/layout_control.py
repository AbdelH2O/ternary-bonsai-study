"""Real-weight negative control for Qwen3.8 grouped-to-tiled V conversion."""

from __future__ import annotations

import json

import numpy as np

from phase0 import (INDEX, MODEL, OUT, RangeClient, decode_source, inventory,
                    local_header, shard_headers, transform_source)


def main():
    mapped = {r["tensor"]: r for r in inventory()}
    names = ("blk.0.ssm_conv1d.weight", "blk.0.ssm_a", "blk.0.ssm_dt.bias")
    rows = [mapped[n] for n in names]
    client = RangeClient()
    headers = shard_headers(json.loads(INDEX.read_text()), client, {r["shard"] for r in rows})
    _, _, base = local_header()
    results = {}
    with MODEL.open("rb") as f:
        for row in rows:
            descriptor = headers[row["shard"]]["tensors"][row["source"]]
            start, end = descriptor["data_offsets"]
            remote = client.read(row["shard"], headers[row["shard"]]["data_offset"] + start, end - start)
            raw = decode_source(remote, descriptor["dtype"]).reshape(descriptor["shape"])
            if row["tensor"].endswith("conv1d.weight"):
                wrong = raw.squeeze()
            elif row["tensor"].endswith("ssm_a"):
                wrong = -np.exp(raw)
            else:
                wrong = raw
            correct = transform_source(row["source"], raw)
            f.seek(base + row["offset"])
            target = decode_source(f.read(row["bytes"]), row["storage"]).reshape(correct.shape)
            results[row["tensor"]] = {
                "correct_correlation": float(np.corrcoef(correct.ravel(), target.ravel())[0, 1]),
                "unreordered_correlation": float(np.corrcoef(wrong.ravel(), target.ravel())[0, 1]),
                "correct_mean_abs_delta": float(np.mean(np.abs(correct - target))),
                "unreordered_mean_abs_delta": float(np.mean(np.abs(wrong - target))),
            }
            assert results[row["tensor"]]["correct_mean_abs_delta"] < results[row["tensor"]]["unreordered_mean_abs_delta"]
    (OUT / "layout_control.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

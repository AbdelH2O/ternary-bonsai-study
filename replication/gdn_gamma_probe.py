"""GDN out-projection comparison after the valid norm-gamma symmetry correction."""

from __future__ import annotations

import json

import numpy as np

from matrix_probe import one_matrix, signs_from_gguf
from phase0 import (INDEX, MODEL, OUT, RangeClient, decode_source, inventory,
                    local_header, shard_headers)


def main():
    mapped = {r["tensor"]: r for r in inventory()}
    norm = mapped["blk.0.ssm_norm.weight"]
    out = mapped["blk.0.ssm_out.weight"]
    client = RangeClient()
    headers = shard_headers(json.loads(INDEX.read_text()), client, {norm["shard"], out["shard"]})
    descriptor = headers[norm["shard"]]["tensors"][norm["source"]]
    first, last = descriptor["data_offsets"]
    source = decode_source(client.read(norm["shard"], headers[norm["shard"]]["data_offset"] + first,
                                       last - first), descriptor["dtype"])
    _, _, base = local_header()
    with MODEL.open("rb") as f:
        f.seek(base + norm["offset"])
        target = decode_source(f.read(norm["bytes"]), norm["storage"])
    assert source.shape == target.shape == (128,)
    assert np.all(target != 0)
    ratio = (source / target).astype(np.float64)
    scale = np.tile(ratio, 48)  # grouped V: 48 value heads, each 128 channels
    result = one_matrix(out, headers, client, signs_from_gguf(), base, column_scale=scale)
    (OUT / "gdn_gamma_probe.json").write_text(json.dumps({
        "gamma_ratio_min": float(ratio.min()), "gamma_ratio_max": float(ratio.max()),
        "gamma_ratio_mean": float(ratio.mean()), "range_requests": client.requests,
        "range_bytes": client.bytes_read, "matrix": result}, indent=2) + "\n")
    print(f"gamma-corrected GDN: sign {result['sign_matches']/result['nonzeros']:.6f}; "
          f"fixed86 Jaccard {result['fixed86_intersection']/result['fixed86_union']:.6f}")


if __name__ == "__main__":
    main()

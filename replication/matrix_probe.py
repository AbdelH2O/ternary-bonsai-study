"""Stream three complete rotated Qwen matrices against Bonsai PQ2_0 codes.

No full checkpoint or matrix is stored. Results describe a specified public
candidate, not the unpublished latent checkpoint or training method.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from contract_fixtures import fwht_last
from phase0 import (INDEX, MODEL, OUT, REPO, REVISION, RangeClient, bf16_to_f32,
                    inventory, local_header, shard_headers, tensor_shape)

NAMES = ("blk.0.ffn_down.weight", "blk.3.attn_output.weight", "blk.0.ssm_out.weight")


def signs_from_gguf() -> dict[int, np.ndarray]:
    from inspect_gguf import HeaderReader

    with MODEL.open("rb") as f:
        r = HeaderReader(f)
        assert f.read(4) == b"GGUF"
        r.scalar("I")
        r.scalar("Q")
        count = r.scalar("Q")
        widths = None
        values = None
        for _ in range(count):
            key, kind = r.string(), r.scalar("I")
            if key == "prism.hadamard.sign_widths":
                widths = r.value(kind)
            elif key == "prism.hadamard.sign_values":
                assert kind == 9
                item, n = r.scalar("I"), r.scalar("Q")
                assert item == 5 and n == 28672
                values = np.frombuffer(f.read(n * 4), dtype="<i4").copy()
            else:
                r.value(kind, False)
    assert widths == [5120, 6144, 17408] and values is not None
    assert np.all(np.isin(values, [-1, 1]))
    cuts = np.cumsum([0] + widths)
    return {w: values[cuts[i]:cuts[i + 1]].astype(np.float64) for i, w in enumerate(widths)}


def decode_blocks(data: bytes, nrows: int, ncols: int):
    raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 34)
    scales = np.frombuffer(raw[:, :2].copy().tobytes(), dtype="<f2")
    byte_codes = raw[:, 2:]
    code = np.stack([(byte_codes >> shift) & 3 for shift in (0, 2, 4, 6)], axis=-1).reshape(-1, 128)
    assert not np.any(code == 3)
    return (code.astype(np.int8) - 1), scales


def score_chunks(actual: np.ndarray, scales: np.ndarray, candidate: np.ndarray, summary: dict):
    groups = actual.shape[0]
    source = candidate.reshape(groups, 128)
    abs_source = np.abs(source)
    zero = actual == 0
    nonzero = ~zero
    counts = np.count_nonzero(zero, axis=1)
    summary["zero_hist"].update(map(int, counts))
    summary["groups"] += groups
    summary["nonzeros"] += int(np.count_nonzero(nonzero))
    match = (np.sign(source) == actual) & nonzero
    summary["sign_matches"] += int(np.count_nonzero(match))
    summary["source_zero_on_nonzero"] += int(np.count_nonzero((source == 0) & nonzero))

    rank_order = np.argsort(abs_source, axis=1, kind="stable")
    rank = np.empty_like(rank_order)
    np.put_along_axis(rank, rank_order, np.arange(128)[None, :], axis=1)
    decile = np.minimum(rank * 10 // 128, 9)
    summary["decile_nonzeros"] += np.bincount(decile[nonzero], minlength=10)
    summary["decile_sign_matches"] += np.bincount(decile[match], minlength=10)

    # A deterministic fixed-86 rule: 86 largest magnitudes, stable original-index tie break.
    selected = rank >= 42
    intersection = np.count_nonzero(selected & nonzero, axis=1)
    union = np.count_nonzero(selected | nonzero, axis=1)
    summary["fixed86_exact_groups"] += int(np.count_nonzero(np.all(selected == nonzero, axis=1)))
    summary["fixed86_intersection"] += int(intersection.sum())
    summary["fixed86_union"] += int(union.sum())
    sorted_abs = np.take_along_axis(abs_source, rank_order, axis=1)
    summary["boundary_ties"] += int(np.count_nonzero(sorted_abs[:, 41] == sorted_abs[:, 42]))

    denominator = np.count_nonzero(nonzero, axis=1)
    ls = np.divide(np.sum(actual * source, axis=1), denominator,
                   out=np.zeros(groups, dtype=np.float64), where=denominator != 0)
    ls = np.maximum(ls, 0)
    summary["signed_ls_scale_exact_bits"] += int(np.count_nonzero(ls.astype("<f2").view("<u2") == scales.view("<u2")))
    summary["signed_ls_scale_abs_sum"] += float(np.abs(scales.astype(np.float64) - ls).sum())

    amax = abs_source.max(axis=1)
    ratios = np.divide(source, amax[:, None], out=np.zeros_like(source), where=amax[:, None] != 0)
    absmax_codes = np.sign(ratios) * np.floor(np.abs(ratios) + 0.5)
    summary["absmax_exact_groups"] += int(np.count_nonzero(np.all(absmax_codes == actual, axis=1)))
    summary["absmax_scale_exact_bits"] += int(np.count_nonzero(amax.astype("<f2").view("<u2") == scales.view("<u2")))
    summary["absmax_code_matches"] += int(np.count_nonzero(absmax_codes == actual))

    decoded = actual.astype(np.float64) * scales.astype(np.float64)[:, None]
    summary["candidate_dot_decoded"] += float(np.sum(source * decoded))
    summary["candidate_square"] += float(np.sum(source * source))
    summary["decoded_square"] += float(np.sum(decoded * decoded))
    summary["candidate_sum"] += float(np.sum(source))
    summary["decoded_sum"] += float(np.sum(decoded))
    summary["elements"] += source.size


def one_matrix(row, headers, client, signs, data_start, column_scale=None):
    name, shard = row["source"], row["shard"]
    descriptor = headers[shard]["tensors"][name]
    assert descriptor["dtype"] == "BF16" and tuple(descriptor["shape"]) == tensor_shape(row)
    nrows, ncols = descriptor["shape"]
    source_start = headers[shard]["data_offset"] + descriptor["data_offsets"][0]
    assert descriptor["data_offsets"][1] - descriptor["data_offsets"][0] == row["elements"] * 2
    local_start = data_start + row["offset"]
    bytes_per_row = ncols // 128 * 34
    summary = {"gguf": row["tensor"], "source": name, "shape": [nrows, ncols],
               "candidate": "norm_gamma_corrected" if column_scale is not None else "raw_public_base",
               "groups": 0, "elements": 0, "nonzeros": 0, "sign_matches": 0,
               "source_zero_on_nonzero": 0, "fixed86_exact_groups": 0,
               "fixed86_intersection": 0, "fixed86_union": 0, "boundary_ties": 0,
               "signed_ls_scale_exact_bits": 0, "signed_ls_scale_abs_sum": 0.0,
               "absmax_exact_groups": 0, "absmax_scale_exact_bits": 0, "absmax_code_matches": 0,
               "candidate_dot_decoded": 0.0, "candidate_square": 0.0,
               "decoded_square": 0.0, "candidate_sum": 0.0, "decoded_sum": 0.0,
               "zero_hist": Counter(), "decile_nonzeros": np.zeros(10, dtype=np.int64),
               "decile_sign_matches": np.zeros(10, dtype=np.int64)}
    with MODEL.open("rb") as f:
        for first in range(0, nrows, 128):
            count = min(128, nrows - first)
            data = client.read(shard, source_start + first * ncols * 2, count * ncols * 2)
            w = bf16_to_f32(data).reshape(count, ncols).astype(np.float64)
            if column_scale is not None:
                w *= column_scale
            candidate = fwht_last(w * signs[ncols])
            f.seek(local_start + first * bytes_per_row)
            packed = f.read(count * bytes_per_row)
            if len(packed) != count * bytes_per_row:
                raise ValueError(f"truncated local matrix {row['tensor']}")
            actual, scales = decode_blocks(packed, count, ncols)
            score_chunks(actual, scales, candidate, summary)
            if first and first % 1024 == 0:
                print(f"{row['tensor']}: {first}/{nrows} rows", flush=True)
    assert summary["elements"] == row["elements"]
    for key in ("zero_hist",):
        summary[key] = dict(sorted(summary[key].items()))
    for key in ("decile_nonzeros", "decile_sign_matches"):
        summary[key] = summary[key].tolist()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="*", default=NAMES)
    parser.add_argument("--output", default="matrix_probe.json", help="result filename under results/")
    args = parser.parse_args()
    if args.output != Path(args.output).name:
        raise ValueError("output must be a filename under results/")
    index = json.loads(INDEX.read_text())
    mapped = {r["tensor"]: r for r in inventory()}
    rows = [mapped[n] for n in args.names]
    signs = signs_from_gguf()
    client = RangeClient()
    headers = shard_headers(index, client, {r["shard"] for r in rows})
    _, _, data_start = local_header()
    results = []
    for row in rows:
        result = one_matrix(row, headers, client, signs, data_start)
        results.append(result)
        print(f"{row['tensor']}: sign {result['sign_matches']/result['nonzeros']:.4f}; "
              f"fixed86 Jaccard {result['fixed86_intersection']/result['fixed86_union']:.4f}", flush=True)
        OUT.mkdir(exist_ok=True)
        (OUT / args.output).write_text(json.dumps({"qwen_repo": REPO, "revision": REVISION,
            "range_requests": client.requests, "range_bytes": client.bytes_read,
            "complete": len(results) == len(args.names), "matrices": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()

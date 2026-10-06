"""Compare the pinned C PQ2_0 codec with local and remote F16 GGUF groups."""

from __future__ import annotations

import ctypes
import io
import json
from pathlib import Path

import numpy as np

from phase0 import MODEL, OUT, ROOT, RangeClient, align, local_header

PINNED_LIB = ROOT / "bin/cuda/libggml-base.so"
F16_REPO = "prism-ml/Ternary-Bonsai-2-27B-gguf"
F16_REVISION = "b072e1d3b35a0a630cece372c2127528e0994386"
F16_FILE = "Ternary-Bonsai-2-27B-F16.gguf"


def remote_header(client: RangeClient):
    from inspect_gguf import HeaderReader

    prefix = client.read(F16_FILE, 0, 12 * 1024 * 1024)
    stream = io.BytesIO(prefix)
    reader = HeaderReader(stream)
    assert stream.read(4) == b"GGUF"
    version = reader.scalar("I")
    tensor_count, metadata_count = reader.scalar("Q"), reader.scalar("Q")
    alignment = 32
    metadata = {}
    for _ in range(metadata_count):
        key = reader.string()
        kind = reader.scalar("I")
        if key in {"general.alignment", "prism.hadamard.block_size", "general.basename"}:
            metadata[key] = reader.value(kind)
            if key == "general.alignment":
                alignment = metadata[key]
        else:
            reader.value(kind, False)
    rows = {}
    for _ in range(tensor_count):
        name = reader.string()
        dims = [reader.scalar("Q") for _ in range(reader.scalar("I"))]
        kind, offset = reader.scalar("I"), reader.scalar("Q")
        rows[name] = {"dimensions": dims, "type": kind, "offset": offset}
    return {"version": version, "tensor_count": tensor_count, "metadata": metadata,
            "data_offset": align(stream.tell(), alignment), "rows": rows}


def c_codec():
    lib = ctypes.CDLL(str(PINNED_LIB))
    lib.quantize_row_pq2_0_ref.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int64]
    lib.dequantize_row_pq2_0.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]

    def quantize(x: np.ndarray) -> bytes:
        arr = np.ascontiguousarray(x, dtype=np.float32)
        assert arr.size == 128
        output = ctypes.create_string_buffer(34)
        lib.quantize_row_pq2_0_ref(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), output, 128)
        return output.raw

    def decode(data: bytes) -> np.ndarray:
        assert len(data) == 34
        output = np.empty(128, dtype=np.float32)
        block = ctypes.create_string_buffer(data, 34)
        lib.dequantize_row_pq2_0(block, output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 128)
        return output

    return quantize, decode


def python_decode(data: bytes) -> np.ndarray:
    scale = np.frombuffer(data[:2], dtype="<f2").astype(np.float32)[0]
    bits = np.frombuffer(data[2:], dtype=np.uint8)
    code = np.stack([(bits >> shift) & 3 for shift in (0, 2, 4, 6)], axis=1).reshape(-1)
    return (code.astype(np.float32) - 1) * scale


def synthetic(quantize, decode):
    fixtures = {}
    for label, x in {
        "all_zero": np.zeros(128, dtype=np.float32),
        "three_codes": np.tile(np.array([-1, 0, 1, 0], dtype=np.float32), 32),
        "non_fp16_scale": np.tile(np.array([-1.0003, 0, 1.0003, 0], dtype=np.float32), 32),
        "near_half_threshold": np.tile(np.array([-1, -.4999, .5001, 1], dtype=np.float32), 32),
    }.items():
        block = quantize(x)
        actual = decode(block)
        assert np.array_equal(actual, python_decode(block))
        fixtures[label] = {"scale": float(np.frombuffer(block[:2], dtype="<f2")[0]),
                           "codes": {str(i): int(c) for i, c in enumerate(np.bincount(np.stack(
                               [(np.frombuffer(block[2:], dtype=np.uint8) >> shift) & 3 for shift in (0, 2, 4, 6)],
                               axis=1).reshape(-1), minlength=4)) if c}}
    # The decoder's fourth 2-bit value is +2, even though published ternary groups must not use it.
    fourth = b"\x00\x3c" + b"\xff" * 32  # 1.0 FP16, code 3 in every lane
    assert np.all(decode(fourth) == 2.0)
    assert np.array_equal(decode(fourth), python_decode(fourth))
    return fixtures


def main():
    quantize, decode = c_codec()
    fixtures = synthetic(quantize, decode)
    client = RangeClient(F16_REPO, F16_REVISION)
    remote = remote_header(client)
    _, local_rows, local_start = local_header()
    local = {r["tensor"]: r for r in local_rows}
    assert remote["tensor_count"] == 851
    assert remote["metadata"]["prism.hadamard.block_size"] == 1024
    checked = []
    with MODEL.open("rb") as f:
        for name in ("blk.0.attn_gate.weight", "blk.0.ffn_down.weight", "token_embd.weight", "output.weight"):
            row = local[name]
            other = remote["rows"][name]
            assert row["storage"] == "PQ2_0" and other["type"] == 1  # GGUF F16
            assert other["dimensions"] == list(map(int, row["dimensions_gguf_order"].split("x")))
            ngroups = row["elements"] // 128
            for g in (0, 1, ngroups // 2, ngroups - 1):
                f.seek(local_start + row["offset"] + 34 * g)
                packed = f.read(34)
                source = client.read(F16_FILE, remote["data_offset"] + other["offset"] + 256 * g, 256)
                source_values = np.frombuffer(source, dtype="<f2").astype(np.float32)
                if not np.array_equal(source_values, decode(packed)):
                    raise AssertionError(f"F16 and C-decoded PQ2 differ: {name}, group {g}")
                if quantize(source_values) != packed:
                    raise AssertionError(f"C re-quantization differs from published PQ2: {name}, group {g}")
                if not np.array_equal(decode(packed), python_decode(packed)):
                    raise AssertionError(f"Python decoder differs: {name}, group {g}")
                codes = np.stack([(np.frombuffer(packed[2:], dtype=np.uint8) >> shift) & 3
                                  for shift in (0, 2, 4, 6)], axis=1).reshape(-1)
                assert not np.any(codes == 3)
                checked.append({"tensor": name, "group": g, "zeros": int(np.count_nonzero(codes == 1)),
                                "negative": int(np.count_nonzero(codes == 0)),
                                "positive": int(np.count_nonzero(codes == 2)),
                                "scale": float(np.frombuffer(packed[:2], dtype="<f2")[0])})
    result = {"source_repo": F16_REPO, "revision": F16_REVISION, "source_file": F16_FILE,
              "remote_header": {k: v for k, v in remote.items() if k != "rows"},
              "range_requests": client.requests, "range_bytes": client.bytes_read,
              "synthetic": fixtures, "published_groups": checked}
    OUT.mkdir(exist_ok=True)
    (OUT / "codec_probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"C/Python PQ2 codec and remote F16: {len(checked)} exact groups, {client.bytes_read/1e6:.1f} MB ranged")


if __name__ == "__main__":
    main()

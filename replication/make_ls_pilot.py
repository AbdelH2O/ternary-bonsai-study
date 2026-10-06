"""Replace PQ2_0 payload groups with fixed-support or MSE-optimal ternary codes."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from phase0 import OUT, align

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work/qwen35_08b"

import sys
sys.path.insert(0, str(ROOT.parent))
from inspect_gguf import inspect  # noqa: E402


def encode(values: np.ndarray, method: str) -> bytes:
    """Return 34-byte groups in the pinned PQ2_0 code layout."""
    n = len(values)
    magnitude = np.abs(values)
    order = np.argsort(-magnitude, axis=1, kind="stable")
    ranked = np.take_along_axis(magnitude, order, axis=1)
    prefix = np.cumsum(ranked, axis=1, dtype=np.float64)
    if method == "fixed86":
        k = np.full(n, 86, dtype=np.int64)
    elif method == "ls":
        score = prefix**2 / np.arange(1, 129, dtype=np.float64)
        k = np.argmax(score, axis=1) + 1
        k[prefix[:, -1] == 0] = 0
    else:
        raise ValueError(method)
    scale = np.zeros(n, dtype=np.float32)
    nonzero = k > 0
    scale[nonzero] = (prefix[np.arange(n)[nonzero], k[nonzero] - 1] / k[nonzero]).astype(np.float32)
    scale16 = scale.astype("<f2")
    assert np.all(np.isfinite(scale16))
    selected = np.zeros((n, 128), dtype=bool)
    selected[np.arange(n)[:, None], order] = np.arange(128)[None, :] < k[:, None]
    code = np.ones((n, 128), dtype=np.uint8)
    code[selected & (values < 0)] = 0
    code[selected & (values > 0)] = 2
    packed = code[:, 0::4] | (code[:, 1::4] << 2) | (code[:, 2::4] << 4) | (code[:, 3::4] << 6)
    result = np.empty((n, 34), dtype=np.uint8)
    result[:, :2] = scale16.view(np.uint8).reshape(n, 2)
    result[:, 2:] = packed
    return result.tobytes()


def build(basis: str) -> dict:
    source = WORK / f"{basis}-f32.gguf"
    template = WORK / f"{basis}-pq2.gguf"
    _, _, src_rows, src_header = inspect(source)
    _, _, packed_rows, packed_header = inspect(template)
    by_source = {r["tensor"]: r for r in src_rows}
    by_packed = {r["tensor"]: r for r in packed_rows}
    assert set(by_source) == set(by_packed) and len(by_source) == 320
    methods = ("fixed86", "ls")
    paths = {method: WORK / f"{basis}-pq2-{method}.gguf" for method in methods}
    for path in paths.values():
        shutil.copyfile(template, path)
    outputs = {method: path.open("r+b") for method, path in paths.items()}
    groups = 0
    try:
        with source.open("rb") as inp:
            for name, row in by_packed.items():
                if row["storage"] != "PQ2_0":
                    continue
                sr = by_source[name]
                assert sr["storage"] == "F32" and sr["elements"] == row["elements"]
                assert row["bytes"] == sr["elements"] // 128 * 34
                count = row["elements"] // 128
                for start in range(0, count, 2048):
                    n = min(2048, count - start)
                    inp.seek(align(src_header) + sr["offset"] + start * 512)
                    data = inp.read(n * 512)
                    assert len(data) == n * 512
                    values = np.frombuffer(data, dtype="<f4").reshape(n, 128)
                    assert np.all(np.isfinite(values))
                    for method, out in outputs.items():
                        payload = encode(values, method)
                        assert len(payload) == n * 34
                        out.seek(align(packed_header) + row["offset"] + start * 34)
                        out.write(payload)
                groups += count
                print(f"{basis}: {name} ({count} groups)", flush=True)
    finally:
        for out in outputs.values():
            out.close()
    for path in paths.values():
        assert path.stat().st_size == template.stat().st_size
    return {"basis": basis, "groups": groups, "outputs": {m: str(p) for m, p in paths.items()},
            "bytes": template.stat().st_size}


if __name__ == "__main__":
    rows = [build(basis) for basis in ("base", "folded")]
    (OUT / "quality_pilot_assignment.json").write_text(json.dumps(rows, indent=2) + "\n")

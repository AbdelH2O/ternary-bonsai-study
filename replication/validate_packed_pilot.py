"""Assert small-pilot PQ2 type policy, size, and sampled codec bytes."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from codec_probe import c_codec
from phase0 import OUT
from validate_folded_pilot import align, header_metadata
from inspect_gguf import inspect

WORK = Path(__file__).resolve().parent / "work/qwen35_08b"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    source = WORK / "folded-f32.gguf"
    packed = WORK / "folded-pq2.gguf"
    _, _, src_rows, src_header = inspect(source)
    _, metadata, packed_rows, packed_header = inspect(packed)
    a, b = ({r["tensor"]: r for r in rows} for rows in (src_rows, packed_rows))
    assert set(a) == set(b) and len(b) == 320
    folded = set(metadata["prism.hadamard.weight_names"])
    expected_pq = folded | {"token_embd.weight"}
    expected_bf = {n for n in b if n.endswith(("ssm_alpha.weight", "ssm_beta.weight"))}
    assert len(expected_pq) == 151 and len(expected_bf) == 36
    for name, row in b.items():
        assert row["dimensions_gguf_order"] == a[name]["dimensions_gguf_order"]
        expected = "PQ2_0" if name in expected_pq else "BF16" if name in expected_bf else "F32"
        if row["storage"] != expected:
            raise AssertionError(f"wrong type for {name}: {row['storage']} vs {expected}")
    contract = header_metadata(packed)
    assert contract["prism.hadamard.block_size"] == 512
    assert contract["prism.hadamard.tied_output"] is True
    quantize, decode = c_codec()
    checked = []
    with source.open("rb") as fsrc, packed.open("rb") as fpack:
        for name in ("token_embd.weight", "blk.0.ffn_down.weight", "blk.0.ssm_out.weight"):
            sr, pr = a[name], b[name]
            ngroups = sr["elements"] // 128
            for g in (0, 1, ngroups - 1):
                fsrc.seek(align(src_header) + sr["offset"] + 512*g)
                values = np.frombuffer(fsrc.read(512), dtype="<f4").copy()
                fpack.seek(align(packed_header) + pr["offset"] + 34*g)
                group = fpack.read(34)
                assert quantize(values) == group
                assert np.all(np.isfinite(decode(group)))
                checked.append({"tensor": name, "group": g})
    result = {"source_gguf_sha256": sha256(source), "packed_gguf_sha256": sha256(packed),
              "packed_file_bytes": packed.stat().st_size, "tensor_count": len(b),
              "types": dict(Counter(r["storage"] for r in packed_rows)),
              "metadata": contract, "exact_reference_codec_groups": checked,
              "arm": "H512 folded public Qwen3.5-0.8B, pinned PQ2_0 absmax, explicit BF16 recurrent gates"}
    (OUT / "fp_packed_pilot.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"validated {len(b)} tensor types and {len(checked)} C-codec groups; {packed.stat().st_size} bytes")


if __name__ == "__main__":
    main()

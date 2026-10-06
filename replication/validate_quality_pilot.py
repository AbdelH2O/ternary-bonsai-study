"""Verify all generated quality-pilot GGUF tensor roles and sampled PQ2 bytes."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from codec_probe import c_codec, python_decode
from make_ls_pilot import WORK, encode
from phase0 import OUT, align

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inspect_gguf import inspect  # noqa: E402


def main() -> None:
    checked = []
    _, c_decode = c_codec()
    for basis in ("base", "folded"):
        source = WORK / f"{basis}-f32.gguf"
        template = WORK / f"{basis}-pq2.gguf"
        _, _, source_rows, source_header = inspect(source)
        _, _, template_rows, template_header = inspect(template)
        sr = {row["tensor"]: row for row in source_rows}
        tr = {row["tensor"]: row for row in template_rows}
        for method in ("fixed86", "ls"):
            target = WORK / f"{basis}-pq2-{method}.gguf"
            _, _, target_rows, target_header = inspect(target)
            rows = {row["tensor"]: row for row in target_rows}
            assert set(rows) == set(tr) and len(rows) == 320
            assert target.stat().st_size == template.stat().st_size
            assert Counter(row["storage"] for row in rows.values()) == {"PQ2_0": 151, "BF16": 36, "F32": 133}
            assert {(name, row["storage"], row["elements"], row["offset"]) for name, row in rows.items()} == {
                (name, row["storage"], row["elements"], row["offset"]) for name, row in tr.items()}
            groups = []
            with source.open("rb") as fsrc, template.open("rb") as ftemplate, target.open("rb") as ftarget:
                for name, row in rows.items():
                    if row["storage"] != "PQ2_0":
                        off = align(target_header) + row["offset"]
                        ftarget.seek(off)
                        ftemplate.seek(align(template_header) + row["offset"])
                        assert ftarget.read(row["bytes"]) == ftemplate.read(row["bytes"]), name
                for name in ("token_embd.weight", "blk.0.ffn_down.weight", "blk.0.ssm_out.weight"):
                    source_row, target_row = sr[name], rows[name]
                    for group in (0, 1, target_row["elements"] // 128 - 1):
                        fsrc.seek(align(source_header) + source_row["offset"] + group * 512)
                        vals = np.frombuffer(fsrc.read(512), dtype="<f4").reshape(1, 128)
                        ftarget.seek(align(target_header) + target_row["offset"] + group * 34)
                        packed = ftarget.read(34)
                        assert packed == encode(vals, method)
                        decoded = python_decode(packed)
                        assert np.array_equal(decoded, c_decode(packed))
                        assert np.all(np.isfinite(decoded))
                        groups.append({"tensor": name, "group": group})
            checked.append({"basis": basis, "method": method, "bytes": target.stat().st_size,
                            "tensor_types": {"PQ2_0": 151, "BF16": 36, "F32": 133},
                            "unchanged_non_pq_tensors": 169, "sampled_exact_groups": groups})
    (OUT / "quality_pilot_validation.json").write_text(json.dumps(checked, indent=2) + "\n")
    print(f"verified {len(checked)} packed arms, 36 sampled groups, and every non-PQ2 payload")


if __name__ == "__main__":
    main()

"""Check the GPTQ arms: tensor types/offsets equal the template, non-PQ2 payloads are
byte-identical, and sampled PQ2 groups decode identically in the Python and pinned C codecs."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from codec_probe import c_codec, python_decode
from phase0 import align

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work/qwen35_08b"
OUT = ROOT / "results/gptq_pilot"
sys.path.insert(0, str(ROOT.parent))
from inspect_gguf import inspect  # noqa: E402


def main() -> None:
    _, c_decode = c_codec()
    records = []
    for basis in ("base", "folded"):
        template, target = WORK / f"{basis}-pq2.gguf", WORK / f"{basis}-pq2-gptq.gguf"
        _, _, trows, theader = inspect(template)
        _, _, grows, gheader = inspect(target)
        tr, gr = {r["tensor"]: r for r in trows}, {r["tensor"]: r for r in grows}
        assert theader == gheader and target.stat().st_size == template.stat().st_size
        assert {(n, r["storage"], r["elements"], r["offset"]) for n, r in gr.items()} == {
            (n, r["storage"], r["elements"], r["offset"]) for n, r in tr.items()}
        types = Counter(r["storage"] for r in gr.values())
        assert types == {"PQ2_0": 151, "BF16": 36, "F32": 133}
        changed = sampled = 0
        with template.open("rb") as ft, target.open("rb") as fg:
            ft.seek(0)
            fg.seek(0)
            assert ft.read(align(theader)) == fg.read(align(gheader))
            for name, row in gr.items():
                ft.seek(align(theader) + row["offset"])
                fg.seek(align(gheader) + row["offset"])
                a, b = ft.read(row["bytes"]), fg.read(row["bytes"])
                if row["storage"] != "PQ2_0":
                    assert a == b, name
                    continue
                changed += a != b
                groups = row["bytes"] // 34
                for g in np.linspace(0, groups - 1, 8).astype(int):
                    block = b[34 * g:34 * g + 34]
                    dec = python_decode(block)
                    assert np.array_equal(dec, c_decode(block)) and np.all(np.isfinite(dec)), name
                    sampled += 1
        records.append({"basis": basis, "bytes": target.stat().st_size, "tensor_types": dict(types),
                        "non_pq2_payloads_identical_to_template": 169,
                        "pq2_payloads_changed_from_absmax_template": changed,
                        "sampled_groups_c_python_identical": sampled})
    (OUT / "validation.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()

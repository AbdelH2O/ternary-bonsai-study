"""Integrity checks for the refinement arms: template-identical layout and non-PQ2 bytes,
exact 42-zero groups where required, and sampled Python/C codec agreement."""

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
OUT = ROOT / "results/gptq_v2"
sys.path.insert(0, str(ROOT.parent))
from inspect_gguf import inspect  # noqa: E402

ARMS = {"folded-pq2-gptq42": 86, "folded-pq2-gptqv2": None, "folded-pq2-gptq42v2": 86}


def zero_counts(block_bytes: bytes) -> np.ndarray:
    blocks = np.frombuffer(block_bytes, dtype=np.uint8).reshape(-1, 34)[:, 2:]
    codes = np.stack([(blocks >> s) & 3 for s in (0, 2, 4, 6)], axis=2).reshape(len(blocks), 128)
    return (codes == 1).sum(1)


def main() -> None:
    _, c_decode = c_codec()
    template = WORK / "folded-pq2.gguf"
    _, _, trows, theader = inspect(template)
    tr = {r["tensor"]: r for r in trows}
    records = []
    for arm, nonzeros in ARMS.items():
        target = WORK / f"{arm}.gguf"
        _, _, grows, gheader = inspect(target)
        gr = {r["tensor"]: r for r in grows}
        assert theader == gheader and target.stat().st_size == template.stat().st_size
        assert {(n, r["storage"], r["elements"], r["offset"]) for n, r in gr.items()} == {
            (n, r["storage"], r["elements"], r["offset"]) for n, r in tr.items()}
        types = Counter(r["storage"] for r in gr.values())
        assert types == {"PQ2_0": 151, "BF16": 36, "F32": 133}
        zero_hist, sampled = Counter(), 0
        with template.open("rb") as ft, target.open("rb") as fg:
            assert ft.read(align(theader)) == fg.read(align(gheader))
            for name, row in gr.items():
                ft.seek(align(theader) + row["offset"])
                fg.seek(align(gheader) + row["offset"])
                a, b = ft.read(row["bytes"]), fg.read(row["bytes"])
                if row["storage"] != "PQ2_0":
                    assert a == b, name
                    continue
                zc = zero_counts(b)
                zero_hist.update(zc.tolist())
                if nonzeros is not None:
                    assert np.all((zc == 128 - nonzeros) | (zc == 128)), name
                for g in np.linspace(0, len(b) // 34 - 1, 8).astype(int):
                    block = b[34 * g:34 * g + 34]
                    assert np.array_equal(python_decode(block), c_decode(block)), name
                    sampled += 1
        total = sum(zero_hist.values())
        records.append({"arm": arm, "bytes": target.stat().st_size, "tensor_types": dict(types),
                        "non_pq2_payloads_identical_to_template": 169, "groups": total,
                        "groups_with_42_zeros": zero_hist[42] / total,
                        "mean_zero_fraction": sum(k * v for k, v in zero_hist.items()) / total / 128,
                        "sampled_groups_c_python_identical": sampled})
    (OUT / "validation.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()

"""Check the constructed 1.7B arms against their templates: header and tensor layout equal, the 113 F32
payloads byte-identical, every PQ2_0 payload rewritten, sampled groups identical in the Python and pinned
C decoders, and the byte ceiling from results/q17b/protocol.json."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from codec_probe import c_codec, python_decode
from phase0 import align

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work/qwen3_17b"
OUT = ROOT / "results/q17b"
sys.path.insert(0, str(ROOT.parent))
from inspect_gguf import inspect  # noqa: E402

ARMS = {"ls": ("base-pq2.gguf", "base-pq2-ls.gguf"), "gptq": ("base-pq2.gguf", "base-pq2-gptq.gguf"),
        "gptqh": ("folded-pq2.gguf", "folded-pq2-gptq.gguf")}


def main() -> None:
    _, c_decode = c_codec()
    ceiling = (WORK / "prism_gguf/Ternary-Bonsai-1.7B-PQ2_0.gguf").stat().st_size + 1_000_000
    records = []
    for arm, (tpl, tgt) in ARMS.items():
        template, target = WORK / tpl, WORK / tgt
        _, _, trows, theader = inspect(template)
        _, _, grows, gheader = inspect(target)
        tr, gr = {r["tensor"]: r for r in trows}, {r["tensor"]: r for r in grows}
        assert theader == gheader and target.stat().st_size == template.stat().st_size <= ceiling
        assert {(n, r["storage"], r["elements"], r["offset"]) for n, r in gr.items()} == {
            (n, r["storage"], r["elements"], r["offset"]) for n, r in tr.items()}
        types = Counter(r["storage"] for r in gr.values())
        assert types == {"PQ2_0": 197, "F32": 113}
        changed = sampled = identical = 0
        with template.open("rb") as ft, target.open("rb") as fg:
            assert ft.read(align(theader)) == fg.read(align(gheader))
            for name, row in gr.items():
                ft.seek(align(theader) + row["offset"])
                fg.seek(align(gheader) + row["offset"])
                a, b = ft.read(row["bytes"]), fg.read(row["bytes"])
                if row["storage"] != "PQ2_0":
                    assert a == b, name
                    identical += 1
                    continue
                changed += a != b
                for g in np.linspace(0, row["bytes"] // 34 - 1, 8).astype(int):
                    block = b[34 * g:34 * g + 34]
                    dec = python_decode(block)
                    assert np.array_equal(dec, c_decode(block)) and np.all(np.isfinite(dec)), name
                    sampled += 1
        assert changed == 197
        records.append({"arm": arm, "file": tgt, "bytes": target.stat().st_size, "byte_ceiling": ceiling,
                        "tensor_types": dict(types), "f32_payloads_identical_to_template": identical,
                        "pq2_payloads_changed_from_absmax_template": changed,
                        "sampled_groups_c_python_identical": sampled})
    (OUT / "validation.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()

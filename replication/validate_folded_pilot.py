"""Validate small-pilot GGUF metadata, tensor inventory, and unchanged tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from inspect_gguf import HeaderReader, inspect  # noqa: E402


def align(x: int, n: int = 32) -> int:
    return (x + n - 1) // n * n


def header_metadata(path: Path) -> dict:
    wanted = {"prism.hadamard.version", "prism.hadamard.block_size", "prism.hadamard.transform",
              "prism.hadamard.axis", "prism.hadamard.sign_mode", "prism.hadamard.sign_widths",
              "prism.hadamard.sign_values", "prism.hadamard.weight_names",
              "prism.hadamard.inverse_weight_names", "prism.hadamard.tied_output",
              "prism.hadamard.gdn_v_grouped"}
    with path.open("rb") as f:
        reader = HeaderReader(f)
        assert f.read(4) == b"GGUF"
        reader.scalar("I")
        reader.scalar("Q")
        count = reader.scalar("Q")
        metadata = {}
        for _ in range(count):
            name, kind = reader.string(), reader.scalar("I")
            if name in wanted:
                metadata[name] = reader.value(kind)
            else:
                reader.value(kind, False)
    return metadata


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, default=HERE / "work/qwen35_08b")
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument("--grouped", action="store_true", help="validate the 3:1 V:K derivative")
    variant.add_argument("--h1024", action="store_true", help="validate the H1024 partial-fold pilot")
    parser.add_argument("--output", type=Path, default=HERE / "results/fp_pilot_validation.json")
    args = parser.parse_args()
    prefix = "grouped-" if args.grouped else "h1024-" if args.h1024 else ""
    base = args.work / ("grouped-base-f32.gguf" if args.grouped else "base-f32.gguf")
    folded = args.work / (prefix + "folded-f32.gguf")
    _, base_meta, base_rows, base_header = inspect(base)
    _, fold_meta, fold_rows, fold_header = inspect(folded)
    assert base_meta["general.architecture"] == fold_meta["general.architecture"] == "qwen35"
    a = {r["tensor"]: r for r in base_rows}
    b = {r["tensor"]: r for r in fold_rows}
    assert len(a) == len(b) == 320 and set(a) == set(b)
    metadata = header_metadata(folded)
    assert metadata["prism.hadamard.version"] == 2
    assert metadata["prism.hadamard.tied_output"] is True
    assert metadata["prism.hadamard.block_size"] == (1024 if args.h1024 else 512)
    assert metadata["prism.hadamard.sign_mode"] == "explicit"
    expected_widths = ([1024, 1536, 2048, 3584] if args.grouped else
                       [1024, 2048] if args.h1024 else [1024, 2048, 3584])
    assert metadata["prism.hadamard.sign_widths"] == expected_widths
    assert metadata["prism.hadamard.sign_values"] == {"array_length": sum(expected_widths)}
    assert metadata["prism.hadamard.inverse_weight_names"] == ["token_embd.weight"]
    rotated = set(metadata["prism.hadamard.weight_names"])
    assert len(rotated) == (126 if args.h1024 else 150) and "output.weight" not in a
    if args.grouped:
        assert metadata["prism.hadamard.gdn_v_grouped"] is True
    else:
        assert "prism.hadamard.gdn_v_grouped" not in metadata  # nk == nv
    different = []
    same = []
    with base.open("rb") as af, folded.open("rb") as bf:
        with mmap.mmap(af.fileno(), 0, access=mmap.ACCESS_READ) as am, mmap.mmap(bf.fileno(), 0, access=mmap.ACCESS_READ) as bm:
            av, bv = memoryview(am), memoryview(bm)
            for name in sorted(a):
                x, y = a[name], b[name]
                assert (x["dimensions_gguf_order"], x["storage"], x["bytes"]) == (
                    y["dimensions_gguf_order"], y["storage"], y["bytes"])
                astart, bstart = align(base_header) + x["offset"], align(fold_header) + y["offset"]
                (same if av[astart:astart + x["bytes"]] == bv[bstart:bstart + y["bytes"]] else different).append(name)
            del av, bv
    assert set(different) == rotated | {"token_embd.weight"}
    assert len(same) == (193 if args.h1024 else 169)
    assert len(different) == (127 if args.h1024 else 151)
    result = {"base_sha256": sha256(base), "folded_sha256": sha256(folded),
              "tensor_count": 320, "unchanged_tensors_exact": len(same),
              "changed_tensors_exactly_expected": len(different), "metadata": metadata,
              "pilot_kind": "synthetic 3:1 V:K GDN derivative" if args.grouped else
                            "public equal-head checkpoint with H1024 partial fold" if args.h1024 else
                            "public equal-head checkpoint with H512 full fold",
              "pilot_limitation": "Synthetic cropped GDN weights are a graph fixture, not a quality-preserving model." if args.grouped else
                                  "FFN down input width 3584 is not divisible by H1024, so those 24 tensors remain unrotated." if args.h1024 else
                                  "Qwen3.5-0.8B has equal K and V head counts; grouped-V permutation is not exercised."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Validated {len(same)} unchanged and {len(different)} folded tensors")


if __name__ == "__main__":
    main()

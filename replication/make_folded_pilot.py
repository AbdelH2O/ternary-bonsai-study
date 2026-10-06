"""Fold a pinned small Qwen3.5 checkpoint for a native FP runtime identity check.

This pilot is a transform/graph check. It does not reproduce Bonsai's learned
codes or establish the 27B model's quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
WORK = Path(__file__).resolve().parent / "work/qwen35_08b"
BLOCK = 512
SEED = 20260930
FOLD_SUFFIXES = {
    "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
    "self_attn.o_proj.weight", "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight", "linear_attn.out_proj.weight",
    "mlp.down_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
}


def folded_role(name: str) -> str | None:
    if name == "model.language_model.embed_tokens.weight":
        return "inverse-after-lookup"
    if not re.fullmatch(r"model\.language_model\.layers\.\d+\..*", name):
        return None
    if any(name.endswith("." + suffix) for suffix in FOLD_SUFFIXES):
        return "fold-before-matmul"
    return None


def signs_for_width(width: int) -> np.ndarray:
    rng = np.random.default_rng(SEED + width)
    signs = rng.choice(np.array([-1, 1], dtype=np.int8), size=width)
    assert np.any(signs < 0) and np.any(signs > 0)
    return signs


def hadamard_rows(x: torch.Tensor, signs: np.ndarray, row_chunk: int = 512) -> torch.Tensor:
    assert x.ndim == 2 and x.shape[1] == len(signs) and len(signs) % BLOCK == 0
    rows, width = x.shape
    output = torch.empty((rows, width), dtype=torch.float32)
    for start in range(0, rows, row_chunk):
        end = min(start + row_chunk, rows)
        y = x[start:end].to(torch.float32).numpy().copy()
        y *= signs.astype(np.float32)
        blocks = y.reshape(end - start, width // BLOCK, BLOCK)
        for step in (1 << k for k in range(BLOCK.bit_length() - 1)):
            pairs = blocks.reshape(end - start, width // BLOCK, BLOCK // (2 * step), 2, step)
            a, b = pairs[..., 0, :].copy(), pairs[..., 1, :].copy()
            pairs[..., 0, :], pairs[..., 1, :] = a + b, a - b
        y *= np.float32(1 / np.sqrt(BLOCK))
        output[start:end] = torch.from_numpy(y)
    return output


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    global BLOCK
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, default=WORK / "base")
    p.add_argument("--out", type=Path, default=WORK / "folded")
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--skip-nondivisible", action="store_true")
    args = p.parse_args()
    BLOCK = args.block_size
    if BLOCK <= 0 or BLOCK & (BLOCK - 1):
        raise ValueError("block size must be a positive power of two")
    source = args.base / "model.safetensors-00001-of-00001.safetensors"
    if not source.exists():
        raise FileNotFoundError(source)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"):
        shutil.copy2(args.base / name, args.out / name)
    output: dict[str, torch.Tensor] = {}
    records = []
    signs: dict[int, np.ndarray] = {}
    source_widths = set()
    with safe_open(source, framework="pt", device="cpu") as f:
        names = sorted(k for k in f.keys() if k.startswith("model.language_model."))
        for i, name in enumerate(names, 1):
            role = folded_role(name)
            tensor = f.get_tensor(name)
            if role:
                assert tensor.ndim == 2
                width = tensor.shape[-1]
                if width % BLOCK:
                    if args.skip_nondivisible:
                        output[name] = tensor
                    else:
                        raise ValueError(f"cannot fold {name}: width {width} is not divisible by {BLOCK}")
                else:
                    signs.setdefault(width, signs_for_width(width))
                    output[name] = hadamard_rows(tensor, signs[width])
                    source_widths.add(width)
                    records.append({"name": name, "role": role, "axis": -1})
            else:
                output[name] = tensor
            if i % 40 == 0 or i == len(names):
                print(f"prepared {i}/{len(names)} language tensors; {len(records)} folded", flush=True)
    assert len(records) > 100 and sum(r["role"] == "inverse-after-lookup" for r in records) == 1
    assert any("linear_attn.out_proj" in r["name"] for r in records)
    manifest = {
        "schema_version": 3, "kind": "hadamard-weight-fold", "status": "requires-matching-runtime",
        "transform": {"name": "normalized-signed-sylvester-walsh-hadamard",
                      "block_size": BLOCK, "sign_mode": "explicit"},
        "signs": {str(w): signs[w].tolist() for w in sorted(source_widths)},
        "tied_output": True, "tensors": records,
    }
    (args.out / "hadamard_packing.json").write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    target = args.out / "model.safetensors"
    save_file(output, str(target))
    source_label = "grouped-V synthetic derivative of " + REPO if (args.base / "derivative_record.json").exists() else REPO
    print(json.dumps({"source": source_label, "parent_revision": REVISION,
                      "source_sha256": hash_file(source), "folded_sha256": hash_file(target),
                      "source_tensors": len(names), "folded_tensors": len(records),
                      "sign_widths": sorted(source_widths), "block_size": BLOCK}, indent=2))


if __name__ == "__main__":
    main()

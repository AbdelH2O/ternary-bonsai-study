"""Prepare tied and H1024-folded Qwen3-1.7B checkpoints for the pinned converter.

base_tied/: the pinned ancestor with the duplicate lm_head.weight removed (it equals
            embed_tokens bit for bit; tie_word_embeddings=true), BF16 values unchanged.
folded/:    all 196 decoder projections folded as W R^T and the tied embedding as E R^T,
            R = H S / sqrt(1024), explicit signs per width; hadamard_packing.json schema 3.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import make_folded_pilot as fold

WORK = Path(__file__).resolve().parent / "work/qwen3_17b"
FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json")
SUFFIXES = ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
            "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")


def load() -> dict[str, torch.Tensor]:
    index = json.loads((WORK / "base/model.safetensors.index.json").read_text())["weight_map"]
    tensors = {}
    for shard in sorted(set(index.values())):
        with safe_open(WORK / "base" / shard, framework="pt") as f:
            for name in f.keys():
                tensors[name] = f.get_tensor(name)
    assert torch.equal(tensors.pop("lm_head.weight"), tensors["model.embed_tokens.weight"])
    return tensors


def main() -> None:
    fold.BLOCK = 1024
    tensors = load()
    for sub in ("base_tied", "folded"):
        (WORK / sub).mkdir(exist_ok=True)
        for name in FILES:
            shutil.copy2(WORK / "base" / name, WORK / sub / name)
    save_file(tensors, str(WORK / "base_tied/model.safetensors"), metadata={"format": "pt"})
    signs, records, folded = {}, [], {}
    for name, t in tensors.items():
        if name == "model.embed_tokens.weight":
            role = "inverse-after-lookup"
        elif name.endswith(SUFFIXES):
            role = "fold-before-matmul"
        else:
            folded[name] = t
            continue
        width = t.shape[1]
        signs.setdefault(width, fold.signs_for_width(width))
        folded[name] = fold.hadamard_rows(t, signs[width], row_chunk=4096)
        records.append({"name": name, "role": role, "axis": -1})
    assert len(records) == 197 and sorted(signs) == [2048, 6144]
    manifest = {"schema_version": 3, "kind": "hadamard-weight-fold", "status": "requires-matching-runtime",
                "transform": {"name": "normalized-signed-sylvester-walsh-hadamard", "block_size": 1024,
                              "sign_mode": "explicit"},
                "signs": {str(w): signs[w].tolist() for w in sorted(signs)}, "tied_output": True, "tensors": records}
    (WORK / "folded/hadamard_packing.json").write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    save_file(folded, str(WORK / "folded/model.safetensors"), metadata={"format": "pt"})
    print(f"base_tied: {len(tensors)} tensors; folded: {len(records)} folded, sign seeds {fold.SEED}+width")


if __name__ == "__main__":
    main()

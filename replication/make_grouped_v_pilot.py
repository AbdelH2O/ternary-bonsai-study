"""Create a synthetic Qwen3.5-0.8B derivative with a 3:1 V:K GDN layout.

This is a graph fixture, not a quality-preserving model conversion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

OLD_K, OLD_V, NEW_K, NEW_V, HD = 16, 16, 4, 12, 128
WORK = Path(__file__).resolve().parent / "work/qwen35_08b"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def crop(name: str, t: torch.Tensor) -> torch.Tensor:
    if ".linear_attn." not in name:
        return t
    suffix = name.split(".linear_attn.", 1)[1]
    qk_old, qk_new = OLD_K * HD, NEW_K * HD
    v_new = NEW_V * HD
    if suffix == "in_proj_qkv.weight":
        assert t.shape[0] == qk_old * 2 + OLD_V * HD
        return torch.cat((t[:qk_new], t[qk_old:qk_old + qk_new],
                          t[2*qk_old:2*qk_old + v_new]), dim=0).contiguous()
    if suffix == "conv1d.weight":
        assert t.shape[0] == qk_old * 2 + OLD_V * HD
        return torch.cat((t[:qk_new], t[qk_old:qk_old + qk_new],
                          t[2*qk_old:2*qk_old + v_new]), dim=0).contiguous()
    if suffix == "in_proj_z.weight":
        return t[:v_new].contiguous()
    if suffix in ("in_proj_a.weight", "in_proj_b.weight", "A_log", "dt_bias"):
        return t[:NEW_V].contiguous()
    if suffix == "out_proj.weight":
        return t[:, :v_new].contiguous()
    return t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, default=WORK / "base")
    p.add_argument("--out", type=Path, default=WORK / "grouped-base")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.base / "config.json").read_text())
    assert config["text_config"]["linear_num_key_heads"] == OLD_K
    assert config["text_config"]["linear_num_value_heads"] == OLD_V
    config["text_config"]["linear_num_key_heads"] = NEW_K
    config["text_config"]["linear_num_value_heads"] = NEW_V
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for name in ("tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"):
        shutil.copy2(args.base / name, args.out / name)
    source = args.base / "model.safetensors-00001-of-00001.safetensors"
    tensors = {}
    changed = []
    with safe_open(source, framework="pt", device="cpu") as f:
        for name in sorted(k for k in f.keys() if k.startswith("model.language_model.")):
            old = f.get_tensor(name)
            new = crop(name, old)
            tensors[name] = new
            if new.shape != old.shape:
                changed.append({"name": name, "old_shape": list(old.shape), "new_shape": list(new.shape)})
    assert len(changed) == 18 * 8, len(changed)
    target = args.out / source.name
    save_file(tensors, str(target))
    record = {"source_sha256": sha256(source), "derivative_sha256": sha256(target),
              "source_key_heads": OLD_K, "source_value_heads": OLD_V,
              "derivative_key_heads": NEW_K, "derivative_value_heads": NEW_V,
              "head_dim": HD, "changed_tensors": changed}
    (args.out / "derivative_record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Wrote {len(tensors)} language tensors; cropped {len(changed)} GDN tensors")


if __name__ == "__main__":
    main()

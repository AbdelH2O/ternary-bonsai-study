"""Do Prism's ternary and 1-bit Qwen3-1.7B releases depart from the ancestor at the same weights beyond
what weight magnitude and a shared objective explain? CPU only; descriptive. Follows prism_variants_17b.py.

1. Prism, magnitude-conditioned: within bins of ancestor |w| / group mean |w|, compare
   P(1-bit flips | ternary flips) with P(1-bit flips | ternary keeps its sign).
2. Control from our own work: two *independent* 0.8B QAT runs from the same init and data
   (QAT08 qat_free, absmean; qat_42, top-86). The same statistic, in their folded H512 basis, shows how much
   coincidence a shared objective alone produces.

    python3 prism_lineage_17b.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

import prism_variants_17b as pv

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/prism_lineage_17b.json"
BINS = [0.0, 0.25, 0.5, 1.0, 2.0, math.inf]
LAYERS_17B = [0, 9, 18, 27]
ROLE08 = {"attn_qkv": "linear_attn.in_proj_qkv", "attn_gate": "linear_attn.in_proj_z", "ssm_out": "linear_attn.out_proj",
          "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
          "attn_output": "self_attn.o_proj", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}


def new_table():
    return {f"{BINS[i]}-{BINS[i + 1]}": {"ref_flip": 0, "ref_flip_and_other_flip": 0, "ref_keep": 0,
                                          "ref_keep_and_other_flip": 0, "ref_zero": 0, "ref_zero_and_other_flip": 0}
            for i in range(len(BINS) - 1)}


def accumulate(table: dict, ref_codes: np.ndarray, other_sign: np.ndarray, anc: np.ndarray) -> None:
    """ref_codes: ternary codes of the reference model; other_sign: +-1 of the other model; anc: ancestor (same basis)."""
    sa = np.where(anc >= 0, 1, -1).astype(np.int8)
    g = np.abs(anc).reshape(anc.shape[0], -1, 128)
    rel = (g / np.maximum(g.mean(-1, keepdims=True), 1e-12)).reshape(anc.shape)
    other_flip = other_sign != sa
    nz = ref_codes != 0
    ref_flip = nz & (ref_codes != sa)
    ref_keep = nz & (ref_codes == sa)
    for i, key in enumerate(table):
        m = (rel >= BINS[i]) & (rel < BINS[i + 1])
        t = table[key]
        for name, mask in (("ref_flip", ref_flip), ("ref_keep", ref_keep), ("ref_zero", ~nz)):
            mm = m & mask
            t[name] += int(mm.sum())
            t[f"{name}_and_other_flip"] += int((mm & other_flip).sum())


def finish(table: dict) -> dict:
    out = {}
    for key, t in table.items():
        out[key] = {
            "share_of_ref_flips": t["ref_flip"],
            "P_other_flips_given_ref_flips": t["ref_flip_and_other_flip"] / t["ref_flip"] if t["ref_flip"] else None,
            "P_other_flips_given_ref_keeps": t["ref_keep_and_other_flip"] / t["ref_keep"] if t["ref_keep"] else None,
            "P_other_flips_given_ref_zero": t["ref_zero_and_other_flip"] / t["ref_zero"] if t["ref_zero"] else None,
            "ref_flip_rate_among_nonzero": t["ref_flip"] / (t["ref_flip"] + t["ref_keep"]) if t["ref_flip"] + t["ref_keep"] else None,
        }
    total_flips = sum(t["ref_flip"] for t in table.values())
    for key in out:
        out[key]["share_of_ref_flips"] = table[key]["ref_flip"] / total_flips if total_flips else None
    return out


def prism_part() -> dict:
    readers = {k: pv.GGUFReader(p) for k, p in pv.FILES.items() if k != "t64"}
    tensors = {k: {t.name: t for t in r.tensors} for k, r in readers.items()}
    table = new_table()
    with pv._BaseReader(pv.BASE) as base:
        for layer in LAYERS_17B:
            for role in pv.ROLE:
                name = f"blk.{layer}.{role}.weight"
                c, _ = pv.decode(np.asarray(tensors["t128"][name].data), "t128")
                b, _ = pv.decode(np.asarray(tensors["b128"][name].data), "b128")
                accumulate(table, c, b, base.get(pv.hf_name(name)))
    return {"scope": f"layers {LAYERS_17B}, all 7 projection roles; reference = ternary g128, other = 1-bit",
            "bins_by_ancestor_rel_magnitude": finish(table)}


def fold(w: torch.Tensor, signs: torch.Tensor, block: int) -> torch.Tensor:
    """W R^T with R = H S / sqrt(B), blockwise on the input axis (as make_gptq_pilot.Hadamard.fold)."""
    rows, width = w.shape
    y = (w * signs).reshape(rows, width // block, block).clone()
    h = 1
    while h < block:
        y = y.reshape(rows, width // block, block // (2 * h), 2, h)
        a, b = y[..., 0, :].clone(), y[..., 1, :].clone()
        y[..., 0, :], y[..., 1, :] = a + b, a - b
        h *= 2
    return y.reshape(rows, width) / math.sqrt(block)


def quantize(w: torch.Tensor, rule: str) -> torch.Tensor:
    g = w.float().reshape(w.shape[0], -1, 128)
    if rule == "absmean":
        s = g.abs().mean(-1).half().float()
        safe = torch.where(s > 0, s, torch.ones_like(s))
        c = torch.clamp(torch.round(g / safe[..., None]), -1, 1) * (s > 0)[..., None]
    else:
        order = torch.sort(-g.abs(), dim=-1, stable=True).indices
        keep = torch.zeros_like(g, dtype=torch.bool).scatter_(-1, order[..., :86], True)
        c = torch.where(g >= 0, 1.0, -1.0) * keep
    return c.reshape(w.shape).to(torch.int8)


def control_part() -> dict:
    manifest = json.loads((ROOT / "work/qwen35_08b/folded/hadamard_packing.json").read_text())
    block = manifest["transform"]["block_size"]
    signs = {int(k): torch.tensor(v, dtype=torch.float32) for k, v in manifest["signs"].items()}
    free = torch.load(ROOT / "work/qat08/qat_free/latent_final.pt", map_location="cpu", mmap=True)
    z42 = torch.load(ROOT / "work/qat08/qat_42/latent_final.pt", map_location="cpu", mmap=True)
    base_file = next((ROOT / "work/qwen35_08b/base").glob("*.safetensors"))
    tables = {"ref_free_other_42": new_table(), "ref_42_other_free": new_table()}
    agree = {"free_nonzero_vs_anc": [0, 0], "42_nonzero_vs_anc": [0, 0], "free_vs_42_nonzero_both": [0, 0]}
    with safe_open(base_file, "pt") as f:
        for name in sorted(free):
            _, layer, role, _ = name.split(".")
            anc = fold(f.get_tensor(f"model.language_model.layers.{layer}.{ROLE08[role]}.weight").float(),
                       signs[free[name].shape[1]], block)
            lf, l42 = free[name].float(), z42[name].float()
            cf, c42 = quantize(lf, "absmean"), quantize(l42, "top86")
            sf, s42 = torch.where(lf >= 0, 1, -1).to(torch.int8), torch.where(l42 >= 0, 1, -1).to(torch.int8)
            a = anc.numpy()
            accumulate(tables["ref_free_other_42"], cf.numpy(), s42.numpy(), a)
            accumulate(tables["ref_42_other_free"], c42.numpy(), sf.numpy(), a)
            sa = torch.where(anc >= 0, 1, -1).to(torch.int8)
            for key, c in (("free_nonzero_vs_anc", cf), ("42_nonzero_vs_anc", c42)):
                nz = c != 0
                agree[key][0] += int((c[nz] == sa[nz]).sum()); agree[key][1] += int(nz.sum())
            both = (cf != 0) & (c42 != 0)
            agree["free_vs_42_nonzero_both"][0] += int((cf[both] == c42[both]).sum())
            agree["free_vs_42_nonzero_both"][1] += int(both.sum())
    return {"scope": "QAT08 qat_free (absmean) and qat_42 (top-86), 65.5M tokens each, same folded init and data, "
                     "all 150 projections, folded H512 basis; 'other' sign = sign of its latent",
            "sign_agreement": {k: v[0] / v[1] for k, v in agree.items()},
            **{k: finish(t) for k, t in tables.items()}}


def main() -> None:
    result = {"prism_17b": prism_part()}
    print(json.dumps(result, indent=1), flush=True)
    result["control_08b"] = control_part()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["control_08b"], indent=1))


if __name__ == "__main__":
    main()

"""Static cross-check of Prism's three Qwen3-1.7B releases: ternary g128 (PQ2_0), ternary g64 (Q2_0) and
1-bit (Q1_0). CPU only; descriptive.

Questions:
  1. Are the g128 and g64 ternary files projections of one shared latent? If both use BitNet absmean
     (scale = mean|L| per group), then s128 = (s64a + s64b) / 2 up to scale rounding, no element can have
     opposite nonzero signs, and every element's two codes must admit a common latent value.
  2. Is the 1-bit model a projection of the same latent? Under absmean for both, s1 = s128 per group and
     binary signs equal ternary nonzero signs.
  3. How far did each release move from the ancestor (sign agreement), and do their departures coincide?

    python3 prism_variants_17b.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "work/llama.cpp-842b1880415d6f508f03b789e5ce70194def7bfd/gguf-py"))
from gguf import GGUFReader  # noqa: E402

FILES = {"t128": ROOT / "work/qwen3_17b/prism_gguf/Ternary-Bonsai-1.7B-PQ2_0.gguf",
         "t64": ROOT / "work/prism17_variants/Ternary-Bonsai-1.7B-Q2_0_g64.gguf",
         "b128": ROOT / "work/prism17_variants/Bonsai-1.7B-Q1_0.gguf"}
BASE = ROOT / "work/qwen3_17b/base"
OUT = ROOT / "results/prism_variants_17b.json"
ROLE = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
        "attn_output": "self_attn.o_proj", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj"}
CHUNK = 4096


def hf_name(gguf: str) -> str:
    if gguf == "token_embd.weight":
        return "model.embed_tokens.weight"
    _, layer, role, _ = gguf.split(".")
    return f"model.layers.{layer}.{ROLE[role]}.weight"


def decode(raw: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """raw (rows, bytes) -> codes int8 (rows, cols) and fp16 scales as float32 (rows, groups)."""
    group = {"t128": 128, "t64": 64, "b128": 128}[kind]
    payload = group // 4 if kind != "b128" else group // 8
    blocks = raw.reshape(raw.shape[0], -1, 2 + payload)
    scales = blocks[..., :2].copy().view(np.float16)[..., 0].astype(np.float32)
    qs = blocks[..., 2:]
    if kind == "b128":
        bits = np.unpackbits(qs, axis=-1, bitorder="little")  # element j = bit j%8 of byte j//8
        codes = bits.astype(np.int8) * 2 - 1
    else:
        shifts = np.array([0, 2, 4, 6], dtype=np.uint8)
        codes = ((qs[..., :, None] >> shifts) & 3).reshape(*qs.shape[:-1], group).astype(np.int8) - 1
    return codes.reshape(raw.shape[0], -1), scales


def bf16_exact(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32).view(np.uint32) & 0xFFFF) == 0


class Acc:
    def __init__(self):
        self.n = defaultdict(float)

    def add(self, key: str, value) -> None:
        self.n[key] += float(value)


def compare(name: str, raws: dict, anc: np.ndarray, acc: Acc, ratios: dict) -> None:
    c128, s128 = decode(raws["t128"], "t128")
    c64, s64 = decode(raws["t64"], "t64")
    b, s1 = decode(raws["b128"], "b128")
    if np.any(np.abs(c128) > 1) or np.any(np.abs(c64) > 1):
        acc.add("code_plus2", (c128 > 1).sum() + (c64 > 1).sum())
    n = c128.size
    acc.add("elements", n)
    acc.add("groups128", s128.size)
    # 1. ternary g128 vs g64
    acc.add("t_code_agree", (c128 == c64).sum())
    acc.add("t_sign_contradiction", ((c128 * c64) == -1).sum())
    acc.add("zero128", (c128 == 0).sum())
    acc.add("zero64", (c64 == 0).sum())
    pair = s64.reshape(s64.shape[0], -1, 2)
    pred = pair.mean(-1)
    ok = (s128 > 0) & (pred > 0)
    ratios["s128_over_mean_s64"].append((s128[ok] / pred[ok]).astype(np.float32))
    # common-latent feasibility: interval of L/s implied by each code, intersected across group sizes
    s128e = np.repeat(s128, 128, axis=1)
    s64e = np.repeat(s64, 64, axis=1)

    def interval(c, s):
        lo = np.where(c == 1, 0.5 * s, np.where(c == 0, -0.5 * s, -np.inf))
        hi = np.where(c == -1, -0.5 * s, np.where(c == 0, 0.5 * s, np.inf))
        return lo, hi

    lo1, hi1 = interval(c128, s128e)
    lo2, hi2 = interval(c64, s64e)
    acc.add("t_infeasible", (np.maximum(lo1, lo2) > np.minimum(hi1, hi2)).sum())
    # 2. binary vs ternary
    nz = c128 != 0
    acc.add("t_nonzero", nz.sum())
    acc.add("b_agree_t_nonzero", (b[nz] == c128[nz]).sum())
    ok1 = (s128 > 0) & (s1 > 0)
    ratios["s1_over_s128"].append((s1[ok1] / s128[ok1]).astype(np.float32))
    # 3. ancestor sign agreement and coincident departures
    sa = np.where(anc >= 0, 1, -1).astype(np.int8)
    acc.add("t_agree_anc_nonzero", (c128[nz] == sa[nz]).sum())
    acc.add("b_agree_anc", (b == sa).sum())
    acc.add("b_agree_anc_on_t_nonzero", (b[nz] == sa[nz]).sum())
    flip_t = nz & (c128 != sa)
    acc.add("t_flips", flip_t.sum())
    acc.add("b_flips_where_t_flips", (b[flip_t] != sa[flip_t]).sum())
    keep_t = nz & (c128 == sa)
    acc.add("t_keeps", keep_t.sum())
    acc.add("b_flips_where_t_keeps", (b[keep_t] != sa[keep_t]).sum())
    # where ternary zeroed, which way did binary go relative to the ancestor?
    z = ~nz
    acc.add("b_agree_anc_on_t_zero", (b[z] == sa[z]).sum())
    acc.add("t_zero", z.sum())
    # ancestor magnitude at ternary zeros vs nonzeros, in units of the group's absmean
    g = np.abs(anc).reshape(anc.shape[0], -1, 128)
    rel = (g / np.maximum(g.mean(-1, keepdims=True), 1e-12)).reshape(anc.shape)
    acc.add("anc_rel_mag_at_t_zero", rel[z].sum())
    acc.add("anc_rel_mag_at_t_nonzero", rel[nz].sum())
    # scale fingerprints
    for k, s in (("t128", s128), ("t64", s64), ("b128", s1)):
        acc.add(f"{k}_scales", s.size)
        acc.add(f"{k}_scales_bf16_exact", bf16_exact(s).sum())


def _div(a: float, b: float) -> float | None:
    return a / b if b else None


def summarize(acc: Acc, ratios: dict) -> dict:
    n = acc.n
    out = {
        "elements": n["elements"],
        "ternary_g128_vs_g64": {
            "code_agreement": _div(n["t_code_agree"], n["elements"]),
            "opposite_nonzero_signs": _div(n["t_sign_contradiction"], n["elements"]),
            "no_common_latent_fraction": _div(n["t_infeasible"], n["elements"]),
            "zero_fraction_g128": _div(n["zero128"], n["elements"]),
            "zero_fraction_g64": _div(n["zero64"], n["elements"]),
        },
        "binary_vs_ternary_g128": {
            "sign_agreement_on_ternary_nonzeros": _div(n["b_agree_t_nonzero"], n["t_nonzero"]),
        },
        "vs_ancestor": {
            "ternary_nonzero_sign_agreement": _div(n["t_agree_anc_nonzero"], n["t_nonzero"]),
            "binary_sign_agreement_all": _div(n["b_agree_anc"], n["elements"]),
            "binary_sign_agreement_on_ternary_nonzeros": _div(n["b_agree_anc_on_t_nonzero"], n["t_nonzero"]),
            "binary_sign_agreement_on_ternary_zeros": _div(n["b_agree_anc_on_t_zero"], n["t_zero"]),
            "P_binary_flips_given_ternary_flips": _div(n["b_flips_where_t_flips"], n["t_flips"]),
            "P_binary_flips_given_ternary_keeps": _div(n["b_flips_where_t_keeps"], n["t_keeps"]),
            "ancestor_rel_magnitude_at_ternary_zeros": _div(n["anc_rel_mag_at_t_zero"], n["t_zero"]),
            "ancestor_rel_magnitude_at_ternary_nonzeros": _div(n["anc_rel_mag_at_t_nonzero"], n["t_nonzero"]),
        },
        "scale_bf16_exact_fraction": {k: _div(n[f"{k}_scales_bf16_exact"], n[f"{k}_scales"]) for k in ("t128", "t64", "b128")},
        "code_plus2_count": n.get("code_plus2", 0.0),
    }
    for key, parts in ratios.items():
        r = np.concatenate(parts)
        out[key] = {"groups": int(r.size), "median": float(np.median(r)),
                    "p01": float(np.percentile(r, 1)), "p99": float(np.percentile(r, 99)),
                    "within_0.5pct": float(np.mean(np.abs(r - 1) <= 0.005)),
                    "within_2pct": float(np.mean(np.abs(r - 1) <= 0.02))}
    return out


def main() -> None:
    readers = {k: GGUFReader(p) for k, p in FILES.items()}
    tensors = {k: {t.name: t for t in r.tensors} for k, r in readers.items()}
    names = sorted(n for n, t in tensors["t128"].items() if t.tensor_type.name == "PQ2_0")
    assert len(names) == 197
    total, by_kind = Acc(), defaultdict(Acc)
    ratios_total = defaultdict(list)
    ratios_kind = defaultdict(lambda: defaultdict(list))
    with _BaseReader(BASE) as base:
        for i, name in enumerate(names):
            kind = "embedding" if name == "token_embd.weight" else name.split(".")[2]
            anc_full = base.get(hf_name(name))
            rows = tensors["t128"][name].data.shape[0]
            for r0 in range(0, rows, CHUNK):
                raws = {k: np.asarray(tensors[k][name].data[r0:r0 + CHUNK]) for k in FILES}
                anc = anc_full[r0:r0 + raws["t128"].shape[0]]
                for acc, rat in ((total, ratios_total), (by_kind[kind], ratios_kind[kind])):
                    compare(name, raws, anc, acc, rat)
            if i % 28 == 0:
                print(f"{i + 1}/{len(names)} {name}", flush=True)
    result = {"files": {k: str(p.relative_to(ROOT)) for k, p in FILES.items()},
              "total": summarize(total, ratios_total),
              "by_kind": {k: summarize(a, ratios_kind[k]) for k, a in sorted(by_kind.items())}}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["total"], indent=2))


class _BaseReader:
    """Ancestor BF16 tensors from the sharded safetensors checkpoint, as float32 numpy."""

    def __init__(self, base: Path):
        index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
        self.files = {n: base / f for n, f in index.items()}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, name: str) -> np.ndarray:
        import torch
        with safe_open(self.files[name], "pt") as f:
            t = f.get_tensor(name).float().numpy()
        return t[:151669] if name == "model.embed_tokens.weight" else t


if __name__ == "__main__":
    main()

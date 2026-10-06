"""How far did training move the ternary codes from rotate-then-round of the ancestor? (descriptive, CPU only)

Context: an HF forensics thread reports ~92% of Bonsai 2 27B codes equal rotate-then-absmean-RTN of the ancestor,
with the ~8% trained deviations carrying all the quality. This measures the same quantities for our 0.8B students
(codes = top86 of the latent, in the H512 basis; ancestor = fold(W) exactly as at init) and for Prism's
Ternary-Bonsai-1.7B against Qwen3-1.7B (unrotated basis, as in PHASE1_17B.md).

Read-only over frozen checkpoints; not part of any sealed protocol.

    CUDA_VISIBLE_DEVICES= ../../../.venv/bin/python code_drift_probe.py   # writes results/code_drift/drift.json
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

import make_gptq_pilot

make_gptq_pilot.DEV = "cpu"
from make_gptq_pilot import GROUP, ROLE, Hadamard  # noqa: E402

torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/code_drift"
BASE08 = ROOT / "work/qwen35_08b"
CKPTS = {
    "qat_42 (plain text)": ROOT / "work/qat08/qat_42",
    "C qat_chat_42": ROOT / "work/qat08_chat/qat_chat_42",
    "M (MC data)": ROOT / "work/qat08_mcu/M",
    "U (trained FP tensors, partial)": ROOT / "work/qat08_mcu/U",
}
STAGES = ("t8m", "t16m", "t33m", "final")


def quantize(w: torch.Tensor, rule: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy of qat_08b.quantize (importing qat_08b would pull in the CUDA trainer)."""
    g = w.detach().float().reshape(w.shape[0], -1, GROUP)
    if rule == "absmean":
        s = g.abs().mean(-1).half().float()
        safe = torch.where(s > 0, s, torch.ones_like(s))
        c = torch.clamp(torch.round(g / safe[..., None]), -1, 1) * (s > 0)[..., None]
    elif rule == "top86":
        order = torch.sort(-g.abs(), dim=-1, stable=True).indices
        keep = torch.zeros_like(g, dtype=torch.bool).scatter_(-1, order[..., :86], True)
        s = ((g.abs() * keep).sum(-1) / 86).half().float()
        c = torch.where(g >= 0, 1.0, -1.0) * keep * (s > 0)[..., None]
    else:
        raise ValueError(rule)
    return c.reshape(w.shape).to(torch.int8), s


def dequant(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return (codes.float().reshape(codes.shape[0], -1, GROUP) * scales[..., None]).reshape(codes.shape)


def kind(name: str) -> str:
    return name.split(".")[-2]


def accumulate(acc: dict, codes: torch.Tensor, scales: torch.Tensor, w0: torch.Tensor,
               ref_init: torch.Tensor, ref_rtn: torch.Tensor, latent: torch.Tensor | None) -> None:
    nz = codes != 0
    acc["n"] += codes.numel()
    acc["agree_init_rule"] += (codes == ref_init).sum().item()
    acc["agree_absmean_rtn"] += (codes == ref_rtn).sum().item()
    acc["nz"] += nz.sum().item()
    acc["nz_sign_agree"] += (torch.sign(w0).to(torch.int8)[nz] == codes[nz]).sum().item()
    acc["support_flip"] += (nz != (ref_init != 0)).sum().item()
    acc["sign_flip_both_nz"] += ((codes != ref_init) & nz & (ref_init != 0)).sum().item()
    d = dequant(codes, scales) - w0
    acc["d_eff_sq"] += (d * d).sum().item()
    acc["w_sq"] += (w0 * w0).sum().item()
    if latent is not None:
        dl = latent - w0
        acc["d_lat_sq"] += (dl * dl).sum().item()


def summarize(acc: dict) -> dict:
    out = {
        "agree_with_init_rule": acc["agree_init_rule"] / acc["n"],
        "agree_with_absmean_rtn": acc["agree_absmean_rtn"] / acc["n"],
        "zero_fraction": 1 - acc["nz"] / acc["n"],
        "nz_sign_agreement_with_ancestor": acc["nz_sign_agree"] / acc["nz"],
        "support_changed_vs_init_rule": acc["support_flip"] / acc["n"],
        "sign_flipped_vs_init_rule": acc["sign_flip_both_nz"] / acc["n"],
        "rel_effective_weight_change": (acc["d_eff_sq"] / acc["w_sq"]) ** 0.5,
    }
    if "d_lat_sq" in acc:
        out["rel_latent_change"] = (acc["d_lat_sq"] / acc["w_sq"]) ** 0.5
    return out


def ancestor_08b() -> dict[str, torch.Tensor]:
    rot = Hadamard(BASE08 / "folded/hadamard_packing.json")
    names = sorted(torch.load(CKPTS["qat_42 (plain text)"] / "latent_final.pt", map_location="cpu", mmap=True))
    st = safe_open(BASE08 / "base/model.safetensors-00001-of-00001.safetensors", framework="pt")
    out = {}
    for n in names:
        m = re.fullmatch(r"blk\.(\d+)\.(\w+)\.weight", n)
        key = f"model.language_model.layers.{m[1]}.{ROLE[m[2]]}.weight"
        out[n] = rot.fold(st.get_tensor(key).float())
    return out


def students() -> dict:
    w0s = ancestor_08b()
    refs = {n: (quantize(w, "top86")[0], quantize(w, "absmean")[0]) for n, w in w0s.items()}
    result = {}
    # untrained baseline: top86 of the ancestor itself
    tot, by_kind = defaultdict(float), defaultdict(lambda: defaultdict(float))
    for n, w0 in w0s.items():
        c, s = quantize(w0, "top86")
        for a in (tot, by_kind[kind(n)]):
            accumulate(a, c, s, w0, *refs[n], w0)
    result["init (top86 of ancestor, no training)"] = {"overall": summarize(tot)}
    for arm, d in CKPTS.items():
        result[arm] = {}
        for stage in STAGES:
            path = d / f"latent_{stage}.pt"
            if not path.exists():
                continue
            lat = torch.load(path, map_location="cpu", mmap=True)
            tot, by_kind, by_block = defaultdict(float), defaultdict(lambda: defaultdict(float)), defaultdict(lambda: defaultdict(float))
            for n, w0 in w0s.items():
                wt = lat[n].float()
                c, s = quantize(wt, "top86")
                for a in (tot, by_kind[kind(n)], by_block[int(n.split(".")[1])]):
                    accumulate(a, c, s, w0, *refs[n], wt)
            result[arm][stage] = {"overall": summarize(tot),
                                  "by_kind": {k: summarize(v) for k, v in sorted(by_kind.items())},
                                  "by_block": {b: summarize(v) for b, v in sorted(by_block.items())}}
            print(f"{arm} {stage}: " + json.dumps({k: round(v, 4) for k, v in result[arm][stage]["overall"].items()}), flush=True)
            del lat
    return result


def prism_17b() -> dict:
    """Prism Ternary-Bonsai-1.7B (gen 1, unrotated) codes vs Qwen3-1.7B, same metrics."""
    w17 = ROOT / "work/qwen3_17b"
    anc = safe_open(w17 / "base_tied/model.safetensors", framework="pt")
    pri = safe_open(w17 / "bonsai_unpacked/model.safetensors", framework="pt")
    proj = [k for k in pri.keys() if re.search(r"layers\.\d+\.(self_attn|mlp)\.\w+_proj\.weight$", k)]
    tot, by_kind = defaultdict(float), defaultdict(lambda: defaultdict(float))
    for k in sorted(proj):
        w0 = anc.get_tensor(k).float()
        p = pri.get_tensor(k).float()
        g = p.reshape(p.shape[0], -1, GROUP)
        scales = g.abs().amax(-1)
        codes = torch.where(g == 0, 0, torch.sign(g)).to(torch.int8).reshape(p.shape)
        ref_top = quantize(w0, "top86")[0]
        ref_rtn = quantize(w0, "absmean")[0]
        for a in (tot, by_kind[k.split(".")[-2]]):
            accumulate(a, codes, scales, w0, ref_top, ref_rtn, None)
    return {"overall": summarize(tot), "by_kind": {k: summarize(v) for k, v in sorted(by_kind.items())},
            "note": "init_rule here = top86 of the ancestor (Prism's zero count differs, so this is a loose reference); "
                    "absmean_rtn is the HF-forensics comparison. Embedding excluded (it is the absmean projection)."}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"students_08b": students()}
    res["prism_ternary_1p7b"] = prism_17b()
    print("prism 1.7B: " + json.dumps({k: round(v, 4) for k, v in res["prism_ternary_1p7b"]["overall"].items()}))
    (OUT / "drift.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

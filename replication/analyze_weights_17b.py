"""Descriptive static comparison of Prism's Ternary-Bonsai-1.7B with its Qwen3-1.7B ancestor and our arms.

For each of the 197 ternary tensors: Prism's zero rate and per-group zero counts, nonzero sign agreement
with the ancestor, support overlap with (a) the top-|w| rule at Prism's own per-group count, (b) our unrotated LS
and GPTQ codes (the H1024 arm is compared through its unfolded effective weights only), scale ratios, relative weight change, and layer output error tr(D H D^T)/tr(W H W^T)
under Hessians collected from the FP ancestor on the frozen calibration windows (so every arm is judged
on the same inputs; our GPTQ arm was built with sequential packed-input Hessians instead).

    python3 analyze_weights_17b.py      # CUDA; writes results/q17b/weights.json
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM

from make_gptq_17b import ROLE, collect_head
from make_gptq_pilot import DEV, GROUP, Hadamard, collect, inspect, sha
from phase0 import align

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work/qwen3_17b"
OUT = ROOT / "results/q17b"
ARMS = {"ls": "base-pq2-ls.gguf", "gptq": "base-pq2-gptq.gguf", "gptqh": "folded-pq2-gptq.gguf"}
ROTATED = {"gptqh"}  # H1024 basis: compared through unfolded effective weights only
CHUNK = 16384
VOCAB = 151669


def decode(path: Path) -> dict[str, torch.Tensor]:
    _, _, rows, header = inspect(path)
    out = {}
    with path.open("rb") as f:
        for r in rows:
            if r["storage"] != "PQ2_0":
                continue
            f.seek(align(header) + r["offset"])
            raw = np.frombuffer(f.read(r["bytes"]), dtype=np.uint8).reshape(-1, 34)
            scale = raw[:, :2].copy().view("<f2").astype(np.float32)
            b = raw[:, 2:]
            codes = np.stack([(b >> s) & 3 for s in (0, 2, 4, 6)], axis=2).reshape(-1, GROUP).astype(np.int8) - 1
            cols, nrows = (int(x) for x in r["dimensions_gguf_order"].split("x"))
            out[r["tensor"]] = (torch.from_numpy(codes).reshape(nrows, cols), torch.from_numpy(scale).reshape(nrows, -1))
    return out


def hf_key(gguf: str) -> str:
    if gguf == "token_embd.weight":
        return "model.embed_tokens.weight"
    m = re.fullmatch(r"blk\.(\d+)\.(\w+)\.weight", gguf)
    return f"model.layers.{m[1]}.{ROLE[m[2]]}.weight"


def ternary(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    g = w.reshape(w.shape[0], -1, GROUP)
    scale = g.abs().amax(2)
    codes = torch.where(g == 0, 0, torch.sign(g)).to(torch.int8).reshape(w.shape)
    return codes, scale


def effective(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (codes.float().reshape(codes.shape[0], -1, GROUP) * scale[..., None]).reshape(codes.shape)


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    protocol = json.loads((OUT / "protocol.json").read_text())
    ids = torch.from_numpy(np.load(OUT / protocol["calibration"]["ids_file"]).astype(np.int64)).to(DEV)
    ours = {arm: decode(WORK / f) for arm, f in ARMS.items()}
    rot = Hadamard(WORK / "folded/hadamard_packing.json")
    model = AutoModelForCausalLM.from_pretrained(WORK / "base_tied", dtype=torch.float32).float().to(DEV).eval()
    linears = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    prism = safe_open(WORK / "bonsai_unpacked/model.safetensors", framework="pt")
    rows = []

    def analyse(name: str, h: torch.Tensor) -> None:
        """Accumulate every metric over row chunks so the 151669-row embedding fits next to the model."""
        w_all = model.get_input_embeddings().weight.detach()[:VOCAB] if name == "token_embd.weight" else \
            linears[hf_key(name)[:-7]].weight.detach()
        p_all = prism.get_tensor(hf_key(name))[:w_all.shape[0]]
        n_rows = w_all.shape[0]
        acc = defaultdict(float)
        pz_all, ratios = [], defaultdict(list)
        for r0 in range(0, n_rows, CHUNK):
            w = w_all[r0:r0 + CHUNK]
            p_codes, p_scale = ternary(p_all[r0:r0 + CHUNK].to(DEV).float())
            pz = (p_codes == 0).reshape(-1, GROUP).sum(1).float()
            pz_all.append(pz.cpu())
            p_nz = p_codes != 0
            acc["p_nz"] += p_nz.sum().item()
            acc["sign_match"] += (torch.sign(w)[p_nz] == p_codes[p_nz]).sum().item()
            g = w.abs().reshape(-1, GROUP)
            order = torch.argsort(g, dim=1, descending=True, stable=True)
            top = torch.zeros_like(g, dtype=torch.bool)
            top.scatter_(1, order, torch.arange(GROUP, device=DEV)[None, :] < (GROUP - pz).long()[:, None])
            pg = p_nz.reshape(-1, GROUP)
            acc["top_inter"] += (pg & top).sum().item()
            acc["top_union"] += (pg | top).sum().item()
            del order, top, g
            acc["w_sq"] += (w * w).sum().item()
            acc["denom"] += ((w @ h) * w).sum().item()
            effs = {"prism": effective(p_codes, p_scale)}
            for arm, dec in ours.items():
                c, sc = dec[name]
                r1 = min(r0 + CHUNK, n_rows)  # our arms keep Qwen's 151936 padded rows
                c, sc = c[r0:r1].to(DEV), sc[r0:r1].to(DEV)
                q = effective(c, sc)
                effs[arm] = rot.unfold(q) if arm in ROTATED else q
                if arm in ROTATED:
                    continue  # codes live in the rotated basis; only effective-weight metrics are comparable
                nz = c != 0
                acc[f"{arm}_nz"] += nz.sum().item()
                acc[f"{arm}_inter"] += (nz & p_nz).sum().item()
                acc[f"{arm}_union"] += (nz | p_nz).sum().item()
                both = nz & p_nz
                acc[f"{arm}_both"] += both.sum().item()
                acc[f"{arm}_both_match"] += (c[both] == p_codes[both]).sum().item()
                ratios[arm].append((p_scale / sc.clamp(min=1e-12))[(sc > 0) & (p_scale > 0)].cpu())
            for arm, q in effs.items():
                d = q - w
                acc[f"{arm}_d_sq"] += (d * d).sum().item()
                acc[f"{arm}_err"] += ((d @ h) * d).sum().item()
            del effs
        numel = w_all.numel()
        pz = torch.cat(pz_all)
        rec = {"tensor": name, "kind": name.split(".")[-2] if name != "token_embd.weight" else "token_embd",
               "block": int(name.split(".")[1]) if name.startswith("blk.") else None, "numel": numel,
               "prism_zero_fraction": 1 - acc["p_nz"] / numel,
               "prism_group_zeros_q05_q50_q95": torch.quantile(pz[:1_000_000], torch.tensor([.05, .5, .95])).tolist(),
               "prism_sign_agreement_with_ancestor": acc["sign_match"] / acc["p_nz"],
               "prism_support_jaccard_topk_same_count": acc["top_inter"] / acc["top_union"],
               "rel_weight_change": {}, "rel_output_error_fp_hessian": {}}
        for arm in ("prism", *ours):
            rec["rel_weight_change"][arm] = (acc[f"{arm}_d_sq"] / acc["w_sq"]) ** 0.5
            rec["rel_output_error_fp_hessian"][arm] = acc[f"{arm}_err"] / acc["denom"]
            if arm == "prism" or arm in ROTATED:
                continue
            rec[f"{arm}_zero_fraction"] = 1 - acc[f"{arm}_nz"] / numel
            rec[f"{arm}_support_jaccard_with_prism"] = acc[f"{arm}_inter"] / acc[f"{arm}_union"]
            rec[f"{arm}_sign_agreement_with_prism"] = acc[f"{arm}_both_match"] / acc[f"{arm}_both"]
            rec[f"prism_over_{arm}_scale_median"] = torch.cat(ratios[arm]).median().item()
        rows.append(rec)
        e = rec["rel_output_error_fp_hessian"]
        print(f"{name}: prism zeros {rec['prism_zero_fraction']:.3f} sign {rec['prism_sign_agreement_with_ancestor']:.4f} "
              f"err prism {e['prism']:.4f} gptqh {e['gptqh']:.4f} gptq {e['gptq']:.4f} ls {e['ls']:.4f}", flush=True)

    with torch.no_grad():
        for i, layer in enumerate(model.model.layers):
            names = [f"blk.{i}.{k}.weight" for k in ROLE]
            hessians = collect(model, ids, {hf_key(n)[:-7]: linears[hf_key(n)[:-7]] for n in names}, stop_at=layer)
            for n in names:
                analyse(n, hessians[hf_key(n)[:-7]])
        h_head = collect_head(model, ids)
        analyse("token_embd.weight", h_head)

    def weighted(key, sub=None):
        tot = sum(r["numel"] for r in rows)
        return sum((r[key][sub] if sub else r[key]) * r["numel"] for r in rows) / tot

    by_kind = defaultdict(list)
    for r in rows:
        by_kind[r["kind"]].append(r)
    summary = {
        "parameter_weighted": {
            "prism_zero_fraction": weighted("prism_zero_fraction"),
            "prism_sign_agreement_with_ancestor": weighted("prism_sign_agreement_with_ancestor"),
            "prism_support_jaccard_topk_same_count": weighted("prism_support_jaccard_topk_same_count"),
            **{f"{a}_{m}": weighted(f"{a}_{m}") for a in ARMS if a not in ROTATED for m in
               ("zero_fraction", "support_jaccard_with_prism", "sign_agreement_with_prism")},
            "rel_weight_change": {a: weighted("rel_weight_change", a) for a in ("prism", *ARMS)},
        },
        "median_rel_output_error_fp_hessian": {a: float(np.median([r["rel_output_error_fp_hessian"][a] for r in rows]))
                                               for a in ("prism", *ARMS)},
        "by_kind": {k: {"prism_sign_agreement_with_ancestor": float(np.mean([r["prism_sign_agreement_with_ancestor"] for r in v])),
                        "prism_zero_fraction": float(np.mean([r["prism_zero_fraction"] for r in v])),
                        "rel_output_error_fp_hessian": {a: float(np.median([r["rel_output_error_fp_hessian"][a] for r in v]))
                                                        for a in ("prism", *ARMS)}}
                    for k, v in sorted(by_kind.items())},
        "by_depth_sign_agreement": {i: float(np.mean([r["prism_sign_agreement_with_ancestor"] for r in rows if r["block"] == i]))
                                    for i in range(28)},
    }
    record = {"note": "descriptive; not a held-out gate", "protocol_sha256": sha(OUT / "protocol.json"),
              "summary": summary, "tensors": rows}
    (OUT / "weights.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(summary["parameter_weighted"], indent=2))
    print(json.dumps(summary["median_rel_output_error_fp_hessian"], indent=2))


if __name__ == "__main__":
    main()

"""Build the frozen refinement arms of results/gptq_v2/protocol.json (folded basis, CUDA).

    python3 make_gptq_v2.py folded-pq2-gptq42 folded-pq2-gptqv2 folded-pq2-gptq42v2
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from make_gptq_pilot import (DAMP, DEV, GROUP, Hadamard, calibration_nll, collect, hf_name, inspect,
                             ls_quantize, ls_scale, pack, python_decode, rel_error, sha)
from phase0 import align

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work/qwen35_08b"
OUT = ROOT / "results/gptq_v2"
ARMS = {  # arm: (nonzeros per group or None, reconstruction version)
    "folded-pq2-gptq42": (86, "v1"),
    "folded-pq2-gptqv2": (None, "v2"),
    "folded-pq2-gptq42v2": (86, "v2"),
}
STAGES = {
    "linear_attention": [["attn_qkv", "attn_gate"], ["ssm_out"], ["ffn_gate", "ffn_up"], ["ffn_down"]],
    "full_attention": [["attn_q", "attn_k", "attn_v"], ["attn_output"], ["ffn_gate", "ffn_up"], ["ffn_down"]],
}


def budget_support(w1: torch.Tensor, nonzeros: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the `nonzeros` largest |w| per row (stable, lower index wins ties); scale = their mean, FP16."""
    order = torch.sort(-w1.abs(), dim=1, stable=True).indices
    keep = torch.zeros_like(w1, dtype=torch.bool)
    keep.scatter_(1, order[:, :nonzeros], True)
    scale = (w1.abs().double() * keep).sum(1) / nonzeros
    return keep, scale.float().half().float()


def gptq2(w: torch.Tensor, h: torch.Tensor, nonzeros: int | None):
    """GPTQ with the v1 free-budget rule, or an exact fixed support chosen at each group start."""
    w = w.clone().float()
    h = h.clone().float()
    rows, cols = w.shape
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    damp = DAMP * torch.mean(torch.diag(h)).item()
    for _ in range(4):
        try:
            hd = h + damp * torch.eye(cols, device=DEV)
            hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(hd)), upper=True)
            break
        except torch.linalg.LinAlgError:
            damp *= 10
    else:
        raise RuntimeError("Hessian not positive definite after damping")
    q = torch.zeros_like(w)
    codes = torch.zeros((rows, cols), dtype=torch.int8, device=DEV)
    scales = torch.zeros((rows, cols // GROUP), dtype=torch.float32, device=DEV)
    for i1 in range(0, cols, GROUP):
        i2 = i1 + GROUP
        w1 = w[:, i1:i2].clone()
        err1 = torch.zeros_like(w1)
        hinv1 = hinv[i1:i2, i1:i2]
        if nonzeros is None:
            s = ls_scale(w1)
            keep = None
        else:
            keep, s = budget_support(w1, nonzeros)
        scales[:, i1 // GROUP] = s
        safe = torch.where(s > 0, s, torch.ones_like(s))
        for i in range(GROUP):
            col = w1[:, i]
            if keep is None:
                c = torch.clamp(torch.round(col / safe), -1, 1) * (s > 0)
            else:
                c = torch.where(col >= 0, 1.0, -1.0) * keep[:, i] * (s > 0)
            qc = c * s
            q[:, i1 + i] = qc
            codes[:, i1 + i] = c.to(torch.int8)
            e = (col - qc) / hinv1[i, i]
            w1[:, i:] -= e[:, None] * hinv1[i, i:][None, :]
            err1[:, i] = e
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]
    return q, codes, scales, damp


def build(arm: str) -> dict:
    torch.backends.cuda.matmul.allow_tf32 = False
    nonzeros, version = ARMS[arm]
    protocol = json.loads((OUT / "protocol.json").read_text())
    if version == "v1":
        v1 = json.loads((ROOT / "results/gptq_pilot/protocol.json").read_text())
        ids_path = ROOT / "results/gptq_pilot" / v1["calibration"]["ids_file"]
        assert sha(ids_path) == v1["calibration"]["ids_sha256"]
    else:
        ids_path = OUT / protocol["calibration_v2"]["ids_file"]
        assert sha(ids_path) == protocol["calibration_v2"]["ids_sha256"]
    ids = torch.from_numpy(np.load(ids_path).astype(np.int64)).to(DEV)
    source, template, target = WORK / "folded-f32.gguf", WORK / "folded-pq2.gguf", WORK / f"{arm}.gguf"
    _, _, src_rows, src_header = inspect(source)
    _, _, tpl_rows, tpl_header = inspect(template)
    src = {r["tensor"]: r for r in src_rows}
    tpl = {r["tensor"]: r for r in tpl_rows}
    pq = sorted(n for n, r in tpl.items() if r["storage"] == "PQ2_0")
    assert len(pq) == 151
    rot = Hadamard(WORK / "folded/hadamard_packing.json")
    model = AutoModelForCausalLM.from_pretrained(WORK / "base", dtype=torch.float32).float().to(DEV).eval()
    layer_types = model.config.layer_types
    linears = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    shutil.copyfile(template, target)
    fp_nll = calibration_nll(model, ids)
    written, records = set(), []
    t0 = time.time()

    def read_source(name: str, shape) -> torch.Tensor:
        r = src[name]
        with source.open("rb") as f:
            f.seek(align(src_header) + r["offset"])
            data = f.read(r["elements"] * 4)
        return torch.from_numpy(np.frombuffer(data, dtype="<f4").copy()).reshape(shape).to(DEV)

    def quantize(names: list[str], hessians: dict, out) -> None:
        for name in names:
            module = linears[hf_name(name)]
            w_hf = module.weight.detach().float()
            w = read_source(name, tuple(w_hf.shape))
            assert torch.allclose(rot.fold(w_hf), w, rtol=0, atol=1e-5), name
            h = rot.hessian(hessians[hf_name(name)])
            q, codes, scales, damp = gptq2(w, h, nonzeros)
            if nonzeros is not None:
                zeros = (codes.reshape(-1, GROUP) == 0).sum(1)
                assert torch.all((zeros == GROUP - nonzeros) | (zeros == GROUP)), name
            payload = pack(codes, scales)
            r = tpl[name]
            assert len(payload) == r["bytes"]
            out.seek(align(tpl_header) + r["offset"])
            out.write(payload)
            for g in (0, len(payload) // 34 - 1):
                dec = python_decode(payload[34 * g:34 * g + 34])
                assert np.array_equal(dec, q.reshape(-1, GROUP)[g].cpu().numpy()), (name, g)
            with torch.no_grad():
                module.weight.copy_(rot.unfold(q).to(module.weight.dtype))
            records.append({"tensor": name, "damping": damp,
                            "rel_output_error": rel_error(w, q, h),
                            "rel_output_error_ls": rel_error(w, ls_quantize(w), h),
                            "zero_fraction": (codes == 0).float().mean().item()})
            written.add(name)
            print(f"{arm}: {name} err {records[-1]['rel_output_error']:.4f} "
                  f"zeros {records[-1]['zero_fraction']:.3f} ({time.time() - t0:.0f}s)", flush=True)

    with target.open("r+b") as out:
        for i, layer in enumerate(model.model.layers):
            names = {n.split(".")[2]: n for n in pq if n.startswith(f"blk.{i}.")}
            stages = STAGES[layer_types[i]] if version == "v2" else [sorted(names)]
            assert sorted(k for s in stages for k in s) == sorted(names), (i, names)
            for stage in stages:
                stage_names = [names[k] for k in stage]
                mods = {hf_name(n): linears[hf_name(n)] for n in stage_names}
                quantize(stage_names, collect(model, ids, mods, stop_at=layer), out)
        quantize(["token_embd.weight"], collect(model, ids, {"lm_head": linears["lm_head"]}, stop_at=None), out)
    assert written == set(pq)
    packed_nll = calibration_nll(model, ids)
    return {"arm": arm, "nonzeros_per_group": nonzeros, "reconstruction": version,
            "target": str(target.relative_to(ROOT)), "sha256": sha(target), "bytes": target.stat().st_size,
            "template_sha256": sha(template), "source_sha256": sha(source),
            "calibration_ids_sha256": sha(ids_path), "calibration_windows": int(ids.shape[0]),
            "protocol_sha256": sha(OUT / "protocol.json"), "builder_sha256": sha(Path(__file__)),
            "calibration_nll_fp": fp_nll, "calibration_nll_packed_torch": packed_nll,
            "calibration_nll_note": "in-sample diagnostic on this arm's calibration windows; not an evaluation",
            "seconds": time.time() - t0, "tensors": records,
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}


if __name__ == "__main__":
    for arm in sys.argv[1:] or list(ARMS):
        record = build(arm)
        (OUT / f"build_{arm}.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"{arm}: calibration NLL fp {record['calibration_nll_fp']:.4f} "
              f"packed {record['calibration_nll_packed_torch']:.4f}; {record['seconds']:.0f}s", flush=True)

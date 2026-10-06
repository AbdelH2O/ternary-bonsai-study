"""Build the frozen constructed PQ2_0 arms of results/q17b/protocol.json (CUDA).

    python3 make_gptq_17b.py [ls] [gptq] [gptqh]

ls:    data-free least-squares ternary rule on base_tied-f32.gguf
gptq:  0.8B-selected v1 GPTQ-style rule, unrotated
gptqh: same rule in the H1024 folded basis (folded-f32.gguf, Hessians R H R^T)
Helpers (GPTQ loop, LS rule, packing, Hessian collection) are imported unchanged from the 0.8B pilot.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from make_gptq_pilot import (DEV, GROUP, Hadamard, Stop, calibration_nll, collect, gptq, inspect, ls_quantize,
                             pack, python_decode, rel_error, sha)
from phase0 import align

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work/qwen3_17b"
OUT = ROOT / "results/q17b"
ROLE = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
        "attn_output": "self_attn.o_proj", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj"}
ARMS = {"ls": ("base", "base-pq2-ls"), "gptq": ("base", "base-pq2-gptq"), "gptqh": ("folded", "folded-pq2-gptq")}
ROW_CHUNK = 16384


def hf_name(gguf: str) -> str:
    if gguf == "token_embd.weight":
        return "lm_head"
    m = re.fullmatch(r"blk\.(\d+)\.(\w+)\.weight", gguf)
    return f"model.layers.{m[1]}.{ROLE[m[2]]}"


def collect_head(model, ids: torch.Tensor) -> torch.Tensor:
    """H = sum x x^T over the final-norm outputs, stopping before the vocabulary projection."""
    acc = []

    def hook(_, inputs):
        x = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
        acc.append((x.T @ x).double())
        raise Stop

    handle = model.lm_head.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            for start in range(0, len(ids), 8):
                try:
                    model(input_ids=ids[start:start + 8], use_cache=False)
                except Stop:
                    pass
    finally:
        handle.remove()
    return torch.stack(acc).sum(0).float()


def codes_scales(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    g = q.reshape(-1, GROUP)
    return torch.sign(q).to(torch.int8), g.abs().amax(1).reshape(q.shape[0], -1)


def build(arm: str) -> dict:
    torch.backends.cuda.matmul.allow_tf32 = False
    basis, stem = ARMS[arm]
    protocol = json.loads((OUT / "protocol.json").read_text())
    ids_path = OUT / protocol["calibration"]["ids_file"]
    assert sha(ids_path) == protocol["calibration"]["ids_sha256"]
    ids = torch.from_numpy(np.load(ids_path).astype(np.int64)).to(DEV)
    source = WORK / ("base_tied-f32.gguf" if basis == "base" else "folded-f32.gguf")
    template, target = WORK / f"{basis}-pq2.gguf", WORK / f"{stem}.gguf"
    _, _, src_rows, src_header = inspect(source)
    _, _, tpl_rows, tpl_header = inspect(template)
    src = {r["tensor"]: r for r in src_rows}
    tpl = {r["tensor"]: r for r in tpl_rows}
    pq = sorted(n for n, r in tpl.items() if r["storage"] == "PQ2_0")
    assert len(pq) == 197
    rot = Hadamard(WORK / "folded/hadamard_packing.json") if basis == "folded" else None
    data_aware = arm != "ls"
    model = None
    if data_aware:
        model = AutoModelForCausalLM.from_pretrained(WORK / "base_tied", dtype=torch.float32).float().to(DEV).eval()
        assert model.lm_head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
        linears = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
        fp_nll = calibration_nll(model, ids)
    shutil.copyfile(template, target)
    written, records = set(), []
    t0 = time.time()

    def read_source(name: str) -> torch.Tensor:
        r = src[name]
        cols, rows = (int(x) for x in r["dimensions_gguf_order"].split("x"))
        with source.open("rb") as f:
            f.seek(align(src_header) + r["offset"])
            data = f.read(r["elements"] * 4)
        return torch.from_numpy(np.frombuffer(data, dtype="<f4").copy()).reshape(rows, cols)

    def quantize(name: str, h: torch.Tensor | None, out) -> None:
        w_all = read_source(name)
        if data_aware:
            w_hf = linears[hf_name(name)].weight.detach()
            if rot is None:
                assert torch.equal(w_all.to(DEV), w_hf), name
            else:
                for r0 in range(0, w_all.shape[0], ROW_CHUNK):
                    assert torch.allclose(rot.fold(w_hf[r0:r0 + ROW_CHUNK].float()),
                                          w_all[r0:r0 + ROW_CHUNK].to(DEV), rtol=0, atol=1e-5), name
                h = rot.hessian(h)
        parts, err_q, err_ls, zeros, damps = [], 0.0, 0.0, 0, []
        for r0 in range(0, w_all.shape[0], ROW_CHUNK):
            w = w_all[r0:r0 + ROW_CHUNK].to(DEV)
            q_ls = ls_quantize(w)
            if data_aware:
                q, codes, scales, damp = gptq(w, h)
                damps.append(damp)
                d, dl = w - q, w - q_ls
                err_q += ((d @ h) * d).sum().item()
                err_ls += ((dl @ h) * dl).sum().item()
            else:
                q = q_ls
                codes, scales = codes_scales(q)
            zeros += (codes == 0).sum().item()
            parts.append((q.cpu(), codes.cpu(), scales.cpu()))
            del w, q_ls
        q = torch.cat([p[0] for p in parts])
        payload = pack(torch.cat([p[1] for p in parts]), torch.cat([p[2] for p in parts]))
        r = tpl[name]
        assert len(payload) == r["bytes"], name
        out.seek(align(tpl_header) + r["offset"])
        out.write(payload)
        for g in (0, len(payload) // 34 // 2, len(payload) // 34 - 1):
            dec = python_decode(payload[34 * g:34 * g + 34])
            assert np.array_equal(dec.astype(np.float32), q.reshape(-1, GROUP)[g].numpy()), (name, g)
        record = {"tensor": name, "shape": list(w_all.shape), "zero_fraction": zeros / w_all.numel()}
        if data_aware:
            with torch.no_grad():
                weight = linears[hf_name(name)].weight
                for r0 in range(0, q.shape[0], ROW_CHUNK):
                    chunk = q[r0:r0 + ROW_CHUNK].to(DEV)
                    weight[r0:r0 + ROW_CHUNK] = rot.unfold(chunk) if rot is not None else chunk
            norm = sum(((c @ h) * c).sum().item() for c in (w_all[i:i + ROW_CHUNK].to(DEV)
                                                              for i in range(0, w_all.shape[0], ROW_CHUNK)))
            record.update({"damping": max(damps), "rel_output_error_gptq": err_q / norm,
                           "rel_output_error_ls": err_ls / norm})
        records.append(record)
        written.add(name)
        extra = f" gptq {record['rel_output_error_gptq']:.4f} ls {record['rel_output_error_ls']:.4f}" if data_aware else ""
        print(f"{arm}: {name}{extra} zeros {record['zero_fraction']:.3f} ({time.time() - t0:.0f}s)", flush=True)

    with target.open("r+b") as out:
        if data_aware:
            for i, layer in enumerate(model.model.layers):
                names = [n for n in pq if n.startswith(f"blk.{i}.")]
                assert len(names) == 7
                hessians = collect(model, ids, {hf_name(n): linears[hf_name(n)] for n in names}, stop_at=layer)
                for n in names:
                    quantize(n, hessians[hf_name(n)], out)
            quantize("token_embd.weight", collect_head(model, ids), out)
        else:
            for n in pq:
                quantize(n, None, out)
    assert written == set(pq)
    assert target.stat().st_size == template.stat().st_size
    record = {"arm": arm, "basis": basis, "target": str(target.relative_to(ROOT)), "sha256": sha(target),
              "bytes": target.stat().st_size, "template_sha256": sha(template), "source_sha256": sha(source),
              "protocol_sha256": sha(OUT / "protocol.json"), "builder_sha256": sha(Path(__file__)),
              "seconds": time.time() - t0, "tensors": records, "torch": torch.__version__}
    if data_aware:
        record.update({"calibration_nll_fp": fp_nll, "calibration_nll_packed_torch": calibration_nll(model, ids),
                       "calibration_nll_note": "in-sample diagnostic on calibration windows; not an evaluation",
                       "gpu": torch.cuda.get_device_name(0)})
    return record


if __name__ == "__main__":
    for arm in sys.argv[1:] or list(ARMS):
        rec = build(arm)
        (OUT / f"build_{arm}.json").write_text(json.dumps(rec, indent=2) + "\n")
        print(f"{arm}: {rec['seconds']:.0f}s " + (f"calibration NLL fp {rec['calibration_nll_fp']:.4f} packed "
                                                  f"{rec['calibration_nll_packed_torch']:.4f}" if arm != "ls" else ""),
              flush=True)
        torch.cuda.empty_cache()

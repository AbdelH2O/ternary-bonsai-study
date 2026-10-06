"""Build the frozen data-aware (GPTQ-style ternary) PQ2_0 arms for the 0.8B pilot.

Implements results/gptq_pilot/protocol.json. Requires CUDA (host device); run from
any directory:  python3 make_gptq_pilot.py [base|folded ...]
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from codec_probe import python_decode
from phase0 import align

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work/qwen35_08b"
OUT = ROOT / "results/gptq_pilot"
sys.path.insert(0, str(ROOT.parent))
from inspect_gguf import inspect  # noqa: E402

ROLE = {"attn_qkv": "linear_attn.in_proj_qkv", "attn_gate": "linear_attn.in_proj_z",
        "ssm_out": "linear_attn.out_proj", "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
        "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj", "ffn_gate": "mlp.gate_proj",
        "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}
DEV = "cuda"
DAMP = 0.01
GROUP = 128
BATCH = 8


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hf_name(gguf: str) -> str:
    if gguf == "token_embd.weight":
        return "lm_head"
    m = re.fullmatch(r"blk\.(\d+)\.(\w+)\.weight", gguf)
    return f"model.layers.{m[1]}.{ROLE[m[2]]}"


class Hadamard:
    """Blockwise normalized signed Sylvester transform, R = H S / sqrt(B), on the last axis."""

    def __init__(self, manifest: Path):
        data = json.loads(manifest.read_text())
        self.block = data["transform"]["block_size"]
        self.signs = {int(w): torch.tensor(v, dtype=torch.float32, device=DEV) for w, v in data["signs"].items()}

    def _fwht(self, x: torch.Tensor) -> torch.Tensor:
        rows, width = x.shape
        y = x.reshape(rows, width // self.block, self.block).clone()
        h = 1
        while h < self.block:
            y = y.reshape(rows, width // self.block, self.block // (2 * h), 2, h)
            a, b = y[..., 0, :].clone(), y[..., 1, :].clone()
            y[..., 0, :], y[..., 1, :] = a + b, a - b
            h *= 2
        return y.reshape(rows, width) / np.sqrt(self.block)

    def fold(self, w: torch.Tensor) -> torch.Tensor:  # W R^T
        return self._fwht(w * self.signs[w.shape[1]])

    def unfold(self, w: torch.Tensor) -> torch.Tensor:  # W_rot R
        return self._fwht(w) * self.signs[w.shape[1]]

    def hessian(self, h: torch.Tensor) -> torch.Tensor:  # R H R^T
        return self.fold(self.fold(h).T.contiguous()).T.contiguous()


def ls_scale(w: torch.Tensor) -> torch.Tensor:
    """Per-row scale of the independent-group least-squares ternary rule, rounded to FP16."""
    mag = torch.sort(w.abs().double(), dim=1, descending=True).values
    prefix = torch.cumsum(mag, dim=1)
    k = torch.arange(1, w.shape[1] + 1, device=w.device, dtype=torch.float64)
    best = torch.argmax(prefix**2 / k, dim=1)
    scale = prefix.gather(1, best[:, None])[:, 0] / (best + 1)
    scale[prefix[:, -1] == 0] = 0
    return scale.float().half().float()


def ls_quantize(w: torch.Tensor, chunk: int = 16384) -> torch.Tensor:
    """Data-free baseline on the same matrix: the 'ls' rule applied group by group."""
    if w.shape[0] > chunk:
        return torch.cat([ls_quantize(w[i:i + chunk]) for i in range(0, w.shape[0], chunk)])
    rows, cols = w.shape
    g = w.reshape(rows * cols // GROUP, GROUP)
    mag, order = torch.sort(g.abs().double(), dim=1, descending=True)
    prefix = torch.cumsum(mag, dim=1)
    k = torch.argmax(prefix**2 / torch.arange(1, GROUP + 1, device=w.device, dtype=torch.float64), dim=1) + 1
    k[prefix[:, -1] == 0] = 0
    scale = torch.zeros(len(g), dtype=torch.float64, device=w.device)
    nz = k > 0
    scale[nz] = prefix[nz, k[nz] - 1] / k[nz]
    scale = scale.float().half().float()
    keep = torch.zeros_like(g, dtype=torch.bool)
    keep.scatter_(1, order, torch.arange(GROUP, device=w.device)[None, :] < k[:, None])
    return (torch.sign(g) * keep * scale[:, None]).reshape(rows, cols)


def gptq(w: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Return dequantized Q, int8 codes, FP16-exact scales (rows x groups) and the damping used."""
    w = w.clone().float()
    h = h.clone().float()
    rows, cols = w.shape
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    damp = DAMP * torch.mean(torch.diag(h)).item()
    for attempt in range(4):
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
        s = ls_scale(w1)
        scales[:, i1 // GROUP] = s
        safe = torch.where(s > 0, s, torch.ones_like(s))
        for i in range(GROUP):
            col = w1[:, i]
            c = torch.clamp(torch.round(col / safe), -1, 1) * (s > 0)
            qc = c * s
            q[:, i1 + i] = qc
            codes[:, i1 + i] = c.to(torch.int8)
            e = (col - qc) / hinv1[i, i]
            w1[:, i:] -= e[:, None] * hinv1[i, i:][None, :]
            err1[:, i] = e
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]
    return q, codes, scales, damp


def pack(codes: torch.Tensor, scales: torch.Tensor) -> bytes:
    """PQ2_0 blocks: FP16 scale then 32 bytes of 2-bit codes (0=-1, 1=0, 2=+1), row-major groups."""
    c = (codes.to(torch.int16) + 1).to(torch.uint8).cpu().numpy().reshape(-1, GROUP)
    packed = c[:, 0::4] | (c[:, 1::4] << 2) | (c[:, 2::4] << 4) | (c[:, 3::4] << 6)
    out = np.empty((len(c), 34), dtype=np.uint8)
    out[:, :2] = scales.cpu().numpy().reshape(-1).astype("<f2").view(np.uint8).reshape(-1, 2)
    out[:, 2:] = packed
    return out.tobytes()


def rel_error(w: torch.Tensor, q: torch.Tensor, h: torch.Tensor) -> float:
    d = w - q
    return ((d @ h) * d).sum().item() / max(((w @ h) * w).sum().item(), 1e-30)


class Stop(Exception):
    pass


def collect(model, ids: torch.Tensor, modules: dict[str, torch.nn.Module], stop_at) -> dict[str, torch.Tensor]:
    """Accumulate H = sum x x^T for each module's input, stopping the forward after `stop_at`."""
    acc = {name: None for name in modules}
    hooks = []
    for name, mod in modules.items():
        def hook(_, inputs, name=name):
            x = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
            part = (x.T @ x).double()
            acc[name] = part if acc[name] is None else acc[name] + part
        hooks.append(mod.register_forward_pre_hook(hook))
    if stop_at is not None:
        def halt(*_):
            raise Stop
        hooks.append(stop_at.register_forward_hook(halt))
    try:
        with torch.no_grad():
            for start in range(0, len(ids), BATCH):
                try:
                    model(input_ids=ids[start:start + BATCH], use_cache=False)
                except Stop:
                    pass
    finally:
        for h in hooks:
            h.remove()
    return {name: value.float() for name, value in acc.items()}


def calibration_nll(model, ids: torch.Tensor) -> float:
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(len(ids)):
            batch = ids[start:start + 1]
            logits = model(input_ids=batch, use_cache=False).logits.float()
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                                     batch[:, 1:].reshape(-1), reduction="sum")
            total += loss.item()
            count += batch[:, 1:].numel()
    return total / count


def build(basis: str) -> dict:
    torch.backends.cuda.matmul.allow_tf32 = False
    protocol = json.loads((OUT / "protocol.json").read_text())
    ids_path = OUT / protocol["calibration"]["ids_file"]
    assert sha(ids_path) == protocol["calibration"]["ids_sha256"]
    ids = torch.from_numpy(np.load(ids_path).astype(np.int64)).to(DEV)
    source, template = WORK / f"{basis}-f32.gguf", WORK / f"{basis}-pq2.gguf"
    target = WORK / f"{basis}-pq2-gptq.gguf"
    _, _, src_rows, src_header = inspect(source)
    _, _, tpl_rows, tpl_header = inspect(template)
    src = {r["tensor"]: r for r in src_rows}
    tpl = {r["tensor"]: r for r in tpl_rows}
    pq = sorted(n for n, r in tpl.items() if r["storage"] == "PQ2_0")
    assert len(pq) == 151
    rot = Hadamard(WORK / "folded/hadamard_packing.json") if basis == "folded" else None
    model = AutoModelForCausalLM.from_pretrained(WORK / "base", dtype=torch.float32).float().to(DEV).eval()
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

    def quantize(names: list[str], hessians: dict[str, torch.Tensor], out) -> None:
        for name in names:
            module = linears[hf_name(name)]
            w_hf = module.weight.detach().float()
            w = read_source(name, tuple(w_hf.shape))
            h = hessians[hf_name(name)]
            if rot is None:
                assert torch.equal(w, w_hf), name
            else:
                assert torch.allclose(rot.fold(w_hf), w, rtol=0, atol=1e-5), name
                h = rot.hessian(h)
            q, codes, scales, damp = gptq(w, h)
            payload = pack(codes, scales)
            r = tpl[name]
            assert len(payload) == r["bytes"]
            out.seek(align(tpl_header) + r["offset"])
            out.write(payload)
            for g in (0, len(payload) // 34 - 1):
                dec = python_decode(payload[34 * g:34 * g + 34])
                ref = q.reshape(-1, GROUP)[g].cpu().numpy()
                assert np.array_equal(dec.astype(np.float32), ref), (name, g)
            q_ls = ls_quantize(w)
            effective = rot.unfold(q) if rot is not None else q
            with torch.no_grad():
                module.weight.copy_(effective.to(module.weight.dtype))
            zeros = (codes == 0).float().mean().item()
            records.append({"tensor": name, "shape": list(w.shape), "damping": damp,
                            "rel_output_error_gptq": rel_error(w, q, h),
                            "rel_output_error_ls": rel_error(w, q_ls, h),
                            "zero_fraction": zeros, "sign_agreement_nonzero":
                                (torch.sign(q) == torch.sign(w))[codes != 0].float().mean().item()})
            written.add(name)
            print(f"{basis}: {name} gptq {records[-1]['rel_output_error_gptq']:.4f} "
                  f"ls {records[-1]['rel_output_error_ls']:.4f} ({time.time() - t0:.0f}s)", flush=True)

    with target.open("r+b") as out:
        for i, layer in enumerate(model.model.layers):
            names = [n for n in pq if n.startswith(f"blk.{i}.")]
            mods = {hf_name(n): linears[hf_name(n)] for n in names}
            hessians = collect(model, ids, mods, stop_at=layer)
            quantize(names, hessians, out)
        hessians = collect(model, ids, {"lm_head": linears["lm_head"]}, stop_at=None)
        quantize(["token_embd.weight"], hessians, out)
    assert written == set(pq)
    assert model.lm_head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
    packed_nll = calibration_nll(model, ids)
    assert target.stat().st_size == template.stat().st_size
    return {"basis": basis, "target": str(target.relative_to(ROOT)), "sha256": sha(target),
            "bytes": target.stat().st_size, "template_sha256": sha(template), "source_sha256": sha(source),
            "protocol_sha256": sha(OUT / "protocol.json"), "builder_sha256": sha(Path(__file__)),
            "calibration_nll_fp": fp_nll, "calibration_nll_packed_torch": packed_nll,
            "calibration_nll_note": "in-sample diagnostic on calibration windows; not an evaluation",
            "seconds": time.time() - t0, "tensors": records,
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}


if __name__ == "__main__":
    bases = sys.argv[1:] or ["base", "folded"]
    for basis in bases:
        record = build(basis)
        (OUT / f"build_{basis}.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"{basis}: calibration NLL fp {record['calibration_nll_fp']:.4f} "
              f"packed {record['calibration_nll_packed_torch']:.4f}; {record['seconds']:.0f}s")

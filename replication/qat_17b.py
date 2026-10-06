"""Qwen3-1.7B adapter for the 0.8B quantization-aware distillation code (qat_08b.py): measurement only.

Reuses the frozen 0.8B student/teacher/loss/optimizer unchanged and points them at the pinned
Qwen/Qwen3-1.7B (H1024 folded basis from Phase 1). Nothing here is a protocol: `profile` measures
throughput and peak memory on whatever GPU it runs on (locally or on Modal, see modal_qat17b.py) so
that a 1.7B protocol can be budgeted. The optional `qcache` variant reuses each projection's ternary
value between optimizer steps; it is checked bit-for-bit against the per-forward quantizer before use.

    python3 qat_17b.py prep                # local: profile stream, tensor names, base checksums
    python3 qat_17b.py profile             # any CUDA host with QAT17_WORK / QAT17_DATA populated
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import qat_08b as qat

ROOT = Path(__file__).resolve().parent
WORK = Path(os.environ.get("QAT17_WORK", ROOT / "work/qwen3_17b"))
DATA = Path(os.environ.get("QAT17_DATA", ROOT / "work/qat17b"))
SHARD = ROOT / "work/qat08/data/fineweb_edu_000_00000.parquet"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
ROLES = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")
LAYERS = 28
PROFILE_TOKENS = 2 * 1024 * 1024

qat.WORK = WORK
qat.projection_names = lambda: json.loads((DATA / "names.json").read_text())


def prep() -> None:
    """Tokenize a profile stream with the Qwen3 tokenizer and record names and base checksums."""
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    DATA.mkdir(parents=True, exist_ok=True)
    names = sorted(f"blk.{i}.{r}.weight" for i in range(LAYERS) for r in ROLES)
    _, _, rows, _ = qat.inspect(WORK / "folded-pq2.gguf")
    stored = sorted(r["tensor"] for r in rows if r["storage"] == "PQ2_0")
    assert stored == sorted(names + ["token_embd.weight"]), "names must match the Phase 1 folded template"
    (DATA / "names.json").write_text(json.dumps(names) + "\n")

    tok = AutoTokenizer.from_pretrained(WORK / "base")
    eos = tok.convert_tokens_to_ids("<|endoftext|>")
    ids: list[int] = []
    table = pq.ParquetFile(SHARD)
    for batch in table.iter_batches(columns=["text"], batch_size=256):
        for text in batch.column(0).to_pylist():
            ids.extend(tok(text)["input_ids"]); ids.append(eos)
        if len(ids) >= PROFILE_TOKENS:
            break
    arr = np.asarray(ids[:PROFILE_TOKENS], dtype=np.uint32)
    assert arr.max() < 151936
    arr.tofile(DATA / "profile.u32")

    sums = {p.name: qat.sha(p) for p in sorted((WORK / "base").iterdir()) if p.is_file()}
    record = {"purpose": "throughput/memory profile only; not a training or evaluation stream",
              "source": SHARD.name, "source_sha256": qat.sha(SHARD), "tokens": int(arr.size), "eos_id": int(eos),
              "ancestor": f"Qwen/Qwen3-1.7B@{REVISION}", "base_sha256": sums,
              "names": len(names), "hadamard_sha256": qat.sha(WORK / "folded/hadamard_packing.json")}
    (DATA / "prep_record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "base_sha256"}, indent=2))


# ---------------------------------------------------------------- optional per-step quantizer cache

def _cached_forward(self: qat.QLinear, x: torch.Tensor) -> torch.Tensor:
    if getattr(self, "_qver", None) != self.weight._version:
        with torch.no_grad():
            self._delta = qat.dequant(*qat.quantize(self.weight, self.rule)) - self.weight
        self._qver = self.weight._version
    return F.linear(self.rotate(x), (self.weight + self._delta.detach()).to(x.dtype))


_PLAIN_FORWARD = qat.QLinear.forward


def set_qcache(on: bool, model) -> None:
    qat.QLinear.forward = _cached_forward if on else _PLAIN_FORWARD
    for m in model.modules():
        if isinstance(m, qat.QLinear):
            m._qver, m._delta = None, None


def qcache_identity() -> None:
    """The cached path must give the same forward value and STE gradient as the frozen per-forward path."""
    torch.manual_seed(0)
    rot = qat.Hadamard(WORK / "folded/hadamard_packing.json")
    hmat = qat.sylvester(rot.block).to(qat.DEV)
    for rule in ("absmean", "top86"):
        w = torch.randn(256, 2048, device=qat.DEV) * 0.02
        x = torch.randn(4, 2048, device=qat.DEV, dtype=torch.bfloat16)
        outs, grads = [], []
        for fwd in (_PLAIN_FORWARD, _cached_forward):
            layer = qat.QLinear(w.clone(), rot.signs[2048], hmat, rule)
            y = fwd(layer, x)
            y.float().square().sum().backward()
            outs.append(y.detach()); grads.append(layer.weight.grad)
        assert torch.equal(outs[0], outs[1]) and torch.equal(grads[0], grads[1]), rule
    print("qcache identity: forward and gradient bit-identical for absmean and top86", flush=True)


# ---------------------------------------------------------------- profile

def profile(rule: str, configs: list[tuple[int, bool, bool]], steps: int, seqs_per_step: int) -> dict:
    """Time full optimizer steps of seqs_per_step sequences per config; lr 0 keeps weights fixed across configs."""
    torch.backends.cuda.matmul.allow_tf32 = True
    qcache_identity()
    t0 = time.time()
    teacher = qat.load_teacher()
    student, names, _ = qat.build_student(rule)
    student.train()
    params = [student.get_submodule(qat.hf_name(n)).weight for n in names]
    load_s = time.time() - t0
    trainable = sum(p.numel() for p in params)
    print(f"loaded in {load_s:.0f}s; trainable {trainable / 1e6:.1f}M in {len(params)} tensors", flush=True)
    opt = qat.AdamWBF16(params)
    stream = qat.Stream(DATA / "profile.u32")
    gpu = torch.cuda.get_device_name()
    results = []
    for micro, ckpt, qcache in configs:
        accum = seqs_per_step // micro
        name = {"micro": micro, "accum": accum, "grad_checkpointing": ckpt, "qcache": qcache}
        if ckpt:
            student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            student.gradient_checkpointing_disable()
        set_qcache(qcache, student)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        times, losses = [], []
        try:
            for step in range(steps):
                torch.cuda.synchronize(); ts = time.time()
                loss_sum = 0.0
                for k in range(accum):
                    loss = qat.distill_loss(student, teacher, stream.batch((step * accum + k) * micro, micro)) / accum
                    loss.backward(); loss_sum += loss.item()
                gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
                opt.step(0.0)
                torch.cuda.synchronize(); times.append(time.time() - ts); losses.append((loss_sum, gnorm))
                print(name, f"step {step}: kl {loss_sum:.5f} gnorm {gnorm:.4f} {times[-1]:.2f}s", flush=True)
        except torch.OutOfMemoryError:
            for p in params:
                p.grad = None
            results.append({**name, "oom": True})
            print(name, "OOM", flush=True)
            continue
        dt = float(np.median(times[1:])) if len(times) > 1 else times[0]
        results.append({**name, "step_s": dt, "tokens_per_s": seqs_per_step * qat.SEQ / dt,
                        "peak_gib": torch.cuda.max_memory_allocated() / 2**30, "kl_gnorm": losses})
        print(json.dumps(results[-1]), flush=True)
    set_qcache(False, student)
    tokens = 2000 * seqs_per_step * qat.SEQ
    best = min((r for r in results if not r.get("oom")), key=lambda r: r["step_s"])
    out = {"gpu": gpu, "rule": rule, "seq": qat.SEQ, "seqs_per_step": seqs_per_step, "trainable": trainable,
           "load_s": load_s, "configs": results, "best": best,
           "projected_hours_per_2000_steps": 2000 * best["step_s"] / 3600, "projected_tokens": tokens,
           "torch": str(torch.__version__), "script_sha256": qat.sha(Path(__file__)), "qat_08b_sha256": qat.sha(Path(qat.__file__))}
    print(json.dumps({k: v for k, v in out.items() if k != "configs"}, indent=2), flush=True)
    return out


DEFAULT_CONFIGS = [(4, True, False), (8, True, False), (8, True, True), (16, True, True),
                   (8, False, True), (16, False, True), (32, False, True)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "profile"])
    ap.add_argument("--rule", default="top86")
    ap.add_argument("--steps", type=int, default=3)
    a = ap.parse_args()
    if a.cmd == "prep":
        prep()
    else:
        result = profile(a.rule, DEFAULT_CONFIGS, a.steps, 32)
        (DATA / f"profile_{result['gpu'].replace(' ', '_')}.json").write_text(json.dumps(result, indent=2) + "\n")

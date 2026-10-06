"""Bounded quantization-aware distillation of Qwen3.5-0.8B into PQ2_0 ternary (results/qat08/protocol.json).

Student: the 150 folded decoder projections are trainable FP32 latents W_t in the H512 basis; every
forward uses the ternary projection Q(W_t) with a straight-through estimator, applied to rotated
activations (y = Q(W_t) R x), which is exactly the function the PQ2_0 runtime computes. The tied
embedding is a fixed ternary projection of the folded FP embedding; every other tensor is frozen at its
FP value (they stay byte-identical to the GGUF template). Teacher: the FP model. Loss: forward KL over
the full vocabulary.

    python3 qat_08b.py selftest
    python3 qat_08b.py profile --rule absmean --micro 2
    python3 qat_08b.py probe --rule absmean --lr 1e-4 --steps 120   # pre-freeze, training data only
    python3 qat_08b.py train --arm qat_free           # settings from the frozen protocol; resumable
    python3 qat_08b.py export --arm qat_free --ckpt final
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM

from make_gptq_pilot import GROUP, Hadamard, hf_name, inspect, pack, python_decode, sha
from phase0 import align

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work/qwen35_08b"
QAT = ROOT / "work/qat08"
OUT = ROOT / "results/qat08"
DEV = "cuda"
SEQ = 1024
LOSS_CHUNK = 128


# ---------------------------------------------------------------- quantizers (shared by training and export)

def quantize(w: torch.Tensor, rule: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return int8 codes (rows x cols) and FP16-exact float32 scales (rows x cols/128)."""
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


def ste(w: torch.Tensor, rule: str) -> torch.Tensor:
    with torch.no_grad():
        q = dequant(*quantize(w, rule))
    return w + (q - w).detach()


# ---------------------------------------------------------------- rotated ternary linear

def sylvester(n: int) -> torch.Tensor:
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


class QLinear(torch.nn.Module):
    """y = Q(W_t) (R x) with R = H S / sqrt(B) blockwise; W_t is the folded latent (W R^T at init)."""

    def __init__(self, latent: torch.Tensor, signs: torch.Tensor, hmat: torch.Tensor, rule: str):
        super().__init__()
        self.weight = torch.nn.Parameter(latent)
        self.register_buffer("signs", signs, persistent=False)
        self.register_buffer("hmat", hmat, persistent=False)
        self.rule = rule

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        b = self.hmat.shape[0]
        y = (x * self.signs.to(x.dtype)).reshape(*x.shape[:-1], x.shape[-1] // b, b)
        return (y @ self.hmat.to(x.dtype)).reshape(x.shape) / math.sqrt(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(self.rotate(x), ste(self.weight, self.rule).to(x.dtype))


def projection_names() -> list[str]:
    _, _, rows, _ = inspect(WORK / "folded-pq2.gguf")
    names = sorted(r["tensor"] for r in rows if r["storage"] == "PQ2_0")
    assert len(names) == 151
    return [n for n in names if n != "token_embd.weight"]


def fixed_embedding(model, rot: Hadamard, rule: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Folded-embedding codes/scales and the unfolded effective table used for lookup and output."""
    e = model.get_input_embeddings().weight.detach()
    codes, scales, eff = [], [], []
    for r0 in range(0, e.shape[0], 16384):
        et = rot.fold(e[r0:r0 + 16384].float().to(DEV))
        c, s = quantize(et, RULE_OF_EMBEDDING[0])
        codes.append(c.cpu()); scales.append(s.cpu())
        eff.append(rot.unfold(dequant(c, s)).to(torch.bfloat16))
    return torch.cat(codes), torch.cat(scales), torch.cat(eff)


RULE_OF_EMBEDDING = ["absmean"]


def build_student(rule: str, latents: dict[str, torch.Tensor] | None = None):
    rot = Hadamard(WORK / "folded/hadamard_packing.json")
    hmat = sylvester(rot.block).to(DEV)
    model = AutoModelForCausalLM.from_pretrained(WORK / "base", dtype=torch.float32).float().to(DEV)
    RULE_OF_EMBEDDING[0] = rule
    e_codes, e_scales, e_eff = fixed_embedding(model, rot, rule)
    for p in model.parameters():
        p.requires_grad_(False)
    names = projection_names()
    for name in names:
        path = hf_name(name)
        parent, attr = path.rsplit(".", 1)
        old = model.get_submodule(path)
        assert old.bias is None
        w = latents[name].to(DEV) if latents is not None else rot.fold(old.weight.detach().float())
        new = QLinear(w.clone(), rot.signs[old.in_features], hmat, rule)
        setattr(model.get_submodule(parent), attr, new)
    emb = model.get_input_embeddings()
    emb.weight = torch.nn.Parameter(e_eff, requires_grad=False)
    model.lm_head.weight = emb.weight
    model.config.use_cache = False
    return model, names, (e_codes, e_scales)


def load_teacher():
    t = AutoModelForCausalLM.from_pretrained(WORK / "base", dtype=torch.bfloat16).to(torch.bfloat16).to(DEV).eval()
    for p in t.parameters():
        p.requires_grad_(False)
    return t


def kl_chunk(hs: torch.Tensor, ht: torch.Tensor, es: torch.Tensor, et: torch.Tensor) -> torch.Tensor:
    ls = (hs.to(torch.bfloat16) @ es.T).float()
    lt = (ht.to(torch.bfloat16) @ et.T).float()
    logpt = F.log_softmax(lt, -1)
    return (logpt.exp() * (logpt - F.log_softmax(ls, -1))).sum()


def distill_loss(student, teacher, ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ht = teacher.model(input_ids=ids).last_hidden_state
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hs = student.model(input_ids=ids).last_hidden_state
    hs, ht = hs.reshape(-1, hs.shape[-1]), ht.reshape(-1, ht.shape[-1])
    es, et = student.lm_head.weight, teacher.lm_head.weight
    total = hs.new_zeros((), dtype=torch.float32)
    for i in range(0, hs.shape[0], LOSS_CHUNK):
        total = total + checkpoint(kl_chunk, hs[i:i + LOSS_CHUNK], ht[i:i + LOSS_CHUNK], es, et, use_reentrant=False)
    return total / hs.shape[0]


# ---------------------------------------------------------------- optimizer: AdamW with BF16 moments

class AdamWBF16:
    def __init__(self, params: list[torch.nn.Parameter], betas=(0.9, 0.95), eps=1e-8):
        self.params, self.b1, self.b2, self.eps, self.t = params, betas[0], betas[1], eps, 0
        self.m = [torch.zeros_like(p, dtype=torch.bfloat16) for p in params]
        self.v = [torch.zeros_like(p, dtype=torch.bfloat16) for p in params]

    @torch.no_grad()
    def step(self, lr: float) -> None:
        self.t += 1
        bc1, bc2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad
            m32 = m.float().mul_(self.b1).add_(g, alpha=1 - self.b1)
            v32 = v.float().mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            m.copy_(m32); v.copy_(v32)
            p.addcdiv_(m32, v32.div_(bc2).sqrt_().add_(self.eps), value=-lr / bc1)
            p.grad = None

    def state(self) -> dict:
        return {"t": self.t, "m": self.m, "v": self.v}

    def load(self, state: dict) -> None:
        self.t = state["t"]
        for a, b in zip(self.m, state["m"]):
            a.copy_(b)
        for a, b in zip(self.v, state["v"]):
            a.copy_(b)


# ---------------------------------------------------------------- data

class Stream:
    def __init__(self, path: Path):
        self.data = np.memmap(path, dtype=np.uint32, mode="r")
        self.n = len(self.data) // SEQ

    def batch(self, start_seq: int, count: int) -> torch.Tensor:
        idx = [(start_seq + k) % self.n for k in range(count)]
        arr = np.stack([self.data[i * SEQ:(i + 1) * SEQ] for i in idx]).astype(np.int64)
        return torch.from_numpy(arr).to(DEV)


@torch.no_grad()
def monitor_kl(student, teacher, stream: Stream, seqs: int = 32) -> float:
    total = 0.0
    for i in range(0, seqs, 2):
        total += distill_loss(student, teacher, stream.batch(i, 2)).item()
    return total / (seqs / 2)


# ---------------------------------------------------------------- commands

def selftest() -> None:
    torch.manual_seed(0)
    rot = Hadamard(WORK / "folded/hadamard_packing.json")
    hmat = sylvester(rot.block).to(DEV)
    for width in sorted(rot.signs):
        w = torch.randn(64, width, device=DEV)
        x = torch.randn(3, width, device=DEV)
        layer = QLinear(rot.fold(w), rot.signs[width], hmat, "absmean")
        y_rot = F.linear(layer.rotate(x), layer.weight)
        err = ((y_rot - x @ w.T).norm() / (x @ w.T).norm()).item()
        print(f"width {width}: rotated identity rel err {err:.2e}")
        assert err < 1e-5
    for rule in ("absmean", "top86"):
        w = torch.randn(16, 1024, device=DEV) * 0.02
        c, s = quantize(w, rule)
        payload = pack(c, s)
        dec = np.stack([python_decode(payload[34 * g:34 * g + 34]) for g in range(len(payload) // 34)])
        assert np.array_equal(dec.astype(np.float32), dequant(c, s).reshape(-1, GROUP).cpu().numpy()), rule
        zeros = (c.reshape(-1, GROUP) == 0).sum(1)
        print(f"{rule}: pack/decode exact; zeros per group mean {zeros.float().mean():.1f} "
              f"min {zeros.min().item()} max {zeros.max().item()}")
        if rule == "top86":
            assert torch.all(zeros == 42)
        g = w.clone().requires_grad_(True)
        ste(g, rule).sum().backward()
        assert torch.equal(g.grad, torch.ones_like(g)), "STE gradient must be identity"
    print("selftest passed")


def profile(rule: str, micro: int, steps: int) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    teacher = load_teacher()
    student, names, _ = build_student(rule)
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.train()
    params = [p for p in student.parameters() if p.requires_grad]
    print(f"trainable {sum(p.numel() for p in params) / 1e6:.1f}M in {len(params)} tensors", flush=True)
    opt = AdamWBF16(params)
    stream = Stream(QAT / "data/train.u32")
    torch.cuda.reset_peak_memory_stats()
    times = []
    for step in range(steps):
        torch.cuda.synchronize(); t0 = time.time()
        loss = distill_loss(student, teacher, stream.batch(step * micro, micro))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(1e-5)
        torch.cuda.synchronize(); times.append(time.time() - t0)
        print(f"step {step}: kl {loss.item():.4f} {times[-1]:.2f}s peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    dt = float(np.median(times[1:]))
    print(json.dumps({"rule": rule, "micro_batch": micro, "seq": SEQ, "median_step_s": dt,
                      "tokens_per_s": micro * SEQ / dt, "peak_gib": torch.cuda.max_memory_allocated() / 2**30}))


def lr_at(step: int, total: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    frac = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))


def train(arm: str) -> None:
    protocol = json.loads((OUT / "protocol.json").read_text())
    cfg = protocol["training"]
    a = protocol["arms"][arm]
    rule = a["rule"]
    run = QAT / arm
    run.mkdir(parents=True, exist_ok=True)
    resume = run / "resume.pt"
    state = torch.load(resume, map_location="cpu", weights_only=False) if resume.exists() else None
    teacher = load_teacher()
    student, names, _ = build_student(rule, state["latents"] if state else None)
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.train()
    modules = {n: student.get_submodule(hf_name(n)) for n in names}
    params = [modules[n].weight for n in names]
    opt = AdamWBF16(params, betas=tuple(cfg["betas"]))
    step = 0
    if state:
        opt.load(state["opt"])
        step = state["step"]
        del state
    train_stream, monitor = Stream(QAT / "data/train.u32"), Stream(QAT / "data/monitor.u32")
    micro, accum, total = cfg["micro_batch"], cfg["grad_accum"], cfg["steps"]
    seqs_per_step = micro * accum
    saves = {int(round(f * total)): tag for f, tag in cfg["checkpoints"]}
    log = (run / "train.jsonl").open("a")
    last_resume, t_start, tok_done = time.time(), time.time(), 0

    def save_resume() -> None:
        tmp = run / "resume.tmp"
        torch.save({"step": step, "latents": {n: modules[n].weight.detach().cpu() for n in names},
                    "opt": {k: ([t.cpu() for t in v] if isinstance(v, list) else v) for k, v in opt.state().items()}},
                   tmp)
        tmp.replace(resume)

    if step == 0:
        rec = {"step": 0, "tokens": 0, "monitor_kl": monitor_kl(student, teacher, monitor)}
        log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
    while step < total:
        lr = lr_at(step, total, cfg["peak_lr"], cfg["warmup_steps"])
        loss_sum = 0.0
        for k in range(accum):
            ids = train_stream.batch(step * seqs_per_step + k * micro, micro)
            loss = distill_loss(student, teacher, ids) / accum
            loss.backward()
            loss_sum += loss.item()
        gnorm = torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"]).item()
        opt.step(lr)
        step += 1
        tok_done += seqs_per_step * SEQ
        rec = {"step": step, "tokens": step * seqs_per_step * SEQ, "lr": lr, "train_kl": loss_sum,
               "grad_norm": gnorm, "tok_per_s": tok_done / (time.time() - t_start)}
        if step % cfg["monitor_every"] == 0 or step == total:
            rec["monitor_kl"] = monitor_kl(student, teacher, monitor)
        log.write(json.dumps(rec) + "\n"); log.flush()
        if step % 10 == 0 or "monitor_kl" in rec:
            print(rec, flush=True)
        if step in saves:
            torch.save({n: modules[n].weight.detach().cpu() for n in names}, run / f"latent_{saves[step]}.pt")
        if time.time() - last_resume > cfg["resume_every_s"] or step == total:
            save_resume(); last_resume = time.time()
    (run / "DONE").write_text(json.dumps({"steps": step, "seconds_this_session": time.time() - t_start}) + "\n")


def export(arm: str, ckpt: str) -> None:
    protocol = json.loads((OUT / "protocol.json").read_text())
    rule = protocol["arms"][arm]["rule"]
    rot = Hadamard(WORK / "folded/hadamard_packing.json")
    template = WORK / "folded-pq2.gguf"
    target = QAT / f"{arm}-{ckpt}.gguf"
    _, _, rows, header = inspect(template)
    tpl = {r["tensor"]: r for r in rows}
    latents = torch.load(QAT / arm / f"latent_{ckpt}.pt", map_location="cpu") if ckpt != "init" else None
    model = AutoModelForCausalLM.from_pretrained(WORK / "base", dtype=torch.float32).float()
    RULE_OF_EMBEDDING[0] = rule
    e_codes, e_scales, _ = fixed_embedding(model.to(DEV), rot, rule)
    shutil.copyfile(template, target)
    zeros, count = 0, 0
    with target.open("r+b") as out:
        for name in projection_names() + ["token_embd.weight"]:
            if name == "token_embd.weight":
                c, s = e_codes, e_scales
            else:
                w = latents[name].to(DEV) if latents is not None else \
                    rot.fold(model.get_submodule(hf_name(name)).weight.detach().float().to(DEV))
                c, s = quantize(w, rule)
            payload = pack(c, s)
            assert len(payload) == tpl[name]["bytes"], name
            out.seek(align(header) + tpl[name]["offset"])
            out.write(payload)
            g = len(payload) // 34 - 1
            assert np.array_equal(python_decode(payload[34 * g:34 * g + 34]).astype(np.float32),
                                  dequant(c, s).reshape(-1, GROUP)[g].cpu().numpy()), name
            zeros += (c == 0).sum().item(); count += c.numel()
    record = {"arm": arm, "checkpoint": ckpt, "rule": rule, "file": target.name, "sha256": sha(target),
              "bytes": target.stat().st_size, "zero_fraction": zeros / count,
              "latent_sha256": sha(QAT / arm / f"latent_{ckpt}.pt") if latents is not None else None,
              "protocol_sha256": sha(OUT / "protocol.json"), "exporter_sha256": sha(Path(__file__))}
    (OUT / f"export_{arm}-{ckpt}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


def probe(rule: str, lr: float, steps: int) -> None:
    """Pre-freeze learning-rate probe on training-domain data only (stream offset beyond any training run)."""
    teacher = load_teacher()
    student, names, _ = build_student(rule)
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.train()
    params = [p for p in student.parameters() if p.requires_grad]
    opt = AdamWBF16(params, betas=(0.9, 0.95))
    stream, monitor = Stream(QAT / "data/train.u32"), Stream(QAT / "data/monitor.u32")
    offset, micro, accum, warm = 300_000, 4, 8, 20
    curve = [{"step": 0, "monitor_kl": monitor_kl(student, teacher, monitor)}]
    for step in range(steps):
        for k in range(accum):
            (distill_loss(student, teacher, stream.batch(offset + (step * accum + k) * micro, micro)) / accum).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(lr * min(1.0, (step + 1) / warm))
        if (step + 1) % 30 == 0:
            curve.append({"step": step + 1, "monitor_kl": monitor_kl(student, teacher, monitor)})
            print(rule, lr, curve[-1], flush=True)
    (QAT / "probe").mkdir(exist_ok=True)
    (QAT / "probe" / f"{rule}_lr{lr:g}.json").write_text(json.dumps({"rule": rule, "lr": lr, "steps": steps,
        "tokens_per_step": micro * accum * SEQ, "stream_offset_seqs": offset, "curve": curve}, indent=2) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--rule", default="absmean")
    ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--arm")
    ap.add_argument("--ckpt", default="final")
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "profile":
        profile(a.rule, a.micro, a.steps)
    elif a.cmd == "train":
        train(a.arm)
    elif a.cmd == "probe":
        probe(a.rule, a.lr, a.steps)
    elif a.cmd == "export":
        export(a.arm, a.ckpt)

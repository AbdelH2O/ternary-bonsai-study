"""Arms M, U, MU and B of the frozen QAT08-MCU factorial (results/qat08_mcu/protocol.json).

The frozen QAT08 quantizer, student, teacher, loss, optimizer, stream and exporter are reused unchanged
(imported from qat_08b). Added per arm: the input stream file; optional training of the 169 FP
non-projection tensors with their own AdamWBF16 and peak LR (same schedule shape, joint clipping);
continuation from the chat arm's final optimizer state at a constant LR; an optimizer-boundary pause.

    python qat_08b_mcu.py selftest              # GPU: frozen QAT08 selftest + FP-tensor byte identity
    python qat_08b_mcu.py probe --lr 1e-4       # pre-design FP-tensor LR probe (QAT08 FineWeb probe offset)
    python qat_08b_mcu.py train --arm M         # requires protocol_approval.json; resumable
    python qat_08b_mcu.py export --arm M --ckpt final
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch

import qat_08b as qat
from phase0 import align
from score_17b import sha

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/qat08_mcu"
WORK = ROOT / "work/qat08_mcu"
MONITOR = ROOT / "work/qat08/data/monitor.u32"
PROBE_STREAM = ROOT / "work/qat08/data/train.u32"
TEMPLATE = ROOT / "work/qwen35_08b/folded-pq2.gguf"
FP_COUNT = 169
FP_PATTERN = re.compile(r"model\.(?:layers\.\d+\.(?:input_layernorm\.weight|post_attention_layernorm\.weight|"
                        r"self_attn\.[qk]_norm\.weight|linear_attn\.(?:A_log|dt_bias|conv1d\.weight|norm\.weight|"
                        r"in_proj_a\.weight|in_proj_b\.weight))|norm\.weight)")
FP_ROLES = {
    "attn_norm.weight": ("input_layernorm.weight", "plus1"),
    "post_attention_norm.weight": ("post_attention_layernorm.weight", "plus1"),
    "attn_q_norm.weight": ("self_attn.q_norm.weight", "plus1"),
    "attn_k_norm.weight": ("self_attn.k_norm.weight", "plus1"),
    "ssm_a": ("linear_attn.A_log", "negexp"),
    "ssm_dt.bias": ("linear_attn.dt_bias", "identity"),
    "ssm_conv1d.weight": ("linear_attn.conv1d.weight", "identity"),
    "ssm_norm.weight": ("linear_attn.norm.weight", "identity"),
    "ssm_alpha.weight": ("linear_attn.in_proj_a.weight", "identity"),
    "ssm_beta.weight": ("linear_attn.in_proj_b.weight", "identity"),
}


def fp_gguf_map() -> dict[str, tuple[str, str, str]]:
    _, _, rows, _ = qat.inspect(TEMPLATE)
    out = {}
    for r in rows:
        if r["storage"] == "PQ2_0":
            continue
        name = r["tensor"]
        if name == "output_norm.weight":
            out[name] = ("model.norm.weight", "plus1", r["storage"])
            continue
        m = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
        hf, transform = FP_ROLES[m[2]]
        out[name] = (f"model.layers.{m[1]}.{hf}", transform, r["storage"])
    assert len(out) == FP_COUNT, len(out)
    return out


def fp_payload(tensor: torch.Tensor, transform: str, dtype: str) -> bytes:
    t = tensor.detach().float().cpu().contiguous()
    t = {"identity": t, "plus1": t + 1, "negexp": -torch.exp(t)}[transform]
    if dtype == "F32":
        return t.numpy().astype("<f4").tobytes()
    assert dtype == "BF16", dtype
    return t.to(torch.bfloat16).view(torch.int16).numpy().astype("<i2").tobytes()


def template_payloads() -> dict[str, bytes]:
    _, _, rows, header = qat.inspect(TEMPLATE)
    raw, base = TEMPLATE.read_bytes(), align(header)
    return {r["tensor"]: raw[base + r["offset"]:base + r["offset"] + r["bytes"]] for r in rows if r["storage"] != "PQ2_0"}


def fp_params(model) -> dict[str, torch.nn.Parameter]:
    found = {n: p for n, p in model.named_parameters() if FP_PATTERN.fullmatch(n)}
    assert len(found) == FP_COUNT, (len(found), FP_COUNT)
    return found


BASE_WEIGHTS = ROOT / "work/qwen35_08b/base/model.safetensors-00001-of-00001.safetensors"


def exact_fp_values() -> dict[str, torch.Tensor]:
    """Checkpoint values of the 169 FP tensors, read directly from safetensors.

    transformers loads the 18 GDN norms (Qwen3_5RMSNormGated) in BF16 even with dtype=float32, so the frozen
    QAT08/chat students trained on BF16-rounded copies while their exported GGUFs keep these exact F32 values.
    Arms that train FP tensors start from, and export, these exact values.
    """
    from safetensors import safe_open
    names = {hf for hf, _, _ in fp_gguf_map().values()}
    with safe_open(BASE_WEIGHTS, "pt") as f:
        return {n: f.get_tensor(n.replace("model.", "model.language_model.", 1)).float() for n in names}


_signalled = False


class Paused(Exception):
    pass


def request_pause(signum, frame) -> None:
    global _signalled
    _signalled = True


def pause_requested() -> bool:
    return _signalled or (WORK / "PAUSE_REQUESTED").exists()


def protocol() -> dict:
    return json.loads((OUT / "protocol.json").read_text())


def approved() -> None:
    a = json.loads((OUT / "protocol_approval.json").read_text())
    assert a["approved_by_user"] is True, "protocol not approved by the user"
    assert a["protocol_sha256"] == sha(OUT / "protocol.json") and a["design_sha256"] == sha(OUT / "design.json")


def _cpu(state: dict) -> dict:
    return {k: ([t.cpu() for t in v] if isinstance(v, list) else v) for k, v in state.items()}


def _lr(step: int, spec: dict, cfg: dict, peak: float) -> float:
    if spec["schedule"] == "constant":
        return spec["lr"]
    return qat.lr_at(step, spec["steps"], peak, cfg["warmup_steps"])


def train(arm: str) -> None:
    p = protocol()
    spec, cfg = p["arms"][arm], p["training"]
    run = WORK / arm
    run.mkdir(parents=True, exist_ok=True)
    resume = run / "resume.pt"
    state = torch.load(resume, map_location="cpu", weights_only=False) if resume.exists() else None
    init = None
    if state is None and spec["init"] != "folded":
        init = torch.load(ROOT / spec["init"], map_location="cpu", weights_only=False)
        assert init["step"] == spec["init_step"], "continuation must start from the completed chat run"
    teacher = qat.load_teacher()
    source = state or init
    student, names, _ = qat.build_student(spec["rule"], source["latents"] if source else None)
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.train()
    modules = {n: student.get_submodule(qat.hf_name(n)) for n in names}
    proj = [modules[n].weight for n in names]
    fps = fp_params(student) if spec["unfreeze"] else {}
    start = state["fp"] if state else (exact_fp_values() if fps and spec.get("fp_init") == "checkpoint_f32" else {})
    for n, prm in fps.items():
        prm.requires_grad_(True)
        if n in start:
            prm.data.copy_(start[n].to(prm.device))
    fp_list = [fps[n] for n in sorted(fps)]
    opt = qat.AdamWBF16(proj, betas=tuple(cfg["betas"]))
    opt_fp = qat.AdamWBF16(fp_list, betas=tuple(cfg["betas"]))
    step = 0
    if state:
        opt.load(state["opt"])
        opt_fp.load(state["opt_fp"])
        step = state["step"]
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
    elif init:
        opt.load(init["opt"])
    del state, init, source
    stream, monitor = qat.Stream(ROOT / spec["stream"]), qat.Stream(MONITOR)
    micro, accum, total = cfg["micro_batch"], cfg["grad_accum"], spec["steps"]
    seqs = micro * accum
    saves = {int(round(f * total)): tag for f, tag in cfg["checkpoints"]}
    log = (run / "train.jsonl").open("a")
    last, t0, done_tokens = time.time(), time.time(), 0

    def save_resume() -> None:
        tmp = run / "resume.tmp"
        torch.save({"step": step, "latents": {n: modules[n].weight.detach().cpu() for n in names},
                    "fp": {n: q.detach().cpu() for n, q in fps.items()},
                    "opt": _cpu(opt.state()), "opt_fp": _cpu(opt_fp.state()),
                    "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}},
                   tmp)
        with tmp.open("rb") as f:
            os.fsync(f.fileno())
        tmp.replace(resume)

    if step == 0:
        rec = {"step": 0, "tokens": 0, "monitor_kl": qat.monitor_kl(student, teacher, monitor)}
        log.write(json.dumps(rec) + "\n"); log.flush()
    while step < total:
        if pause_requested():
            save_resume()
            log.write(json.dumps({"event": "paused", "step": step}) + "\n"); log.flush()
            raise Paused(f"training checkpoint saved at step {step}")
        lr = _lr(step, spec, cfg, cfg["peak_lr"])
        lr_fp = _lr(step, spec, cfg, cfg["fp_peak_lr"]) if fps else 0.0
        loss_sum = 0.0
        for k in range(accum):
            ids = stream.batch(step * seqs + k * micro, micro)
            loss = qat.distill_loss(student, teacher, ids) / accum
            loss.backward()
            loss_sum += loss.item()
        gnorm = torch.nn.utils.clip_grad_norm_(proj + fp_list, cfg["grad_clip"]).item()
        opt.step(lr)
        opt_fp.step(lr_fp)
        step += 1
        done_tokens += seqs * qat.SEQ
        rec = {"step": step, "tokens": step * seqs * qat.SEQ, "lr": lr, "lr_fp": lr_fp, "train_kl": loss_sum,
               "grad_norm": gnorm, "tok_per_s": done_tokens / (time.time() - t0)}
        if step % cfg["monitor_every"] == 0 or step == total:
            rec["monitor_kl"] = qat.monitor_kl(student, teacher, monitor)
        log.write(json.dumps(rec) + "\n"); log.flush()
        if step in saves:
            torch.save({n: modules[n].weight.detach().cpu() for n in names}, run / f"latent_{saves[step]}.pt")
            if fps:
                torch.save({n: q.detach().cpu() for n, q in fps.items()}, run / f"fp_{saves[step]}.pt")
        if time.time() - last > cfg["resume_every_s"] or step == total:
            save_resume()
            last = time.time()
    (run / "DONE").write_text(json.dumps({"steps": step, "seconds_this_session": time.time() - t0}) + "\n")


def probe(lr: float, steps: int = 120) -> None:
    """FP-tensor LR probe before the design freeze: QAT08 probe offset, projections at 1e-4, training-domain data."""
    target = WORK / "probe" / f"fp_lr{lr:g}.json"
    if target.exists():
        return
    teacher = qat.load_teacher()
    student, names, _ = qat.build_student("top86")
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.train()
    proj = [student.get_submodule(qat.hf_name(n)).weight for n in names]
    fps = fp_params(student)
    for q in fps.values():
        q.requires_grad_(True)
    fp_list = [fps[n] for n in sorted(fps)]
    opt, opt_fp = qat.AdamWBF16(proj, betas=(0.9, 0.95)), qat.AdamWBF16(fp_list, betas=(0.9, 0.95))
    stream, monitor = qat.Stream(PROBE_STREAM), qat.Stream(MONITOR)
    offset, micro, accum, warm = 300_000, 4, 8, 20
    curve = [{"step": 0, "monitor_kl": qat.monitor_kl(student, teacher, monitor)}]
    for step in range(steps):
        if pause_requested():
            raise Paused("probe interrupted; it restarts from step 0")
        for k in range(accum):
            ids = stream.batch(offset + (step * accum + k) * micro, micro)
            (qat.distill_loss(student, teacher, ids) / accum).backward()
        torch.nn.utils.clip_grad_norm_(proj + fp_list, 1.0)
        scale = min(1.0, (step + 1) / warm)
        opt.step(1e-4 * scale)
        opt_fp.step(lr * scale)
        if (step + 1) % 30 == 0:
            curve.append({"step": step + 1, "monitor_kl": qat.monitor_kl(student, teacher, monitor)})
            print("fp probe", lr, curve[-1], flush=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps({"fp_lr": lr, "projection_lr": 1e-4, "steps": steps, "stream_offset_seqs": offset,
                               "tokens_per_step": micro * accum * qat.SEQ, "curve": curve}, indent=2) + "\n")
    tmp.replace(target)


def export(arm: str, ckpt: str) -> None:
    p = protocol()
    spec = p["arms"][arm]
    assert not (spec["init"] != "folded" and ckpt == "init"), "a continuation has no separate init export"
    target = WORK / f"{arm}-{ckpt}.gguf"
    record_path = OUT / f"export_{arm}-{ckpt}.json"
    if target.exists() and record_path.exists():
        record = json.loads(record_path.read_text())
        if record.get("sha256") == sha(target) and "mcu_exporter_sha256" in record:
            return
    qat.OUT, qat.QAT = OUT, WORK
    qat.export(arm, ckpt)  # frozen exporter: ternary projections + fixed embedding into a template copy
    record = json.loads(record_path.read_text())
    if spec["unfreeze"]:
        _, _, rows, header = qat.inspect(TEMPLATE)
        tpl = {r["tensor"]: r for r in rows}
        if ckpt == "init":
            values = exact_fp_values()
        else:
            values = torch.load(WORK / arm / f"fp_{ckpt}.pt", map_location="cpu")
        with target.open("r+b") as out:
            for gname, (hf, transform, dtype) in fp_gguf_map().items():
                payload = fp_payload(values[hf], transform, dtype)
                assert len(payload) == tpl[gname]["bytes"], gname
                out.seek(align(header) + tpl[gname]["offset"])
                out.write(payload)
        record.update({"sha256": sha(target), "fp_tensors": "trained" if ckpt != "init" else "template",
                       "fp_sha256": sha(WORK / arm / f"fp_{ckpt}.pt") if ckpt != "init" else None})
    record["mcu_exporter_sha256"] = sha(Path(__file__))
    tmp = record_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    tmp.replace(record_path)
    print(json.dumps(record), flush=True)


def selftest() -> None:
    qat.selftest()
    values, template = exact_fp_values(), template_payloads()
    for gname, (hf, transform, dtype) in fp_gguf_map().items():
        assert fp_payload(values[hf], transform, dtype) == template[gname], gname
    print("FP-tensor byte identity at init: 169/169; selftest passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "probe", "train", "export"])
    ap.add_argument("--arm")
    ap.add_argument("--ckpt", default="final")
    ap.add_argument("--lr", type=float)
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "probe":
        probe(a.lr)
    elif a.cmd == "train":
        approved()
        train(a.arm)
    else:
        export(a.arm, a.ckpt)

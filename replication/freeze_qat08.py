"""Freeze the bounded 0.8B quantization-aware distillation test before any training run.

Reuses the 1.7B case builders (books, MMLU-Redux 2.0, GSM8K, retrieval) with the Qwen3.5 tokenizer and 15
fresh Gutenberg books, fixes the training recipe (learning rate from the pre-freeze probe on training-domain
data), the arms, checkpoints, and decision rules, and records 13-gram overlap of every evaluation case with
the exact training tokens.

    .venv/bin/python freeze_qat08.py      # after the probe; refuses to overwrite
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

import freeze_17b as fz

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/qat08"
QAT = ROOT / "work/qat08"
GUTENBERG = {  # none used by any earlier experiment in this repository
    "northanger_abbey": 121, "silas_marner": 550, "invisible_man": 5230,
    "oliver_twist": 730, "vanity_fair": 599, "tess": 110, "far_from_the_madding_crowd": 107,
    "kidnapped": 421, "prisoner_of_zenda": 95, "ivanhoe": 82, "black_beauty": 271, "mansfield_park": 141,
    "moonstone": 155, "lord_jim": 5658, "ethan_frome": 4517,
}
VALIDATION = ("northanger_abbey", "silas_marner", "invisible_man")
PROBE_LRS = (2e-5, 1e-4, 5e-4)
STEPS, MICRO, ACCUM = 2000, 4, 8
NGRAM = 13


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ngram_hashes(ids: np.ndarray) -> np.ndarray:
    ids = ids.astype(np.uint64)
    h = np.zeros(len(ids) - NGRAM + 1, dtype=np.uint64)
    for k in range(NGRAM):
        h = h * np.uint64(1_000_003) + ids[k:len(ids) - NGRAM + 1 + k]
    return h


def contamination(case_files: dict[str, Path], train_tokens: int) -> dict:
    train = np.memmap(QAT / "data/train.u32", dtype=np.uint32, mode="r")[:train_tokens]
    seen = np.unique(ngram_hashes(np.asarray(train)))
    out = {}
    for key, path in case_files.items():
        flagged, total = 0, 0
        lines = path.read_text().splitlines()
        for line in lines:
            ids = json.loads(line)["prompt_ids"] if path.suffix == ".jsonl" else list(map(int, line.split("\t")[2].split()))
            if len(ids) < NGRAM:
                continue
            h = ngram_hashes(np.asarray(ids))
            pos = np.minimum(np.searchsorted(seen, h), len(seen) - 1)
            hit = (seen[pos] == h).mean()
            flagged += hit > 0.5
            total += 1
        out[key] = {"cases": total, "cases_with_majority_13gram_overlap": int(flagged)}
    return out


def probe_choice() -> dict:
    runs = []
    for lr in PROBE_LRS:
        rec = json.loads((QAT / "probe" / f"absmean_lr{lr:g}.json").read_text())
        runs.append({"lr": lr, "final_monitor_kl": rec["curve"][-1]["monitor_kl"], "curve": rec["curve"]})
    best = min(runs, key=lambda r: r["final_monitor_kl"])
    return {"rule": "lowest training-domain monitor KL after 120 constant-LR steps (20-step warmup), absmean "
                    "student, stream offset 300,000 sequences (beyond any training run)",
            "runs": runs, "chosen_peak_lr": best["lr"]}


def main() -> None:
    if (OUT / "protocol.json").exists():
        raise SystemExit("protocol already frozen; write a versioned amendment instead")
    OUT.mkdir(parents=True, exist_ok=True)
    fz.OUT, fz.BOOKS, fz.CASES = OUT, OUT / "books", OUT / "cases"
    fz.GUTENBERG, fz.VALIDATION = GUTENBERG, VALIDATION
    fz.CASES.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(ROOT / "work/qwen35_08b/base")
    book_entries, book_files = fz.books(tok)
    mmlu, gsm8k, retrieval = fz.mmlu(tok), fz.gsm8k(tok), fz.retrieval(tok)
    probe = probe_choice()
    data = json.loads((QAT / "data/data_record.json").read_text())
    train_tokens = STEPS * MICRO * ACCUM * fz.SLICE_TOKENS
    cases = {"validation_books": OUT / book_files["validation"]["file"], "heldout_books": OUT / book_files["heldout"]["file"],
             "mmlu": OUT / mmlu["file"], "gsm8k": OUT / gsm8k["file"], "retrieval": OUT / retrieval["file"]}
    protocol = {
        "date": "2026-10-02",
        "role": ("bounded quantization-aware distillation test on Qwen3.5-0.8B (hybrid GDN); does training close the "
                 "one-shot gap, and does Bonsai 2's 42-zero budget still cost after training? No Bonsai 2 recipe claim"),
        "source": "Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17",
        "amends": {"gptq_v2_results_sha256": digest((ROOT / "results/gptq_v2/results.json").read_bytes()),
                   "phase1_17b_results_sha256": digest((ROOT / "results/q17b/results.json").read_bytes())},
        "basis": "folded H512 (work/qwen35_08b/folded/hadamard_packing.json), as selected in the 0.8B GPTQ pilot",
        "student": ("150 folded decoder projections are FP32 latents W_t (init: W R^T of the FP model); forward uses "
                    "Q(W_t) with a straight-through estimator on rotated activations, y = Q(W_t)(R x); the tied embedding "
                    "is a fixed (untrained) ternary projection of the folded FP embedding under the arm's rule; all other "
                    "tensors (norms, gates, conv, A_log, dt_bias) are frozen at FP and stay byte-identical to the template"),
        "teacher": "the FP model in BF16",
        "loss": "forward KL(teacher || student) over the full vocabulary, temperature 1, every position",
        "rules": {
            "absmean": ("per 128-group scale s = mean |w| rounded to FP16; code = clip(round_half_even(w/s), -1, 1); "
                        "free zero count (BitNet b1.58 absmean; matches Prism's 1.7B embedding in 99.83% of entries)"),
            "top86": ("per group keep the 86 largest |w| (stable sort, lower index wins ties); s = mean |kept| rounded to "
                      "FP16; code = sign (w >= 0 -> +1) inside, 0 outside: exactly 42 zeros per group (Bonsai 2's budget)"),
        },
        "data": {**data, "order": "contiguous 1024-token sequences from the start of train.u32; identical for both arms"},
        "training": {"steps": STEPS, "micro_batch": MICRO, "grad_accum": ACCUM, "seq": fz.SLICE_TOKENS,
                     "tokens_per_step": MICRO * ACCUM * fz.SLICE_TOKENS, "train_tokens": train_tokens,
                     "optimizer": "AdamW with BF16 moment buffers, no weight decay", "betas": [0.9, 0.95],
                     "peak_lr": probe["chosen_peak_lr"], "warmup_steps": 50,
                     "schedule": "linear warmup, cosine decay to 10% of peak", "grad_clip": 1.0,
                     "precision": "BF16 autocast, FP32 latents, gradient checkpointing per decoder layer",
                     "checkpoints": [[0.125, "t8m"], [0.25, "t16m"], [0.5, "t33m"], [1.0, "final"]],
                     "monitor_every": 100, "resume_every_s": 1200,
                     "kernels": "flash-linear-attention 0.5.2 (Triton) for GDN; causal-conv1d absent (torch path)",
                     "profile": "RTX 5070 12 GB: ~3,000 tokens/s, 7.9 GiB peak at micro-batch 4 (pre-freeze)",
                     "gpu_hour_ceiling": "about 6.5 h per arm (2000 steps); no extension without a new amendment"},
        "lr_probe": probe,
        "arms": {
            "fp": {"file": "work/qwen35_08b/base-f32.gguf"},
            "gptq": {"file": "work/qwen35_08b/folded-pq2-gptq.gguf", "note": "best one-shot arm (v1, folded)"},
            "absmean_init": {"rule": "absmean", "checkpoint": "init", "note": "the free student before training"},
            "top86_init": {"rule": "top86", "checkpoint": "init", "note": "the 42-zero student before training"},
            "qat_free": {"rule": "absmean", "checkpoint": "final"},
            "qat_42": {"rule": "top86", "checkpoint": "final"},
        },
        "splits": {"validation_books": list(VALIDATION), "heldout_books": [b for b in GUTENBERG if b not in VALIDATION]},
        "book_entries": book_entries, "book_cases": book_files,
        "book_scoring": {"slice": "8192 bytes after the next newline at 25% and 65% of the stripped body",
                         "tokens": "first 1024 Qwen3.5 token IDs, no special tokens", "targets": "positions 64-1023",
                         "primary": "mean NLL over targets 64-1023; book = mean of its two slices"},
        "mmlu": mmlu, "gsm8k": gsm8k, "retrieval": retrieval,
        "scoring_backend": "pinned runtime on CUDA, score_17b scorer, softmax over token IDs 0..248076 (tokenizer size)",
        "heldout_arms": {"books_mmlu_retrieval": ["fp", "gptq", "absmean_init", "top86_init", "qat_free", "qat_42"],
                         "gsm8k": ["fp", "gptq", "qat_free", "qat_42"]},
        "validation_use": "descriptive token curve: every checkpoint of both arms on the validation books; no selection",
        "contamination_13gram": contamination(cases, train_tokens),
        "decision": {
            "primary": "held-out book NLL: qat_free minus gptq (book-level 95% t interval, 12 books)",
            "training_helps": "difference <= -0.25 nat/token and upper bound < 0",
            "gap_closure": "C = (gptq - qat_free) / (gptq - fp) on held-out book NLL",
            "budget_after_training": ("qat_42 minus qat_free: within +/-0.05 = the 42-zero budget costs little after "
                                      "training; > +0.05 = costs; < -0.05 = helps"),
            "capability": ("MMLU accuracy lower 95% bound above 25% (chance) and GSM8K accuracy above 0 for a qat arm "
                           "= capability retained at least partially"),
            "recommend_1p7b_rental": ("if training_helps and (C >= 0.5 or the validation NLL of qat_free still falls by "
                                      ">= 0.02 between the t33m and final checkpoints)"),
            "claim_limit": ("one 0.8B hybrid model, one recipe and seed, 65.5M training tokens, frozen non-projection "
                            "tensors, logit-scored MMLU and one GSM8K prompt; says nothing about PrismML's recipe"),
        },
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(f"froze {digest((OUT / 'protocol.json').read_bytes())}; lr {probe['chosen_peak_lr']}; "
          f"contamination {protocol['contamination_13gram']}")


if __name__ == "__main__":
    main()

"""Freeze the cheap-refinement amendment of the 0.8B GPTQ arm before building or scoring.

Three new folded arms form a 2x2 with the existing folded GPTQ arm:
zero budget {free, 42 per group} x reconstruction {v1, v2 = 4x calibration + within-layer sequencing}.
Selection and held-out testing use freshly downloaded books that no earlier arm has seen.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/gptq_v2"
BOOKS = OUT / "books"
WORK = ROOT / "work/qwen35_08b"
DOCS = ROOT.parent / "fresh_state/documents"
GUTENBERG = {"great_expectations": 1400, "jane_eyre": 1260, "tom_sawyer": 74, "moby_dick": 2701,
             "tale_of_two_cities": 98, "treasure_island": 120, "dorian_gray": 174,
             "time_machine": 35, "heart_of_darkness": 219}
VALIDATION = ("great_expectations", "jane_eyre", "tom_sawyer")
HELDOUT = ("moby_dick", "tale_of_two_cities", "treasure_island", "dorian_gray", "time_machine",
           "heart_of_darkness")
CALIBRATION_BOOKS = ("emma.txt", "little_women.txt", "middlemarch.txt")
CAL_WINDOWS = (86, 85, 85)
WINDOW = 512


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strip_gutenberg(raw: bytes) -> bytes:
    text = raw.decode("utf-8-sig").replace("\r\n", "\n")
    lines = text.split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("*** START OF"))
    end = next(i for i, l in enumerate(lines) if l.startswith("*** END OF"))
    return "\n".join(lines[start + 1:end]).strip().encode("utf-8") + b"\n"


def slices(name: str, split: str) -> list[dict]:
    raw = (BOOKS / f"{name}.raw.txt").read_bytes()
    body = strip_gutenberg(raw)
    (BOOKS / f"{name}.txt").write_bytes(body)
    entries = []
    for fraction in (0.25, 0.65):
        start = body.find(b"\n", int(len(body) * fraction)) + 1
        assert start > 0
        text = body[start:start + 4096].decode("utf-8", errors="replace")
        stem = f"{name}-{int(100 * fraction)}"
        path = OUT / f"{stem}.txt"
        path.write_text(text, encoding="utf-8")
        entries.append({"id": stem, "split": split, "book": f"{name}.txt",
                        "gutenberg_id": GUTENBERG[name],
                        "raw_url": f"https://www.gutenberg.org/cache/epub/{GUTENBERG[name]}/pg{GUTENBERG[name]}.txt",
                        "raw_sha256": digest(raw), "body_sha256": digest(body), "byte_offset": start,
                        "text_sha256": digest(path.read_bytes()), "text_file": path.name})
    return entries


def calibration(tokenizer) -> tuple[np.ndarray, list[dict]]:
    windows, records = [], []
    for name, count in zip(CALIBRATION_BOOKS, CAL_WINDOWS):
        source = (DOCS / name).read_bytes()
        ids = tokenizer(source.decode("utf-8"), add_special_tokens=False)["input_ids"]
        lo, hi = int(0.05 * len(ids)), int(0.95 * len(ids)) - WINDOW
        starts = np.linspace(lo, hi, count).astype(np.int64)
        assert np.all(np.diff(starts) >= WINDOW), "windows must not overlap"
        windows.extend(ids[s:s + WINDOW] for s in starts)
        records.append({"book": name, "source_sha256": digest(source), "windows": count,
                        "starts": starts.tolist()})
    return np.asarray(windows, dtype=np.int32), records


def main() -> None:
    if (OUT / "protocol.json").exists():
        raise SystemExit("protocol already frozen; write a versioned amendment instead")
    tokenizer = AutoTokenizer.from_pretrained(WORK / "base")
    ids, cal = calibration(tokenizer)
    np.save(OUT / "calibration_ids_v2.npy", ids)
    entries = [e for n in VALIDATION for e in slices(n, "validation")] + \
              [e for n in HELDOUT for e in slices(n, "heldout")]
    prior = ROOT / "results/gptq_pilot"
    protocol = {
        "date": "2026-10-01",
        "role": "cheap refinements of the 0.8B data-aware arm; no Bonsai 2 quality or recipe claim",
        "amends": {"protocol_sha256": digest((prior / "protocol.json").read_bytes()),
                   "selection_sha256": digest((prior / "selection.json").read_bytes()),
                   "results_sha256": digest((prior / "results.json").read_bytes()),
                   "builder_v1_sha256": digest((ROOT / "make_gptq_pilot.py").read_bytes())},
        "source": "Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17",
        "basis": "folded H512 only (the basis selected in the previous amendment); unrotated arms are not rebuilt",
        "arms": {
            "folded-pq2-gptq": "existing v1 arm (free zero count, 64 windows, layer-sequential); comparator",
            "folded-pq2-gptq42": "v1 reconstruction with exactly 42 zeros per 128-weight group",
            "folded-pq2-gptqv2": "v2 reconstruction, free zero count",
            "folded-pq2-gptq42v2": "v2 reconstruction with exactly 42 zeros per group",
        },
        "zero_budget_42": ("at the start of each 128-column group, per row, keep the 86 largest |w| of the error-updated "
                           "group values (stable descending sort, lower index wins ties); scale = mean |w| of those 86, "
                           "rounded to FP16; each column then gets code sign(w) (w >= 0 -> +1) inside the support and 0 "
                           "outside, using the current error-updated value; GPTQ error feedback unchanged"),
        "free_budget": "unchanged from v1: least-squares scale at group start, code clamp(round_half_even(w/s), -1, 1)",
        "reconstruction_v1": "64 calibration windows (results/gptq_pilot/calibration_ids.npy); Hessians per decoder layer",
        "reconstruction_v2": ("256 non-overlapping 512-token windows from the same three calibration books "
                              "(calibration_ids_v2.npy); within each decoder layer, quantize in stages "
                              "[attention input projections] -> [attention output projection] -> [MLP gate, up] -> "
                              "[MLP down], recollecting Hessians after each stage with earlier stages packed"),
        "unchanged": "damping 0.01, no activation reordering, 128-column blocks, tied embedding quantized last, "
                     "same PQ2_0/BF16/F32 tensor policy and templates, float32 on CUDA",
        "calibration_v2": {"books": cal, "window_tokens": WINDOW, "windows": int(ids.shape[0]),
                           "ids_file": "calibration_ids_v2.npy",
                           "ids_sha256": digest((OUT / "calibration_ids_v2.npy").read_bytes())},
        "entries": entries,
        "scoring": "unchanged CPU scorer work/qwen35_08b/score_quality_pilot; 128 tokens with BOS, targets 64-127",
        "selection": ("lowest mean validation NLL over the three new validation books among the four folded GPTQ "
                      "arms; lexical tie break; written before any held-out score"),
        "heldout": {"arms": ["base-f32", "folded-pq2-gptq", "folded-pq2-gptq42", "folded-pq2-gptqv2",
                             "folded-pq2-gptq42v2"],
                    "unit": "book (mean of two slices); six books, none seen by any earlier arm"},
        "decision": {
            "primary": "held-out mean NLL of the selected arm minus folded-pq2-gptq, with a book-level 95% t interval",
            "refinements_help": "difference <= -0.10 nat/token and interval upper bound < 0: carry the selected arm into the 1.7B comparison",
            "no_material_gain": "otherwise: keep the v1 arm; refinements did not materially help at this scale",
            "budget_effect": ("mean over both reconstruction levels of (42-budget minus free) held-out NLL: "
                              "within +/-0.05 = the 42-zero format costs little here; > +0.05 = costs; < -0.05 = helps"),
            "training_entry": ("if the best held-out arm is still > 1.0 nat/token above FP, the plan's entry criterion "
                               "for proposing a bounded 0.8B training test is met (user decision, profiled resources)"),
            "claim_limit": ("one 0.8B model, 128-token natural-text NLL, six held-out books, deterministic methods "
                            "without seed variation; famous public-domain books may be in the model's pretraining data, "
                            "which affects all arms alike"),
        },
        "byte_ceiling": 220000000,
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(f"froze {len(entries)} slices and {ids.shape[0]} calibration windows; "
          f"protocol {digest((OUT / 'protocol.json').read_bytes())}")


if __name__ == "__main__":
    main()

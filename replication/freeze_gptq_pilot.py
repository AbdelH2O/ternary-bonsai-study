"""Freeze the data-aware (GPTQ-style) 0.8B arm before building or scoring it.

Amends the 0.8B quality pilot (results/quality_pilot/protocol.json) with two new
packed arms, a disjoint calibration set, three new sealed held-out books, and
predeclared selection and decision rules.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
PILOT = ROOT / "results/quality_pilot"
OUT = ROOT / "results/gptq_pilot"
WORK = ROOT / "work/qwen35_08b"
DOCS = ROOT.parent / "fresh_state/documents"
CALIBRATION_BOOKS = ("emma.txt", "little_women.txt", "middlemarch.txt")
NEW_HELDOUT_BOOKS = ("monte_cristo.txt", "war_of_the_worlds.txt", "wuthering_heights.txt")
WINDOWS = (22, 21, 21)
WINDOW_TOKENS = 512


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha(path: Path) -> str:
    return digest(path.read_bytes())


def calibration(tokenizer) -> tuple[np.ndarray, list[dict]]:
    windows, records = [], []
    for name, count in zip(CALIBRATION_BOOKS, WINDOWS):
        source = (DOCS / name).read_bytes()
        ids = tokenizer(source.decode("utf-8"), add_special_tokens=False)["input_ids"]
        lo, hi = int(0.05 * len(ids)), int(0.95 * len(ids)) - WINDOW_TOKENS
        starts = np.linspace(lo, hi, count).astype(np.int64)
        for start in starts:
            windows.append(ids[start:start + WINDOW_TOKENS])
        records.append({"book": name, "source_sha256": digest(source), "tokens": len(ids),
                        "windows": count, "starts": starts.tolist()})
    array = np.asarray(windows, dtype=np.int32)
    assert array.shape == (sum(WINDOWS), WINDOW_TOKENS)
    return array, records


def heldout_slices() -> list[dict]:
    """Same slicing rule as freeze_quality_pilot.py: 4096 bytes after the next newline at 25% and 65%."""
    entries = []
    for name in NEW_HELDOUT_BOOKS:
        source = (DOCS / name).read_bytes()
        for fraction in (0.25, 0.65):
            start = source.find(b"\n", int(len(source) * fraction)) + 1
            assert start > 0
            text = source[start:start + 4096].decode("utf-8", errors="replace")
            stem = f"{Path(name).stem}-{int(100 * fraction)}"
            target = OUT / f"{stem}.txt"
            target.write_text(text, encoding="utf-8")
            entries.append({"id": stem, "split": "heldout_new", "book": name,
                            "source_path": f"analysis/bonsai2/fresh_state/documents/{name}",
                            "source_sha256": digest(source), "byte_offset": start,
                            "text_sha256": sha(target), "text_file": target.name})
    return entries


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise SystemExit("protocol already frozen; write a versioned amendment instead")
    tokenizer = AutoTokenizer.from_pretrained(WORK / "base")
    ids, calibration_records = calibration(tokenizer)
    np.save(OUT / "calibration_ids.npy", ids)
    new_entries = heldout_slices()
    prior = json.loads((PILOT / "protocol.json").read_text())
    protocol = {
        "date": "2026-10-01",
        "role": "data-aware assignment arm for the 0.8B hybrid screen; no Bonsai 2 quality or recipe claim",
        "amends": {"protocol": "results/quality_pilot/protocol.json",
                   "protocol_sha256": sha(PILOT / "protocol.json"),
                   "selection_sha256": sha(PILOT / "selection.json"),
                   "results_sha256": sha(PILOT / "results.json")},
        "source": prior["source"],
        "scoring": prior["scoring"],
        "calibration": {
            "books": calibration_records,
            "window_tokens": WINDOW_TOKENS, "windows": int(ids.shape[0]),
            "tokenization": "HF tokenizer from work/qwen35_08b/base, no special tokens",
            "ids_file": "calibration_ids.npy", "ids_sha256": sha(OUT / "calibration_ids.npy"),
            "disjointness": "calibration books are not validation or held-out books of either protocol",
        },
        "new_heldout_entries": new_entries,
        "arms": ["base-pq2-gptq", "folded-pq2-gptq"],
        "gptq": {
            "objective": "per linear layer, minimize tr((W - Q) H (W - Q)^T) with H = sum over calibration tokens of x x^T of that layer's input",
            "order": "sequential by decoder layer: inputs to layer i are computed with layers < i already replaced by their dequantized packed weights; linears inside layer i see full-precision earlier sublayers",
            "tied_embedding": "token_embd (tied output head) quantized last using the Hessian of final-norm hidden states; earlier layers are calibrated with full-precision input embeddings",
            "folded_basis": "H_rot = R H R^T with R = H512 S / sqrt(512) and the signs in work/qwen35_08b/folded/hadamard_packing.json; W is the folded F32 GGUF tensor",
            "damping": "0.01 * mean(diag(H)) added to the diagonal; zero-diagonal columns get H_jj=1 and W[:, j]=0",
            "columns": "natural order (no activation reordering), 128-column blocks aligned with PQ2_0 groups",
            "group_scale": "at the start of each 128-column group, per row: the independent-group least-squares ternary rule (make_ls_pilot.py 'ls') applied to the error-updated group values; scale rounded to FP16",
            "code": "clamp(round_half_even(w / s), -1, 1) with the FP16 scale; s == 0 gives an all-zero group",
            "precision": "float32 weights and Hessians, torch highest matmul precision, CUDA",
            "tensor_policy": "only the same 151 PQ2_0 payloads are rewritten in a copy of {basis}-pq2.gguf; BF16 and F32 tensors stay byte-identical",
        },
        "selection": {
            "rule": "lowest mean validation NLL among all eight packed arms; prior six arms use the validation scores in results/quality_pilot/selection.json under the same scorer and slices; lexical tie break",
            "note": "validation books (dracula, pride_and_prejudice) are unchanged",
        },
        "heldout": {
            "books": ["frankenstein.txt", "sherlock_holmes.txt"] + list(NEW_HELDOUT_BOOKS),
            "arms_scored": ["base-f32", "base-pq2-ls", "the GPTQ arm with the lower validation NLL"],
            "sealing": "the three new books are unscored by any arm until selection.json is written; frankenstein and sherlock_holmes were opened once before for base-f32 and base-pq2-ls, so their GPTQ comparison is a second look",
            "unit": "book (mean of its two slices); five books",
        },
        "decision": {
            "gap_closure": "C = (NLL_ls - NLL_gptq) / (NLL_ls - NLL_fp) on the five-book held-out mean; also reported per book",
            "rule_1": "C >= 0.5: data-aware reconstruction substantially helps; include it in the matched 1.7B comparison",
            "rule_2": "C < 0.5 and NLL_gptq - NLL_fp > 1.0 nat/token: a quality shortfall remains after data-aware PTQ; this meets the plan's entry criterion to propose a bounded training experiment (user decision, profiled resources)",
            "otherwise": "report without a routing decision",
            "claim_limit": "five books, one seed-free deterministic method, one 0.8B model; descriptive, not a non-inferiority test",
        },
        "byte_ceiling": prior["byte_ceiling"],
    }
    path = OUT / "protocol.json"
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    print(f"froze {ids.shape[0]} calibration windows and {len(new_entries)} new held-out slices; "
          f"protocol {sha(path)}")


if __name__ == "__main__":
    main()

"""Freeze a small, paired natural-text screen before packed-arm scoring."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BOOKS = ROOT.parent / "scorer"
OUT = ROOT / "results/quality_pilot"
SPLIT = {
    "dracula.txt": "validation",
    "pride_and_prejudice.txt": "validation",
    "frankenstein.txt": "heldout",
    "sherlock_holmes.txt": "heldout",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, split in SPLIT.items():
        source = (BOOKS / name).read_bytes()
        for fraction in (0.25, 0.65):
            offset = int(len(source) * fraction)
            start = source.find(b"\n", offset) + 1
            if start <= 0:
                raise ValueError(name)
            text = source[start:start + 4096].decode("utf-8", errors="replace")
            stem = f"{Path(name).stem}-{int(100 * fraction)}"
            target = OUT / f"{stem}.txt"
            target.write_text(text, encoding="utf-8")
            entries.append({"id": stem, "split": split, "book": name,
                            "source_sha256": digest(source), "byte_offset": start,
                            "text_sha256": digest(target.read_bytes()),
                            "text_file": target.name})
    protocol = {
        "date": "2026-09-30",
        "role": "0.8B hybrid method screen; no Bonsai 2 quality claim",
        "source": "Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17",
        "split_rule": "two fixed offsets (25%, 65%) per book; books are independent units",
        "entries": entries,
        "scoring": {"runtime": "prism-b10735-842b188 CPU F32 logits",
                    "tokenization": "model tokenizer, BOS added, first 128 tokens of each slice",
                    "targets": "token positions 64 through 127 inclusive; 64 targets per slice",
                    "reset": "clear recurrent and attention memory between slices",
                    "batch": 128, "microbatch": 128, "threads": 8,
                    "metric": "mean teacher-forced NLL in nat/token, paired by book"},
        "arms": ["base-f32", "folded-f32", "base-pq2-absmax", "folded-pq2-absmax",
                 "base-pq2-fixed86", "folded-pq2-fixed86",
                 "base-pq2-ls", "folded-pq2-ls"],
        "amendment": "v2 before any fixed86/LS scores; v1 absmax validation already observed and retained separately",
        "fixed86": "Within each 128-weight group, choose 86 largest absolute values; stable lower-index tie break; signed code; nonnegative least-squares scale mean absolute selected, rounded to FP16",
        "ls": "Choose k=1..128 maximizing squared prefix sum of k largest absolute values divided by k; first k on tie; signed code; scale prefix/k rounded to FP16; all-zero group gets zero scale and all-zero codes",
        "tensor_policy": "same 151 PQ2_0 roles, 36 BF16 recurrent gates, 133 F32 other tensors",
        "byte_ceiling": 220_000_000,
        "screen_rule": "select lower mean validation NLL of packed arms; held-out book scores stay sealed until selection",
        "interpretation": "descriptive screen only; two validation books cannot establish non-inferiority",
        "next_gate": "expand to independent books plus retrieval/task metrics before any quality-pass claim",
    }
    path = OUT / "protocol.json"
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    print(f"froze {len(entries)} slices; protocol {digest(path.read_bytes())}")


if __name__ == "__main__":
    main()

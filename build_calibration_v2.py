"""Freeze a harder, independent short-context calibration set."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from chat_tokens import chat_template, count_tokens, render_chat


BASE = Path(__file__).resolve().parent / "prompts"
SEED = 20260929


def make_item(number: int) -> dict:
    rng = random.Random(f"cal-v2:{SEED}:{number}")
    target_bay, target_vault = number % 4, (number // 4) % 4
    target = (target_bay, target_vault)
    pairs = [target]
    pairs.extend((target_bay, vault) for vault in range(4) if vault != target_vault)
    pairs.extend((bay, target_vault) for bay in range(4) if bay != target_bay)
    other = [(bay, vault) for bay in range(4) for vault in range(4)
             if bay != target_bay and vault != target_vault]
    rng.shuffle(other)
    pairs.extend(other[:3])
    rng.shuffle(pairs)
    answer = None
    records = []
    for index, (bay, vault) in enumerate(pairs):
        current = f"K{number:02d}R{(number * 47 + bay * 83 + vault * 131 + 317) % 997:03d}"
        retired = f"Z{number:02d}R{(number * 31 + bay * 71 + vault * 103 + 193) % 997:03d}"
        if (bay, vault) == target:
            answer = current
        records.append(
            f"Record {index + 1:02d}: incoming bay B{bay}; release vault V{vault}; "
            f"current routing code {current}; retired routing code {retired}. "
            "The retired code is obsolete."
        )
    question = (f"Which current routing code applies to the parcel from incoming bay "
                f"B{target_bay} to release vault V{target_vault}?")
    prompt = ("Use both location fields to find the parcel. Several records share a bay or a vault. "
              "Do not confuse the current code with the retired code. Give only the current routing code.\n\n"
              + "\n".join(records) + "\n\nQuestion: " + question + "\nAnswer:")
    assert answer is not None and prompt.count(answer) == 1
    return {"id": f"cal-v2-{number:02d}", "answer": answer, "question": question,
            "prompt": prompt, "target_record": pairs.index(target) + 1}


def main():
    output = BASE / "calibration_v2.jsonl"
    manifest_path = BASE / "calibration_v2_manifest.json"
    if output.exists() or manifest_path.exists():
        raise FileExistsError("Calibration v2 is frozen and cannot be overwritten")
    rows = [make_item(number) for number in range(32)]
    template = chat_template()
    for row in rows:
        row["prompt_tokens"] = count_tokens(render_chat(template, row["prompt"]))
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {"seed": SEED, "items": len(rows), "prompt_tokens_min": min(r["prompt_tokens"] for r in rows),
                "prompt_tokens_max": max(r["prompt_tokens"] for r in rows),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

"""Freeze paired synthetic retrieval items before looking at any model scores."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path


OUT = Path(__file__).resolve().parent / "prompts"
SEED = 20260929
DOMAINS = [
    "archive", "observatory", "harbor", "laboratory", "museum", "expedition",
    "railway", "library", "clinic", "survey", "foundry", "aquarium",
    "greenhouse", "workshop", "depot", "gallery", "reservoir", "university",
    "orchard", "airport", "theater", "factory", "station", "research center",
]
ADJECTIVES = ["amber", "birch", "cobalt", "dawn", "elm", "frost", "granite", "hazel",
              "indigo", "juniper", "kelp", "larch", "marble", "nickel", "onyx", "pearl",
              "quartz", "reed", "silver", "topaz", "umber", "violet", "willow", "yarrow"]
NOUNS = ["ledger", "map", "crate", "sample", "permit", "folder", "shipment", "record",
         "case", "parcel", "survey", "register", "dossier", "batch", "voucher", "capsule",
         "manifest", "catalog", "ticket", "specimen", "bundle", "invoice", "report", "journal"]


def item(number: int, length: str, calibration: bool = False) -> dict:
    rng = random.Random(f"{SEED}:{number}:{length}:{calibration}")
    domain = DOMAINS[number % len(DOMAINS)]
    count = 6 if length == "short" else 152
    placement = "beginning" if number % 2 == 0 else "middle"
    target_index = 1 if placement == "beginning" else count // 2
    target_name = f"{ADJECTIVES[number % 24]} {NOUNS[(number * 7) % 24]}"
    code_prefix = "C" if calibration else "R"
    answer = f"{code_prefix}{number:02d}X{(number * 37 + 413) % 997:03d}"
    records = []
    for index in range(count):
        name = target_name if index == target_index else f"{ADJECTIVES[(index + number + 1) % 24]} {NOUNS[(index * 5 + number + 1) % 24]} {index:03d}"
        code = answer if index == target_index else f"D{number:02d}X{index:03d}"
        month = 1 + (index * 7 + number) % 12
        room = 10 + (index * 13 + number) % 89
        sentence = (
            f"Entry {index + 1:03d}: At the {domain}, the {name} was assigned reference code {code}. "
            f"It arrived in month {month:02d} and was placed in room {room}. "
            "The intake clerk checked its paper seal, recorded the custody handoff, "
            "and filed the ordinary inspection note. The reference code is the lookup value for this entry."
        )
        records.append(sentence)
    # Assert the target name and answer appear in exactly one entry.
    assert sum(target_name in record for record in records) == 1
    assert sum(answer in record for record in records) == 1
    context = "\n".join(records)
    question = f"At the {domain}, what reference code was assigned to the {target_name}?"
    prompt = (
        "Read the registry entries and answer the question. Output only the reference code; "
        "do not explain.\n\n" + context + "\n\nQuestion: " + question + "\nAnswer:"
    )
    return {"id": ("cal" if calibration else "primary") + f"-{number:02d}-{length}",
            "pair_id": ("cal" if calibration else "primary") + f"-{number:02d}",
            "length": length, "placement": placement, "target_entry": target_index + 1,
            "entry_count": count, "question": question, "answer": answer, "prompt": prompt,
            "approx_words": len(prompt.split())}


def write_jsonl(path: Path, rows: list[dict]):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main():
    OUT.mkdir(exist_ok=True)
    primary = [item(number, length) for number in range(24) for length in ("short", "long")]
    calibration = [item(number, "short", True) for number in range(8)]
    primary_path = OUT / "primary.jsonl"
    calibration_path = OUT / "calibration.jsonl"
    if primary_path.exists() or calibration_path.exists():
        raise FileExistsError("Frozen prompts already exist; refuse to overwrite them")
    write_jsonl(primary_path, primary)
    write_jsonl(calibration_path, calibration)
    corpus_path = OUT / "calibration_corpus.txt"
    corpus_path.write_text("\n\n".join(row["prompt"] + " " + row["answer"]
                                      for row in calibration) + "\n")
    summary = {"seed": SEED, "primary_pairs": 24, "calibration_items": 8,
               "placement": "12 beginning, 12 middle",
               "primary_sha256": hashlib.sha256(primary_path.read_bytes()).hexdigest(),
               "calibration_sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
               "calibration_corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
               "word_ranges": {length: [min(row["approx_words"] for row in primary if row["length"] == length),
                                       max(row["approx_words"] for row in primary if row["length"] == length)]
                               for length in ("short", "long")}}
    (OUT / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Freeze CPU-only retrieval prompts and audit answer-blind shortcuts.

Run from the repository root. This script never loads the language model or writes
the earlier primary/calibration inputs. It refuses to replace its own frozen files.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from chat_tokens import MODEL, TOKENIZER, chat_template, count_tokens, render_chat


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "prompts"
ROWS = OUT / "retrieval_v2.jsonl"
MANIFEST = OUT / "retrieval_v2_manifest.json"
SEED = 20260929
REGISTRIES = 16
RECORDS = 16
QUESTIONS = 2

ADJECTIVES = ("amber", "birch", "cobalt", "dawn", "elm", "frost", "granite", "hazel",
              "indigo", "juniper", "kelp", "larch", "marble", "nickel", "onyx", "pearl",
              "quartz", "reed", "silver", "topaz", "umber", "violet", "willow", "yarrow")
NOUNS = ("ledger", "map", "crate", "sample", "permit", "folder", "shipment", "record",
         "case", "parcel", "survey", "register", "dossier", "batch", "voucher", "capsule",
         "manifest", "catalog", "ticket", "specimen", "bundle", "invoice", "report", "journal")
PLACES = ("atrium", "courtyard", "warehouse", "workshop", "depot", "library",
          "observatory", "garden", "gallery", "annex", "station", "hall")
TASK = ("Use the registry entries to find the assigned reference code for the named item. "
        "Answer with the code alone. The background notes contain no registry entries.\n\n")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_text(value: str) -> str:
    return sha_bytes(value.encode("utf-8"))


def make_registry(number: int) -> tuple[list[dict], list[int]]:
    rng = random.Random(f"retrieval-v2:records:{SEED}:{number}")
    # Name and code pools are sampled independently before targets are chosen.
    names = [f"{a} {n} {suffix:03d}"
             for a, n, suffix in ((rng.choice(ADJECTIVES), rng.choice(NOUNS), rng.randrange(100, 1000))
                                  for _ in range(RECORDS))]
    while len(set(names)) != RECORDS:
        names = [f"{a} {n} {suffix:03d}"
                 for a, n, suffix in ((rng.choice(ADJECTIVES), rng.choice(NOUNS), rng.randrange(100, 1000))
                                      for _ in range(RECORDS))]
    codes = rng.sample(range(100000, 1000000), RECORDS)
    records = [{"name": name, "code": f"K-{code:06d}"} for name, code in zip(names, codes)]
    targets = rng.sample(range(RECORDS), QUESTIONS)
    return records, targets


def filler_paragraph(number: int, index: int) -> str:
    rng = random.Random(f"retrieval-v2:filler:{SEED}:{number}:{index}")
    place = rng.choice(PLACES)
    action = rng.choice(("checked the windows", "arranged the chairs", "washed the stone floor",
                         "repaired the lamps", "updated the visitor calendar", "inspected the doors"))
    weather = rng.choice(("clear", "cloudy", "windy", "rainy", "warm", "cool"))
    return (f"Background note {index + 1}: During an ordinary {weather} afternoon at the {place}, "
            f"the caretaker {action}. The staff then reviewed the weekly schedule, replaced worn "
            "supplies, and wrote a routine note about building maintenance. The day's activity "
            "ended without changes to the ordinary building schedule.")


def registry_lines(records: list[dict]) -> str:
    return "\n".join(
        f"Entry {index + 1:02d}: Item {record['name']} was assigned reference code {record['code']}. "
        "The clerk recorded its arrival and completed the same ordinary intake check."
        for index, record in enumerate(records)
    )


def prompt_for(records: list[dict], target: int, placement: str, filler: str) -> str:
    registry = "Registry entries:\n" + registry_lines(records)
    notes = "Background notes:\n" + filler
    context = notes + "\n\n" + registry if placement == "near" else registry + "\n\n" + notes
    question = f"What reference code was assigned to item {records[target]['name']}?"
    return TASK + context + "\n\nQuestion: " + question + "\nAnswer:"


def audit(rows: list[dict]) -> dict:
    # Score one representative per registry/question to avoid counting layouts
    # and swapped counterparts as independent observations.
    base = [r for r in rows if r["layout"] == "short-near" and r["swap"] == "base"]
    rules = {
        "first_record": lambda rec: rec[0]["code"],
        "last_record": lambda rec: rec[-1]["code"],
        "lexicographically_first_name": lambda rec: min(rec, key=lambda x: x["name"])["code"],
        "lexicographically_first_code": lambda rec: min(rec, key=lambda x: x["code"])["code"],
        "smallest_numeric_suffix": lambda rec: min(rec, key=lambda x: int(x["name"].split()[-1]))["code"],
        "unique_code_prefix": lambda rec: next((x["code"] for x in rec if sum(y["code"][0] == x["code"][0] for y in rec) == 1), None),
        "unique_name_format": lambda rec: next((x["code"] for x in rec if sum(len(y["name"].split()) == len(x["name"].split()) for y in rec) == 1), None),
    }
    results = {}
    for name, rule in rules.items():
        hits = sum(rule(row["records"]) == row["answer"] for row in base)
        results[name] = {"correct": hits, "total": len(base), "accuracy": hits / len(base)}
    positions = Counter(row["target_entry"] for row in base)
    return {"chance_per_record": 1 / RECORDS, "rules": results,
            "target_position_counts": dict(sorted(positions.items())),
            "all_name_formats_uniform": all(len({len(x["name"].split()) for x in r["records"]}) == 1 for r in base),
            "all_code_prefixes_uniform": all(len({x["code"][0] for x in r["records"]}) == 1 for r in base),
            "all_codes_unique": all(len({x["code"] for x in r["records"]}) == RECORDS for r in base),
            "two_different_answers_per_registry": all(
                len({r["answer"] for r in base if r["registry_id"] == f"registry-{i:02d}"}) == QUESTIONS
                for i in range(REGISTRIES)),
            "question_blind_shared_registry_upper_bound": 1 / QUESTIONS}


def main() -> None:
    if ROWS.exists() or MANIFEST.exists():
        raise FileExistsError("Retrieval v2 files are frozen; refusing to overwrite")
    template = chat_template()
    first_records, first_targets = make_registry(0)

    def choose_filler_count(target_tokens: int) -> int:
        def measure(count: int) -> int:
            filler = "\n".join(filler_paragraph(0, j) for j in range(count))
            prompt = prompt_for(first_records, first_targets[0], "near", filler)
            return count_tokens(render_chat(template, prompt))

        base, probe = measure(0), measure(8)
        count = max(1, round((target_tokens - base) * 8 / (probe - base)))
        for _ in range(4):
            actual = measure(count)
            if abs(actual - target_tokens) <= 45:
                return count
            count = max(1, count + round((target_tokens - actual) * 8 / (probe - base)))
        raise RuntimeError(f"Could not calibrate {target_tokens}-token filler; last count {actual}")

    short_count = choose_filler_count(1024)
    long_count = choose_filler_count(12288)
    rows = []
    for registry_number in range(REGISTRIES):
        base_records, targets = make_registry(registry_number)
        swapped_records = [dict(r) for r in base_records]
        a, b = targets
        swapped_records[a]["code"], swapped_records[b]["code"] = (
            swapped_records[b]["code"], swapped_records[a]["code"])
        assert base_records[a]["code"] == swapped_records[b]["code"]
        assert base_records[b]["code"] == swapped_records[a]["code"]
        for swap, records in (("base", base_records), ("swap", swapped_records)):
            for q_index, target in enumerate(targets):
                for layout, placement, n in (("short-near", "near", short_count),
                                             ("long-near", "near", long_count),
                                             ("long-far", "far", long_count)):
                    filler = "\n".join(filler_paragraph(registry_number, j) for j in range(n))
                    prompt = prompt_for(records, target, placement, filler)
                    rendered = render_chat(template, prompt)
                    row = {"id": f"rv2-{registry_number:02d}-q{q_index + 1}-{layout}-{swap}",
                           "registry_id": f"registry-{registry_number:02d}",
                           "question_id": f"q{q_index + 1}", "layout": layout,
                           "swap": swap, "target_entry": target + 1,
                           "entry_count": RECORDS, "records": records,
                           "question": f"What reference code was assigned to item {records[target]['name']}?",
                           "answer": records[target]["code"], "prompt": prompt,
                           "prompt_sha256": sha_text(prompt),
                           "chat_sha256": sha_text(rendered),
                           "prompt_tokens": count_tokens(rendered)}
                    assert prompt.count(row["answer"]) == 1
                    rows.append(row)

    keyed = {(r["registry_id"], r["question_id"], r["layout"], r["swap"]): r for r in rows}
    for row in rows:
        other_layout = "long-far" if row["layout"] == "long-near" else "long-near"
        if row["layout"] != "short-near":
            peer = keyed[row["registry_id"], row["question_id"], other_layout, row["swap"]]
            assert peer["records"] == row["records"]
            assert peer["answer"] == row["answer"]
            assert abs(peer["prompt_tokens"] - row["prompt_tokens"]) <= 8
        counter = keyed[row["registry_id"], row["question_id"], row["layout"],
                        "swap" if row["swap"] == "base" else "base"]
        assert row["answer"] != counter["answer"]
        assert row["question"] == counter["question"]
        assert row["target_entry"] == counter["target_entry"]
    lengths = {layout: [r["prompt_tokens"] for r in rows if r["layout"] == layout]
               for layout in ("short-near", "long-near", "long-far")}
    if not all(900 <= n <= 1150 for n in lengths["short-near"]):
        raise AssertionError(f"short lengths out of range: {min(lengths['short-near'])}, {max(lengths['short-near'])}")
    if not all(11900 <= n <= 12600 for layout in ("long-near", "long-far") for n in lengths[layout]):
        raise AssertionError("long lengths out of range")

    ROWS.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows))
    manifest = {"seed": SEED, "registries": REGISTRIES, "records_per_registry": RECORDS,
                "questions_per_registry": QUESTIONS, "items": len(rows),
                "model_path": str(MODEL), "source_model_sha256": "3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1",
                "tokenizer_path": str(TOKENIZER), "tokenizer_sha256": sha_bytes(TOKENIZER.read_bytes()),
                "chat_render_fixture_sha256": sha_text(render_chat(template, "fixture")),
                "token_policy": "GGUF Jinja chat template, user message, add_generation_prompt=True, enable_thinking=False; Prism llama-tokenize --stdin --show-count --no-bos",
                "filler_paragraphs": {"short": short_count, "long": long_count},
                "token_ranges": {k: [min(v), max(v)] for k, v in lengths.items()},
                "max_long_near_far_token_difference": max(
                    abs(r["prompt_tokens"] - keyed[r["registry_id"], r["question_id"], "long-far", r["swap"]]["prompt_tokens"])
                    for r in rows if r["layout"] == "long-near"),
                "audit": audit(rows), "rows_sha256": sha_bytes(ROWS.read_bytes())}
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

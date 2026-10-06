"""Audit the frozen retrieval-v2 file without editing its prompts or answers."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from chat_tokens import chat_template, count_tokens, render_chat
from build_retrieval_v2 import MANIFEST, ROWS, registry_lines


OUT = ROWS.with_name("retrieval_v2_audit.json")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    assert sha(ROWS.read_bytes()) == manifest["rows_sha256"]
    rows = [json.loads(line) for line in ROWS.read_text().splitlines()]
    keyed = {(r["registry_id"], r["question_id"], r["layout"], r["swap"]): r for r in rows}
    assert len(keyed) == len(rows) == 192
    template = chat_template()
    base = [r for r in rows if r["swap"] == "base"]
    short = [r for r in base if r["layout"] == "short-near"]
    fixed_index = Counter(r["target_entry"] for r in short)
    distances = {layout: [] for layout in ("short-near", "long-near", "long-far")}
    for row in base:
        rendered = render_chat(template, row["prompt"])
        assert sha(row["prompt"].encode()) == row["prompt_sha256"]
        assert sha(rendered.encode()) == row["chat_sha256"]
        target_line = registry_lines(row["records"]).splitlines()[row["target_entry"] - 1]
        end = rendered.index(target_line) + len(target_line)
        question = rendered.index("\n\nQuestion: ", end)
        # These are actual GGUF-tokenized prefix differences. A BPE token at a
        # cut boundary can move the count by one; the 11K contrast is unaffected.
        distance = count_tokens(rendered[:question]) - count_tokens(rendered[:end])
        distances[row["layout"]].append(distance)
        counter = keyed[row["registry_id"], row["question_id"], row["layout"], "swap"]
        assert row["answer"] != counter["answer"]
        assert row["question"] == counter["question"]
        assert row["prompt_tokens"] == counter["prompt_tokens"]

    answer_codes = [r["answer"] for r in short]
    registries = {r["registry_id"]: r["records"] for r in short}
    all_codes = [record["code"] for records in registries.values() for record in records]
    name_suffix_matches = sum(int(r["answer"].split("-")[1][-3:]) ==
                              int(r["records"][r["target_entry"] - 1]["name"].split()[-1])
                              for r in short)
    result = {
        "rows_sha256_verified": manifest["rows_sha256"],
        "base_cases_audited": len(base),
        "independent_question_cases": len(short),
        "question_blind_fixed_index": {
            "chance_accuracy": 1 / manifest["records_per_registry"],
            "best_fixed_index": max(fixed_index, key=fixed_index.get),
            "best_fixed_index_hits": max(fixed_index.values()),
            "best_fixed_index_accuracy": max(fixed_index.values()) / len(short),
            "all_position_hits": dict(sorted(fixed_index.items()))},
        "context_blind": {
            "distinct_answer_codes_across_independent_questions": len(set(answer_codes)),
            "all_record_codes_unique_across_registries": len(set(all_codes)) == len(all_codes),
            "answer_name_suffix_equals_code_suffix_hits": name_suffix_matches,
            "answer_name_suffix_equals_code_suffix_total": len(short)},
        "target_record_end_to_question_token_ranges": {
            layout: [min(values), max(values)] for layout, values in distances.items()},
        "counterfactual_answers_follow_swaps": True,
        "counterfactual_questions_unchanged": True,
        "counterfactual_token_counts_equal": True,
        "notes": "Distances subtract two GGUF-tokenized prefixes of the same rendered chat. Tokenization at the cut can differ by about one token. The frozen prompt file was not modified."
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if OUT.exists():
        if OUT.read_text() != encoded:
            raise ValueError("Frozen audit differs from recomputed result")
    else:
        OUT.write_text(encoded)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

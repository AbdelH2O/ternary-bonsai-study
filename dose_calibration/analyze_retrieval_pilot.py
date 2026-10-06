"""Verify baseline accuracy, question/value swaps, and near/far token matching."""

import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    frozen = json.loads((HERE / "DOSE_FREEZE.json").read_text())
    pilot = frozen["retrieval_pilot"]
    prompts = ROOT / pilot["prompt_file"]
    if sha(prompts) != pilot["prompt_file_sha256"]:
        raise ValueError("Frozen prompt file changed")
    raw = [json.loads(line) for line in (HERE / "retrieval_baseline_pilot.jsonl").read_text().splitlines()]
    by_id = {r["id"]: r for r in raw}
    if len(raw) != len(by_id) or set(by_id) != set(pilot["item_ids"]):
        raise ValueError("Pilot IDs incomplete or duplicate")
    expected = {r["id"]: r for r in (json.loads(line) for line in prompts.read_text().splitlines())}
    for item, actual in by_id.items():
        source = expected[item]
        if (actual["prompt_sha256"], actual["chat_sha256"], actual["gold"]) != (
                source["prompt_sha256"], source["chat_sha256"], source["answer"]):
            raise ValueError(f"Prompt/answer identity mismatch: {item}")
        if actual["prompt_tokens"] != source["prompt_tokens"]:
            raise ValueError(f"Server token count mismatch: {item}")
    pairs = {}
    for row in raw:
        key = (row["registry_id"], row["question_id"], row["layout"])
        pairs.setdefault(key, {})[row["swap"]] = row
    if len(pairs) != 24 or any(set(group) != {"base", "swap"} for group in pairs.values()):
        raise ValueError("Missing value-swap pair")
    swap_followed = sum(group["base"]["gold"] != group["swap"]["gold"] and
                        group["base"]["response"].strip() == group["base"]["gold"] and
                        group["swap"]["response"].strip() == group["swap"]["gold"] and
                        group["base"]["response"].strip() != group["swap"]["response"].strip()
                        for group in pairs.values())
    layout = {}
    for name in ("short-near", "long-near", "long-far"):
        rows = [r for r in raw if r["layout"] == name]
        layout[name] = {"n": len(rows), "exact_correct": sum(r["exact_correct"] for r in rows),
                        "extracted_correct": sum(r["extracted_correct"] for r in rows),
                        "prompt_tokens_range": [min(r["prompt_tokens"] for r in rows),
                                                max(r["prompt_tokens"] for r in rows)],
                        "mean_elapsed_seconds": statistics.mean(r["elapsed_seconds"] for r in rows)}
    matched_long = all(group["long-near"]["prompt_tokens"] == group["long-far"]["prompt_tokens"]
        for group in ({r["layout"]: r for r in raw if (r["registry_id"], r["question_id"], r["swap"]) == key}
                      for key in {(r["registry_id"], r["question_id"], r["swap"]) for r in raw}))
    result = {"frozen_prompt_sha256": pilot["prompt_file_sha256"],
              "model_sha256": frozen["baseline_model_sha256"],
              "items": len(raw), "errors": sum(r["error"] is not None for r in raw),
              "exact_correct": sum(r["exact_correct"] for r in raw),
              "swap_pairs": len(pairs), "swap_pairs_followed": swap_followed,
              "long_near_far_token_counts_equal": matched_long,
              "server_chat_token_counts_equal_frozen_counts": True,
              "layout": layout,
              "decision": "baseline validity passed; accuracy at ceiling, so perturbation sensitivity remains untested"}
    out = HERE / "retrieval_pilot_summary.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Audit frozen IDs, stock alignment, reset/repeat/batching, and enrich baseline rows."""

import hashlib
import json
import math
import re
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent


def rows(name):
    return [json.loads(line) for line in (HERE / name).read_text().splitlines()]


def max_delta(left, right):
    assert len(left) == len(right)
    assert all(a["case_id"] == b["case_id"] for a, b in zip(left, right))
    return max(abs(a - b) for x, y in zip(left, right) for a, b in zip(x["nll"], y["nll"]))


def mean_delta(left, right):
    return max(abs(sum(x["nll"]) / x["m"] - sum(y["nll"]) / y["m"])
               for x, y in zip(left, right))


def stock_audit():
    data = (HERE / "stock_fixture_logits.bin").read_bytes()
    assert data[:8] == b"_logits_"
    ctx, vocab, chunks = struct.unpack_from("<iii", data, 8)
    assert (ctx, chunks) == (64, 2)
    ids = list(struct.unpack_from(f"<{ctx * chunks}i", data, 20))
    all_ids = [int(v) for v in (HERE / "pride_and_prejudice.ids").read_text().splitlines()]
    assert ids == all_ids[:len(ids)]
    stride = (2 * ((vocab + 1) // 2) + 4) * 2
    offset = 20 + 4 * len(ids)
    count = chunks * (ctx // 2 - 1)
    assert len(data) == offset + count * stride
    saved_nll = []
    for row in range(count):
        scale, minimum_log_prob = struct.unpack_from("<ff", data, offset + row * stride)
        window, pos = divmod(row, ctx // 2 - 1)
        gold = ids[window * ctx + ctx // 2 + pos + 1]
        quantized_log_prob = struct.unpack_from("<H", data, offset + row * stride + 8 + gold * 2)[0]
        saved_nll.append(-(minimum_log_prob + scale * quantized_log_prob))
    custom = [v for row in rows("stock_compare.jsonl") for v in row["nll"]]
    assert len(custom) == count
    printed = re.search(r"Final estimate: PPL = ([0-9.]+)", (HERE / "stock_fixture.log").read_text())
    assert printed is not None
    printed_ppl = float(printed.group(1))
    reconstructed_ppl = math.exp(sum(saved_nll) / count)
    custom_ppl = math.exp(sum(custom) / count)
    assert abs(reconstructed_ppl - printed_ppl) < 0.0001
    assert max(abs(a - b) for a, b in zip(saved_nll, custom)) < 0.0002
    return {"ctx": ctx, "vocab": vocab, "chunks": chunks, "file_bytes": len(data),
            "ids_match": True, "printed_ppl": printed_ppl,
            "reconstructed_ppl": reconstructed_ppl, "custom_ppl": custom_ppl,
            "max_saved_vs_custom_nll": max(abs(a - b) for a, b in zip(saved_nll, custom))}


def main():
    manifest = json.loads((HERE / "frozen_manifest.json").read_text())
    cases = {case["case_id"]: case for case in manifest["cases"]}
    docs = {doc["id"]: doc for doc in manifest["documents"]}
    baseline = rows("baseline_512.jsonl")
    assert len(cases) == len(baseline) == 12
    assert {row["case_id"] for row in baseline} == set(cases)
    assert not any("failed" in line.lower() or "error:" in line.lower()
                   for line in (HERE / "baseline_512.log").read_text().splitlines()
                   if "failed to" in line.lower() or "error:" in line.lower())
    enriched = []
    for row in baseline:
        case = cases[row["case_id"]]
        doc = docs[case["document_id"]]
        assert (row["p"], row["h"], row["m"]) == (case["target_position"], case["history_length"], case["target_length"])
        assert len(row["nll"]) == row["m"] and all(math.isfinite(v) and v >= 0 for v in row["nll"])
        enriched.append({"case_id": row["case_id"], "arm": "baseline", "document_id": doc["id"],
            "source_url": doc["source_url"], "source_sha256": doc["source_sha256"],
            "tokenizer": manifest["tokenizer"], "all_ids_sha256": doc["all_ids_sha256"],
            "target_position": row["p"], "history_length": row["h"], "token_count": row["m"],
            "history_ids_sha256": case["history_ids_sha256"], "target_ids_sha256": case["target_ids_sha256"],
            "model_sha256": manifest["model_sha256"], "patch_sha256": None,
            "scorer_revision_sha256": manifest["scorer_source_sha256"],
            "gold_nll_nats": row["nll"], "sum_gold_nll_nats": sum(row["nll"]),
            "mean_gold_nll_nats": sum(row["nll"]) / row["m"], "seconds": row["seconds"],
            "backend": {"runtime_release": manifest["runtime_release"], "cuda_layers": 99,
                "n_ctx": 16384, "n_seq_max": 1, "n_batch": 512, "n_ubatch": 512,
                "flash_attention": "on", "kv_cache_k": "f16", "kv_cache_v": "f16",
                "speculative_decoding": False}, "error": None})
    for doc_id in docs:
        hashes = {cases[row["case_id"]]["target_ids_sha256"] for row in baseline
                  if cases[row["case_id"]]["document_id"] == doc_id}
        assert len(hashes) == 1
    (HERE / "baseline_results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in enriched))

    a, b = rows("fixture_run1.jsonl"), rows("fixture_run2.jsonl")
    logical = rows("fixture_batch1119.jsonl")
    physical = rows("fixture_batch256.jsonl")
    same_physical = rows("fixture_ubatch256.jsonl")
    aba = rows("fixture_reset.jsonl")
    assert max_delta(a, b) == 0
    assert max_delta(a, logical) == 0
    assert max_delta(physical, same_physical) == 0
    assert max_delta(aba[:1], aba[2:]) == 0
    assert max_delta(a, aba[:2]) == 0
    stock = stock_audit()
    summary = {"decision": "GO: scorer valid for fixed 512/512 profile; stop before perturbation calibration",
        "stock_alignment": stock,
        "fixture": {"documents": 2, "target_tokens_each": 96,
            "rerun_max_nll_delta": max_delta(a, b),
            "one_vs_split_logical_batch_max_nll_delta": max_delta(a, logical),
            "aba_reset_max_nll_delta": max_delta(aba[:1], aba[2:]),
            "fixed_ubatch_logical_boundary_max_nll_delta": max_delta(physical, same_physical),
            "changed_ubatch_max_token_nll_delta": max_delta(a, physical),
            "changed_ubatch_max_mean_nll_delta": mean_delta(a, physical)},
        "baseline": {"documents": len(docs), "conditions": len(baseline),
            "target_tokens_per_condition": 96, "total_scored_tokens": sum(row["m"] for row in baseline),
            "total_inference_seconds": sum(row["seconds"] for row in baseline),
            "errors": 0,
            "mean_nll_by_document_history": {row["case_id"]: sum(row["nll"]) / row["m"] for row in baseline}}}
    (HERE / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "baseline"}, indent=2))


if __name__ == "__main__":
    main()

"""Freeze natural-text dose-calibration spans, disjoint from scorer validation books."""

import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
MODEL = ROOT / "models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf"
MODEL_SHA = "3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1"
DOCS = {
    "tale_two_cities": (98, "d54c2b80d40a40b982cd88852c6180bb944d95acdb028af3d0e01a1750681784", "5453f30ac15e1e1bd70658543591e68fd70455e2f2f599e8b5d8820d9f28dad0"),
    "jane_eyre": (1260, "13414dee2951c3ee731d76d2ffd822016b2479c892162760c5d0eb2aa5fa7631", "ebdf40238d7daa30408cd73d795ed122f1cd1d0511867c997b5c50fec52509b5"),
    "moby_dick": (2701, "907420db6c4b68c70e2988cd2ad9c8cf79138667a01b63376d18dd17fef1a18b", "b79cb9c4c171ac13b696336333a906b1c175717ec54ed4dbfaabe0bbbd2a0d39"),
    "great_expectations": (1400, "9a637118af8e953e9764ec603d9b0a032883384d465acac2e27966a80cf1c6f8", "1f671fba21ef1d51862d4eefe682f4c0a97b3539b786000a6906e655b436e163"),
}
POSITIONS = (30000, 50000)
HISTORY = 1024
TARGET = 96


def sha(data):
    return hashlib.sha256(data).hexdigest()


def ids_sha(ids):
    return sha(struct.pack("<" + "i" * len(ids), *ids))


def main():
    if sha(MODEL.read_bytes()) != MODEL_SHA:
        raise ValueError("Baseline model hash changed")
    frozen = {"purpose": "dose calibration only; excluded from held-out scoring",
              "model_sha256": MODEL_SHA, "runtime": "prism-b10735-842b188 cuda-13.3",
              "scorer_source_sha256": sha((ROOT / "analysis/bonsai2/scorer/position_scorer.cpp").read_bytes()),
              "tokenizer": "Prism llama_tokenize(add_special=false, parse_special=false); no BOS or chat template",
              "id_hash_encoding": "SHA-256 of little-endian int32 token IDs",
              "history_length": HISTORY, "target_length": TARGET, "documents": [], "cases": []}
    lines = []
    for name, (ebook, expected_source, expected_ids_file) in DOCS.items():
        source = HERE / f"{name}.txt"
        ids_file = HERE / f"{name}.ids"
        if sha(source.read_bytes()) != expected_source or sha(ids_file.read_bytes()) != expected_ids_file:
            raise ValueError(f"Frozen source/tokenization mismatch: {name}")
        ids = [int(v) for v in ids_file.read_text().splitlines()]
        if len(ids) < max(POSITIONS) + TARGET:
            raise ValueError(f"Document too short: {name}")
        frozen["documents"].append({"id": name,
            "source_url": f"https://www.gutenberg.org/cache/epub/{ebook}/pg{ebook}.txt",
            "license": "Project Gutenberg ebook; public domain in the United States; retain source terms for redistribution",
            "source_path": str(source.relative_to(ROOT)), "source_sha256": expected_source,
            "ids_path": str(ids_file.relative_to(ROOT)), "ids_file_sha256": expected_ids_file,
            "all_ids_sha256": ids_sha(ids), "token_count": len(ids)})
        for p in POSITIONS:
            item = {"case_id": f"{name}-p{p}-h{HISTORY}", "document_id": name,
                    "target_position": p, "history_length": HISTORY, "target_length": TARGET,
                    "history_ids_sha256": ids_sha(ids[p-HISTORY:p]),
                    "target_ids_sha256": ids_sha(ids[p:p+TARGET])}
            frozen["cases"].append(item)
            lines.append(f"{item['case_id']}\t{ids_file.relative_to(ROOT)}\t{p}\t{HISTORY}\t{TARGET}\n")
    manifest = HERE / "calibration_manifest.json"
    encoded = json.dumps(frozen, indent=2) + "\n"
    if manifest.exists() and manifest.read_text() != encoded:
        raise ValueError("Existing calibration manifest differs")
    manifest.write_text(encoded)
    cases = HERE / "calibration_cases.tsv"
    if cases.exists() and cases.read_text() != "".join(lines):
        raise ValueError("Existing calibration cases differ")
    cases.write_text("".join(lines))
    print(f"Frozen {len(frozen['cases'])} calibration spans from {len(DOCS)} disjoint documents")


if __name__ == "__main__":
    main()

"""Verify pinned public-domain sources/tokenization and freeze exact nested-ID cases."""

import hashlib
import json
import struct
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
MODEL = ROOT / "models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf"
SCORER = HERE / "position_scorer"
MODEL_SHA = "3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1"
DOCS = {
    "pride_and_prejudice": (1342, "3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75", "53eb8a3291e53611b0d6e36b534f53ee2d291fd998eb7ea1ab86a36b2a3f35b9"),
    "frankenstein": (84, "7810cd483cffcf2cc8a1d8f0d5807931e69d4f48cd14149b8c76f88af82fead3", "707334c7fd798fba2821db628c7b3edd90ee5240b53c89445ec0ecaa95c13019"),
    "sherlock_holmes": (1661, "922e2a12ccb43a4c9544c260b2166c6ad2097aeb5957faeee113f173bb857cd0", "9e56505ccf6a6ccc0b4695ee5b9c47ee40f4d42710714908c3b3d954abe0eb05"),
    "dracula": (345, "96cd16eacdbfebae8fdda5591f66e0cc8ee76be18e0cd1aca02bc00615782d28", "27050e76f5e45c1c1bc63ee5863d993f7b87f555499d61d556af0ad85f005e3b"),
}
HISTORIES = (1024, 4096, 12288)
P = 30000
M = 96


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ids_digest(ids: list[int]) -> str:
    return digest(struct.pack("<" + "i" * len(ids), *ids))


def main() -> None:
    if digest(MODEL.read_bytes()) != MODEL_SHA:
        raise ValueError("model differs from frozen baseline")
    manifest = {
        "model_path": str(MODEL.relative_to(ROOT)), "model_sha256": MODEL_SHA,
        "patch_sha256": None, "arm": "baseline", "runtime_release": "prism-b10735-842b188 cuda-13.3",
        "scorer_source_sha256": digest((HERE / "position_scorer.cpp").read_bytes()),
        "tokenizer": "pinned Prism llama_tokenize; add_special=false; parse_special=false; no BOS; no chat template",
        "id_hash_encoding": "SHA-256 of little-endian signed int32 IDs",
        "position_convention": "zero-based full-document IDs; context [p-h:p]; gold [p:p+m]",
        "history_lengths": HISTORIES, "target_length": M, "documents": [], "cases": [],
    }
    lines = []
    for name, (ebook, source_hash, ids_file_hash) in DOCS.items():
        url = f"https://www.gutenberg.org/cache/epub/{ebook}/pg{ebook}.txt"
        text_path = HERE / f"{name}.txt"
        ids_path = HERE / f"{name}.ids"
        if not text_path.exists():
            text_path.write_bytes(urllib.request.urlopen(url).read())
        if digest(text_path.read_bytes()) != source_hash:
            raise ValueError(f"source changed: {name}")
        if not ids_path.exists():
            subprocess.run([str(SCORER), "tokenize", str(MODEL), str(text_path), str(ids_path)], check=True)
        if digest(ids_path.read_bytes()) != ids_file_hash:
            raise ValueError(f"tokenization changed: {name}")
        ids = [int(x) for x in ids_path.read_text().splitlines()]
        if len(ids) < P + M or P < max(HISTORIES):
            raise ValueError(f"document too short: {name}")
        manifest["documents"].append({"id": name, "source_url": url,
            "source_path": str(text_path.relative_to(ROOT)), "source_sha256": source_hash,
            "license": "Project Gutenberg ebook; public domain in the United States; retain source terms for redistribution",
            "ids_path": str(ids_path.relative_to(ROOT)), "ids_file_sha256": ids_file_hash,
            "all_ids_sha256": ids_digest(ids), "token_count": len(ids)})
        target_hash = ids_digest(ids[P:P + M])
        for h in HISTORIES:
            case_id = f"{name}-p{P}-h{h}"
            manifest["cases"].append({"case_id": case_id, "document_id": name,
                "target_position": P, "target_length": M, "history_length": h,
                "history_ids_sha256": ids_digest(ids[P - h:P]), "target_ids_sha256": target_hash,
                "scored_ids_sha256": target_hash, "first_target_from": P - 1})
            lines.append(f"{case_id}\t{ids_path.relative_to(ROOT)}\t{P}\t{h}\t{M}\n")
    data = json.dumps(manifest, indent=2) + "\n"
    dest = HERE / "frozen_manifest.json"
    if dest.exists() and dest.read_text() != data:
        raise ValueError("frozen manifest differs; refusing overwrite")
    dest.write_text(data)
    (HERE / "baseline_cases.tsv").write_text("".join(lines))
    print(f"Verified {len(DOCS)} sources and {len(lines)} position-matched cases")


if __name__ == "__main__":
    main()

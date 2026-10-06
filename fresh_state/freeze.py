"""Freeze new books, nested state-assay spans, binaries, and decisions before inference."""

import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DOCS = [
    ("emma", 158, "Emma"),
    ("little_women", 514, "Little Women"),
    ("wuthering_heights", 768, "Wuthering Heights"),
    ("war_of_the_worlds", 36, "The war of the worlds"),
    ("monte_cristo", 1184, "The Count of Monte Cristo"),
    ("middlemarch", 145, "Middlemarch"),
]
POSITIONS = (30000, 50000)
HISTORIES = (1024, 12288)
TARGET = 96


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids_sha(ids):
    return hashlib.sha256(struct.pack("<" + "i" * len(ids), *ids)).hexdigest()


def main():
    scorer = ROOT / "analysis/bonsai2/scorer"
    state = ROOT / "analysis/bonsai2/validation_gates/state_transplant"
    controlled = ROOT / "analysis/bonsai2/controlled_state"
    previous = json.loads((controlled / "CONTROLLED_STATE_FREEZE.json").read_text())
    dose = ROOT / "analysis/bonsai2/dose_calibration/DOSE_FREEZE.json"
    frozen = {
        "purpose": "fresh-document recurrent-state contrast; six document clusters, two disjoint spans each",
        "design_sha256": sha(HERE / "DESIGN.md"),
        "prior_controlled_state_freeze_sha256": sha(controlled / "CONTROLLED_STATE_FREEZE.json"),
        "dose_freeze_sha256": sha(dose),
        "runtime_release": "prism-b10735-842b188 cuda-13.3",
        "profile": {"n_ctx": 16384, "n_seq_max": 1, "n_batch": 512,
                    "n_ubatch": 512, "flash_attention": "on", "attention_kv": "f16",
                    "gpu_layers": 99},
        "tokenizer": "pinned Prism llama_tokenize(add_special=false, parse_special=false); no BOS or chat template",
        "id_hash_encoding": "SHA-256 of little-endian signed int32 IDs",
        "target_length": TARGET, "positions": list(POSITIONS), "histories": list(HISTORIES),
        "models": previous["models"], "documents": [], "cases": [],
        "program_hashes": {
            str(p.relative_to(ROOT)): sha(p) for p in (
                scorer / "position_scorer.cpp", scorer / "position_scorer",
                state / "state_gate.cpp", state / "state_gate",
                controlled / "controlled_state.cpp", controlled / "controlled_state",
                HERE / "freeze.py", HERE / "run.py", HERE / "analyze.py")},
        "linked_cuda_library_sha256": previous["linked_cuda_library_sha256"],
        "gates": {
            "identity_max_token_nll_delta": 0.0001,
            "positive_mean_absolute_token_nll_delta_min": 0.01,
            "primary_smallest_meaningful_nats_per_token": 0.005,
            "short_state_dose_equivalence_band": 0.003,
            "bootstrap_replicates": 20000, "bootstrap_seed": 20260929,
            "support": "primary mean >=+0.005, document-bootstrap 95% lower bound >0, >=5/6 document means positive; short-dose 90% CI must fit within +/-0.003 for comparable-dose specificity",
            "exclude_0p005": "document-bootstrap 95% upper bound <+0.005",
        },
    }
    scorer_manifest = json.loads((scorer / "frozen_manifest.json").read_text())
    if scorer_manifest["model_sha256"] != frozen["models"]["original"]["model_sha256"]:
        raise ValueError("Pinned scorer model differs")
    for arm, model in frozen["models"].items():
        if sha(ROOT / model["model_path"]) != model["model_sha256"]:
            raise ValueError(f"Model changed: {arm}")
        if model.get("patch_manifest_path") and sha(ROOT / model["patch_manifest_path"]) != model["patch_manifest_sha256"]:
            raise ValueError(f"Patch manifest changed: {arm}")
    for relative, digest in frozen["linked_cuda_library_sha256"].items():
        if sha(ROOT / relative) != digest:
            raise ValueError(f"Linked Prism library changed: {relative}")
    lines = []
    for name, ebook, title in DOCS:
        source = HERE / "documents" / f"{name}.txt"
        ids_path = HERE / "documents" / f"{name}.ids"
        source_bytes = source.read_bytes()
        if f"The Project Gutenberg eBook of {title}".casefold().encode() not in source_bytes[:300].lower():
            raise ValueError(f"Wrong Gutenberg title: {name}")
        ids = [int(x) for x in ids_path.read_text().splitlines()]
        if len(ids) < max(POSITIONS) + TARGET:
            raise ValueError(f"Book too short: {name}")
        doc = {"id": name, "gutenberg_ebook": ebook,
               "source_url": f"https://www.gutenberg.org/cache/epub/{ebook}/pg{ebook}.txt",
               "catalog_url": f"https://www.gutenberg.org/ebooks/{ebook}",
               "license": "Project Gutenberg; public domain in the United States; retain source terms for redistribution",
               "source_path": str(source.relative_to(ROOT)), "source_sha256": sha(source),
               "ids_path": str(ids_path.relative_to(ROOT)), "ids_file_sha256": sha(ids_path),
               "all_ids_sha256": ids_sha(ids), "token_count": len(ids)}
        frozen["documents"].append(doc)
        if POSITIONS[0] + TARGET >= POSITIONS[1] - max(HISTORIES):
            raise ValueError("12K histories and targets overlap")
        target_hashes = {}
        for p in POSITIONS:
            target_hashes[p] = ids_sha(ids[p:p + TARGET])
            for h in HISTORIES:
                case_id = f"{name}-p{p}-h{h}"
                case = {"case_id": case_id, "document_id": name,
                        "target_position": p, "history_length": h, "target_length": TARGET,
                        "history_ids_sha256": ids_sha(ids[p - h:p]),
                        "target_ids_sha256": target_hashes[p],
                        "first_target_from": p - 1}
                frozen["cases"].append(case)
                lines.append(f"{case_id}\t{doc['ids_path']}\t{p}\t{h}\t{TARGET}\n")
    if len(frozen["cases"]) != 24:
        raise ValueError("Expected 24 nested span/history cases")
    payload = json.dumps(frozen, indent=2) + "\n"
    path = HERE / "FRESH_STATE_FREEZE.json"
    if path.exists() and path.read_text() != payload:
        raise ValueError("Frozen manifest changed")
    path.write_text(payload)
    tsv = HERE / "cases.tsv"
    if tsv.exists() and tsv.read_text() != "".join(lines):
        raise ValueError("Frozen case TSV changed")
    tsv.write_text("".join(lines))
    print("Frozen six fresh books, 12 disjoint spans, 24 nested cases, three existing checkpoints")


if __name__ == "__main__":
    main()

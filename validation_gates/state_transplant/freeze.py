"""Freeze minimal same-model state roundtrip and cross-model import fixtures."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    scorer = json.loads((ROOT / "analysis/bonsai2/scorer/frozen_manifest.json").read_text())
    dose_path = ROOT / "analysis/bonsai2/dose_calibration/DOSE_FREEZE.json"
    dose = json.loads(dose_path.read_text())
    doc = scorer["documents"][0]
    if doc["id"] != "pride_and_prejudice":
        raise ValueError("First frozen scorer document changed")
    variant = next(v for v in dose["selected_variants"] if v["arm"] == "alpha" and v["seed"] == 20260929)
    original_model = ROOT / scorer["model_path"]
    if sha(original_model) != scorer["model_sha256"]:
        raise ValueError("Original checkpoint changed")
    if sha(ROOT / variant["model_path"]) != variant["model_sha256"]:
        raise ValueError("Selected alpha checkpoint changed")
    if sha(ROOT / variant["patch_manifest_path"]) != variant["patch_manifest_sha256"]:
        raise ValueError("Selected alpha patch changed")
    if sha(ROOT / doc["ids_path"]) != doc["ids_file_sha256"] or sha(ROOT / doc["source_path"]) != doc["source_sha256"]:
        raise ValueError("Frozen document changed")
    executable = HERE / "state_gate"
    if not executable.exists():
        raise ValueError("Build state_gate first")
    linked = subprocess.run(["ldd", str(executable)], text=True, capture_output=True, check=True).stdout
    libraries = {}
    for path in re.findall(r"/[^\s()]+", linked):
        resolved = Path(path)
        if resolved.is_file() and str(resolved).startswith(str(ROOT / "bin/cuda")):
            libraries[str(resolved.relative_to(ROOT))] = sha(resolved)
    if not any(Path(path).name.startswith("libllama") for path in libraries):
        raise ValueError("Pinned libllama not identified")
    cases = [c for c in scorer["cases"] if c["document_id"] == doc["id"] and
             c["history_length"] in (1024, 12288)]
    if len(cases) != 2 or len({c["target_ids_sha256"] for c in cases}) != 1:
        raise ValueError("State fixture target identity invalid")
    frozen = {"purpose": "state serialization identity and cross-checkpoint recurrent-only import feasibility",
        "source_release": scorer["runtime_release"],
        "pinned_source_archive_sha256": "84ec38b7e7fb45a9f076e923e064e967e7c30b946b71b3b778444b05e8459c3c",
        "state_gate_source_sha256": sha(HERE / "state_gate.cpp"),
        "state_gate_binary_sha256": sha(executable), "linked_cuda_library_sha256": libraries,
        "original_model_path": scorer["model_path"], "original_model_sha256": scorer["model_sha256"],
        "variant": variant, "dose_freeze_sha256": sha(dose_path),
        "document": doc, "cases": cases,
        "prefill_boundary": "feed all h history IDs in the same 512-token batches as the validated scorer; target 0 uses the last history logit, then save state and decode gold target IDs to score targets 1..95. A state import can affect only targets 1..95.",
        "profile": {"n_ctx": 16384, "n_seq_max": 1, "n_batch": 512,
                    "n_ubatch": 512, "flash_attention": "on", "kv": "f16", "gpu_layers": 99},
        "identity_gate": "Same-model full and recurrent-only-overlay roundtrips must match direct per-token NLL with maximum absolute difference <=0.0001 nat at both histories; cross-model import is attempted only after all identities pass",
        "scope": "one first-in-manifest document, one alpha seed, two nested histories; exploratory feasibility, not a mechanism estimate"}
    target = HERE / "STATE_GATE_FREEZE.json"
    encoded = json.dumps(frozen, indent=2) + "\n"
    if target.exists() and target.read_text() != encoded:
        raise ValueError("Frozen state-gate manifest changed")
    target.write_text(encoded)
    print("Frozen state gate: two histories, original and one alpha checkpoint")


if __name__ == "__main__":
    main()

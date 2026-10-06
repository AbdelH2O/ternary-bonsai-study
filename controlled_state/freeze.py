"""Freeze a controlled state-import method pilot before new CUDA inference."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PREVIOUS = ROOT / "analysis/bonsai2/validation_gates/state_transplant"


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
    prior_gate_path = PREVIOUS / "gate_summary.json"
    prior_gate = json.loads(prior_gate_path.read_text())
    if not prior_gate["passed"] or len(prior_gate["captures"]) != 4:
        raise ValueError("Prior state identity gate did not pass")
    docs = {d["id"]: d for d in scorer["documents"]}
    destination = docs["pride_and_prejudice"]
    unrelated = docs["dracula"]
    for doc in (destination, unrelated):
        if sha(ROOT / doc["ids_path"]) != doc["ids_file_sha256"]:
            raise ValueError(f"Document IDs changed: {doc['id']}")
    original = {"arm": "original", "model_path": scorer["model_path"],
                "model_sha256": scorer["model_sha256"], "patch_manifest_sha256": None}
    models = {"original": original}
    for arm in ("alpha", "mlp"):
        variant = next(v for v in dose["selected_variants"] if v["arm"] == arm and v["seed"] == 20260929)
        models[arm] = variant
    for arm, model in models.items():
        if sha(ROOT / model["model_path"]) != model["model_sha256"]:
            raise ValueError(f"Model changed: {arm}")
    binary = HERE / "controlled_state"
    if not binary.exists():
        raise ValueError("Build controlled_state first")
    linked = subprocess.run(["ldd", str(binary)], capture_output=True, text=True, check=True).stdout
    libraries = {}
    for path in re.findall(r"/[^\s()]+", linked):
        resolved = Path(path)
        if resolved.is_file() and str(resolved).startswith(str(ROOT / "bin/cuda")):
            libraries[str(resolved.relative_to(ROOT))] = sha(resolved)
    if not any(Path(path).name.startswith("libllama") for path in libraries):
        raise ValueError("Pinned libllama not identified")
    previous_states = {}
    for arm in ("original", "alpha"):
        for h in (1024, 12288):
            prior = next(r for r in prior_gate["captures"] if r["arm"] == arm and r["history"] == h)
            for kind, suffix in (("full", "full"), ("partial", "recurrent")):
                path = PREVIOUS / f"{arm}-h{h}-{suffix}.bin"
                expected = prior["full_state_sha256"] if kind == "full" else prior["recurrent_state_sha256"]
                if sha(path) != expected:
                    raise ValueError(f"Prior state changed: {path}")
                previous_states[f"{arm}-h{h}-{kind}"] = {"path": str(path.relative_to(ROOT)),
                    "sha256": expected, "bytes": path.stat().st_size}
    comparisons = {}
    for h in (1024, 12288):
        comparisons[f"original-h{h}"] = [
            {"label": "unrelated_recurrent", "kind": "partial", "source": "unrelated"},
            {"label": "alpha_recurrent", "kind": "partial", "source": "alpha"},
            {"label": "alpha_full", "kind": "full", "source": "alpha"},
            {"label": "mlp_recurrent", "kind": "partial", "source": "mlp"},
            {"label": "mlp_full", "kind": "full", "source": "mlp"},
        ]
        for arm in ("alpha", "mlp"):
            comparisons[f"{arm}-h{h}"] = [
                {"label": "original_recurrent", "kind": "partial", "source": "original"},
                {"label": "original_full", "kind": "full", "source": "original"},
            ]
    frozen = {"purpose": "controlled reciprocal/full/recurrent state-import method pilot",
        "scorer_manifest_sha256": sha(ROOT / "analysis/bonsai2/scorer/frozen_manifest.json"),
        "dose_freeze_sha256": sha(dose_path), "prior_gate_summary_sha256": sha(prior_gate_path),
        "program_source_sha256": sha(HERE / "controlled_state.cpp"),
        "program_binary_sha256": sha(binary), "linked_cuda_library_sha256": libraries,
        "models": models, "destination_document": destination, "unrelated_source_document": unrelated,
        "target_position": 30000, "target_length": 96, "histories": [1024, 12288],
        "existing_state_files": previous_states, "comparisons": comparisons,
        "profile": {"n_ctx": 16384, "n_seq_max": 1, "n_batch": 512, "n_ubatch": 512,
                    "flash_attention": "on", "kv": "f16", "gpu_layers": 99},
        "controls": {"aba_identity_max_token_nll_delta": 0.0001,
                     "direct_vs_prior_scorer_max_token_nll_delta": 0.0001,
                     "unrelated_recurrent_mean_absolute_token_nll_delta_min": 0.01,
                     "state_import_scope": "target 0 is from destination prefill and cannot respond; compare targets 1..95 for state-mediated effects"},
        "interpretation": "Method pilot on one previously examined book and one sign seed; reciprocal/full and unrelated-history controls do not establish a population effect"}
    out = HERE / "CONTROLLED_STATE_FREEZE.json"
    encoded = json.dumps(frozen, indent=2) + "\n"
    if out.exists() and out.read_text() != encoded:
        raise ValueError("Controlled state freeze changed")
    out.write_text(encoded)
    print("Frozen six destination/history runs and reciprocal/full/unrelated controls")


if __name__ == "__main__":
    main()

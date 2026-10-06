"""Score frozen held-out cases with the original and six calibrated variants."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SCORER = ROOT / "analysis/bonsai2/scorer"
DOSE = ROOT / "analysis/bonsai2/dose_calibration"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def validate_raw(path, frozen):
    rows = read_rows(path)
    if len(rows) != 12 or [r["case_id"] for r in rows] != frozen["case_ids"]:
        raise ValueError(f"Case identity differs: {path}")
    for row in rows:
        if (row["p"], row["h"], row["m"], row["n_batch"], row["n_ubatch"]) != (
                30000, int(row["case_id"].split("-h")[-1]), 96, 512, 512):
            raise ValueError(f"Profile differs: {path} {row['case_id']}")
        if len(row["nll"]) != 96:
            raise ValueError(f"Target count differs: {path} {row['case_id']}")
    return rows


def score(model, path, frozen):
    if path.exists():
        return validate_raw(path, frozen)
    temp = path.with_suffix(".partial.jsonl")
    log = path.with_suffix(".log")
    with log.open("w") as stream:
        proc = subprocess.run([str(SCORER / "position_scorer"), "score", str(model),
                               str(SCORER / "baseline_cases.tsv"), str(temp), "512", "512"],
                              cwd=ROOT, stdout=stream, stderr=stream)
    if proc.returncode:
        raise RuntimeError(f"Scorer failed for {model.name}; see {log}")
    rows = validate_raw(temp, frozen)
    temp.rename(path)
    return rows


def main():
    frozen = json.loads((HERE / "HELDOUT_FREEZE.json").read_text())
    for path, expected in ((SCORER / "frozen_manifest.json", frozen["scorer_manifest_sha256"]),
                           (DOSE / "DOSE_FREEZE.json", frozen["dose_freeze_sha256"]),
                           (SCORER / "baseline_results.jsonl", frozen["baseline_results_sha256"]),
                           (SCORER / "baseline_cases.tsv", frozen["cases_tsv_sha256"]),
                           (SCORER / "position_scorer.cpp", frozen["scorer_source_sha256"]),
                           (SCORER / "position_scorer", frozen["scorer_binary_sha256"])):
        if sha(path) != expected:
            raise ValueError(f"Frozen artifact changed: {path}")
    baseline_model = ROOT / frozen["original_model_path"]
    if sha(baseline_model) != frozen["original_model_sha256"]:
        raise ValueError("Original model changed")
    prior = {r["case_id"]: r for r in read_rows(SCORER / "baseline_results.jsonl")}
    rerun = score(baseline_model, HERE / "baseline_rerun.jsonl", frozen)
    max_token = max(abs(a - b) for r in rerun
                    for a, b in zip(r["nll"], prior[r["case_id"]]["gold_nll_nats"]))
    print(f"Baseline rerun max per-token NLL delta: {max_token:.9g}", flush=True)
    if max_token > 0.0001:
        raise ValueError("Baseline numerical floor changed; stop before treatment scoring")
    for variant in frozen["variants"]:
        name = variant["name"]
        model = ROOT / variant["model_path"]
        patch = ROOT / variant["patch_manifest_path"]
        if sha(model) != variant["model_sha256"] or sha(patch) != variant["patch_manifest_sha256"]:
            raise ValueError(f"Variant or patch hash mismatch: {name}")
        rows = score(model, HERE / f"{name}.jsonl", frozen)
        print(f"Scored {name}: {len(rows)} cases, {sum(r['seconds'] for r in rows):.1f} inference seconds", flush=True)


if __name__ == "__main__":
    main()

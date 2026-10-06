"""Run the frozen answer-margin fixture with the validated sparse scorer."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
SCORER = ROOT / "analysis/bonsai2/scorer/position_scorer"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(raw_path, frozen):
    rows = [json.loads(x) for x in raw_path.read_text().splitlines()]
    cases = frozen["cases"]
    if len(rows) != len(cases) or [r["case_id"] for r in rows] != [c["case_id"] for c in cases]:
        raise ValueError(f"Output case mismatch: {raw_path}")
    for row, case in zip(rows, cases):
        if (row["p"], row["h"], row["m"], row["n_batch"], row["n_ubatch"], len(row["nll"])) != (
                case["prefix_token_count"], case["prefix_token_count"],
                case["target_token_count"], 512, 512, case["target_token_count"]):
            raise ValueError(f"Output profile mismatch: {row['case_id']}")


def score(model, output, frozen):
    if output.exists():
        validate(output, frozen)
        print(f"Existing verified output: {output.name}", flush=True)
        return
    temp = output.with_suffix(".partial.jsonl")
    log = output.with_suffix(".log")
    with log.open("w") as stream:
        proc = subprocess.run([str(SCORER), "score", str(model), str(HERE / "cases.tsv"),
                               str(temp), "512", "512"], cwd=ROOT, stdout=stream, stderr=stream)
    if proc.returncode:
        raise RuntimeError(f"Scorer failed; see {log}")
    validate(temp, frozen)
    temp.rename(output)
    print(f"Scored {output.name}: {len(frozen['cases'])} cases", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("baseline", "variants"))
    args = parser.parse_args()
    frozen = json.loads((HERE / "ANSWER_MARGIN_FREEZE.json").read_text())
    if sha(ROOT / "analysis/bonsai2/prompts/retrieval_v2.jsonl") != frozen["prompt_file_sha256"]:
        raise ValueError("Prompt file changed")
    if sha(SCORER) != frozen["scorer_binary_sha256"] or sha(SCORER.with_suffix(".cpp")) != frozen["scorer_source_sha256"]:
        raise ValueError("Scorer changed")
    for case in frozen["cases"]:
        if sha(ROOT / case["full_ids_file"]) != case["full_ids_file_sha256"]:
            raise ValueError(f"Token ID file changed: {case['case_id']}")
    if args.phase == "baseline":
        model = ROOT / frozen["model_path"]
        if sha(model) != frozen["model_sha256"]:
            raise ValueError("Baseline model changed")
        score(model, HERE / "baseline.jsonl", frozen)
    else:
        baseline_gate = json.loads((HERE / "baseline_gate.json").read_text())
        if not baseline_gate["passed"]:
            raise ValueError("Baseline answer-margin gate did not pass")
        dose_file = ROOT / "analysis/bonsai2/dose_calibration/DOSE_FREEZE.json"
        if sha(dose_file) != frozen["dose_freeze_sha256"]:
            raise ValueError("Dose freeze changed")
        for variant in frozen["selected_variants"]:
            model = ROOT / variant["model_path"]
            patch = ROOT / variant["patch_manifest_path"]
            if sha(model) != variant["model_sha256"] or sha(patch) != variant["patch_manifest_sha256"]:
                raise ValueError(f"Variant changed: {variant['name']}")
            score(model, HERE / f"{variant['name']}.jsonl", frozen)


if __name__ == "__main__":
    main()

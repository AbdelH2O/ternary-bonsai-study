"""Score the frozen margin-extension candidates on host CUDA."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SCORER = ROOT / "analysis/bonsai2/scorer/position_scorer"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cases_for(frozen, phase):
    return [c for c in frozen["cases"] if phase == "baseline" or c["control"] == "present"]


def verify_inputs(frozen, phase):
    if sha(ROOT / "analysis/bonsai2/prompts/retrieval_v2.jsonl") != frozen["prompt_file_sha256"]:
        raise ValueError("Prompt source changed")
    if sha(SCORER) != frozen["scorer_binary_sha256"] or sha(SCORER.with_suffix(".cpp")) != frozen["scorer_source_sha256"]:
        raise ValueError("Scorer changed")
    cases = cases_for(frozen, phase)
    tsv = HERE / ("baseline_cases.tsv" if phase == "baseline" else "variant_cases.tsv")
    expected = "".join(f"{c['case_id']}\t{c['full_ids_file']}\t{c['prefix_token_count']}\t"
                       f"{c['prefix_token_count']}\t{c['target_token_count']}\n" for c in cases)
    if tsv.read_text() != expected:
        raise ValueError(f"Case list changed: {tsv}")
    for case in cases:
        if sha(ROOT / case["full_ids_file"]) != case["full_ids_file_sha256"]:
            raise ValueError(f"Token IDs changed: {case['case_id']}")
    return cases, tsv


def validate(path, cases):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != len(cases) or [r["case_id"] for r in rows] != [c["case_id"] for c in cases]:
        raise ValueError(f"Output case mismatch: {path}")
    for row, case in zip(rows, cases):
        actual = (row["p"], row["h"], row["m"], row["n_batch"], row["n_ubatch"], len(row["nll"]))
        expected = (case["prefix_token_count"], case["prefix_token_count"],
                    case["target_token_count"], 512, 512, case["target_token_count"])
        if actual != expected:
            raise ValueError(f"Output profile mismatch: {case['case_id']}")


def score(model, tsv, output, cases):
    if output.exists():
        validate(output, cases)
        print(f"Existing verified output: {output.name}", flush=True)
        return
    temp = output.with_suffix(".partial.jsonl")
    with output.with_suffix(".log").open("w") as log:
        proc = subprocess.run([str(SCORER), "score", str(model), str(tsv), str(temp), "512", "512"],
                              cwd=ROOT, stdout=log, stderr=log)
    if proc.returncode:
        raise RuntimeError(f"Scorer failed; see {output.with_suffix('.log')}")
    validate(temp, cases)
    temp.rename(output)
    print(f"Scored {output.name}: {len(cases)} cases", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("baseline", "variants"))
    args = parser.parse_args()
    frozen = json.loads((HERE / "MARGIN_EXTENSION_FREEZE.json").read_text())
    phase = args.phase
    cases, tsv = verify_inputs(frozen, phase)
    if phase == "baseline":
        model = ROOT / frozen["model_path"]
        if sha(model) != frozen["model_sha256"]:
            raise ValueError("Baseline model changed")
        score(model, tsv, HERE / "baseline.jsonl", cases)
        return
    gate = json.loads((HERE / "baseline_gate.json").read_text())
    if gate["freeze_sha256"] != sha(HERE / "MARGIN_EXTENSION_FREEZE.json") or not gate["passed"]:
        raise ValueError("Baseline gate not passed for this freeze")
    if sha(ROOT / "analysis/bonsai2/dose_calibration/DOSE_FREEZE.json") != frozen["dose_freeze_sha256"]:
        raise ValueError("Dose freeze changed")
    for variant in frozen["variants"]:
        model = ROOT / variant["model_path"]
        patch = ROOT / variant["patch_manifest_path"]
        if sha(model) != variant["model_sha256"] or sha(patch) != variant["patch_manifest_sha256"]:
            raise ValueError(f"Variant changed: {variant['name']}")
        score(model, tsv, HERE / f"{variant['name']}.jsonl", cases)


if __name__ == "__main__":
    main()

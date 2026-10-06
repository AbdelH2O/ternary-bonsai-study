"""Run frozen reciprocal, full-state, and unrelated-history controls on host CUDA."""

import hashlib
import json
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PREVIOUS = ROOT / "analysis/bonsai2/validation_gates/state_transplant"
SCORER = ROOT / "analysis/bonsai2/scorer"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def maxdiff(a, b):
    if len(a) != len(b):
        raise ValueError("NLL vector lengths differ")
    return max(abs(x - y) for x, y in zip(a, b))


def run_process(cmd, output):
    if output.exists():
        return json.loads(output.read_text())
    temp = output.with_suffix(".partial.json")
    with output.with_suffix(".log").open("w") as log:
        result = subprocess.run(cmd[:-1] + [str(temp)], cwd=ROOT, stdout=log, stderr=log)
    if result.returncode:
        raise RuntimeError(f"CUDA program failed: see {output.with_suffix('.log')}")
    row = json.loads(temp.read_text())
    temp.rename(output)
    return row


def prior_artifacts(path):
    return json.loads(path.read_text()) if path.exists() else None


def capture(arm, model, doc, h, reference, old_captures):
    stem = f"{arm}-h{h}"
    ids = ROOT / doc["ids_path"]
    full = HERE / f"{stem}-full.bin"
    partial = HERE / f"{stem}-recurrent.bin"
    output = HERE / f"{stem}-capture.json"
    cmd = [str(PREVIOUS / "state_gate"), "capture", str(model), str(ids), "30000",
           str(h), "96", str(full), str(partial), str(output)]
    reusing = output.exists()
    if reusing and old_captures is None:
        raise ValueError(f"Source capture has no provenance summary: {stem}")
    row = run_process(cmd, output)
    if row["mode"] != "capture" or row["model_path"] != str(model) or row["ids_path"] != str(ids) or (
            row["p"], row["h"], row["m"]) != (30000, h, 96):
        raise ValueError(f"Source capture identity mismatch: {stem}")
    if not full.exists() or not partial.exists():
        raise ValueError(f"Source state file missing: {stem}")
    if reusing:
        old = next((x for x in old_captures["captures"]
                    if x["arm"] == arm and x["history"] == h), None)
        if old is None or sha(full) != old["full_state_sha256"] or sha(partial) != old["partial_state_sha256"]:
            raise ValueError(f"Previously scored source state changed: {stem}")
    checks = {"full": maxdiff(row["direct_nll"], row["full_roundtrip_nll"]),
              "partial": maxdiff(row["direct_nll"], row["partial_roundtrip_nll"]),
              "prior": maxdiff(row["direct_nll"], reference)}
    if max(checks.values()) > 0.0001:
        raise ValueError(f"Source capture identity failed: {stem}: {checks}")
    print(f"Captured {stem}: {checks}", flush=True)
    return {"arm": arm, "history": h, "model_sha256": sha(model),
            "ids_sha256": sha(ids), "checks": checks,
            "full_state_path": str(full.relative_to(ROOT)), "full_state_sha256": sha(full),
            "partial_state_path": str(partial.relative_to(ROOT)), "partial_state_sha256": sha(partial)}


def state_path(source, h, kind):
    suffix = "full" if kind == "full" else "recurrent"
    parent = PREVIOUS if source in ("original", "alpha") else HERE
    return parent / f"{source}-h{h}-{suffix}.bin"


def compare(arm, model, doc, h, specifications, reference, old_summary):
    stem = f"{arm}-h{h}"
    ids = ROOT / doc["ids_path"]
    output = HERE / f"{stem}-compare.json"
    conditions = [(s["label"], s["kind"], state_path(s["source"], h, s["kind"]))
                  for s in specifications]
    cmd = [str(HERE / "controlled_state"), str(model), str(ids), "30000", str(h), "96", str(output)]
    for label, kind, source in conditions:
        cmd.extend((label, kind, str(source)))
    # The controlled_state output argument precedes condition triples, so the temp path
    # must replace argv[6] rather than the final argument.
    if output.exists():
        if old_summary is None:
            raise ValueError(f"Comparison output has no provenance summary: {stem}")
        old = next((x for x in old_summary["comparisons"]
                    if x["destination_arm"] == arm and x["history"] == h), None)
        if old is None or len(old["conditions"]) != len(conditions):
            raise ValueError(f"Existing comparison provenance missing: {stem}")
        for previous, (label, kind, source) in zip(old["conditions"], conditions):
            if (previous["label"], previous["kind"], previous["source_state_sha256"]) != (label, kind, sha(source)):
                raise ValueError(f"Source state changed since comparison: {stem} {label}")
        row = json.loads(output.read_text())
    else:
        temp = output.with_suffix(".partial.json")
        cmd[6] = str(temp)
        with output.with_suffix(".log").open("w") as log:
            result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=log)
        if result.returncode:
            raise RuntimeError(f"Controlled import failed: see {output.with_suffix('.log')}")
        row = json.loads(temp.read_text())
        temp.rename(output)
    if row["model_path"] != str(model) or row["ids_path"] != str(ids) or (
            row["p"], row["h"], row["m"]) != (30000, h, 96):
        raise ValueError(f"Comparison identity mismatch: {stem}")
    if len(row["conditions"]) != len(conditions):
        raise ValueError(f"Comparison condition count mismatch: {stem}")
    direct_delta = maxdiff(row["direct_nll"], reference)
    if direct_delta > 0.0001:
        raise ValueError(f"Comparison direct score differs from scorer: {stem}: {direct_delta}")
    result = {"destination_arm": arm, "history": h, "direct_prior_max_token_delta": direct_delta,
              "conditions": []}
    for actual, (label, kind, source) in zip(row["conditions"], conditions):
        if (actual["label"], actual["kind"], actual["state_path"]) != (label, kind, str(source)):
            raise ValueError(f"Condition identity mismatch: {stem} {label}")
        if len(actual["imported_nll"]) != 96 or len(actual["restored_nll"]) != 96:
            raise ValueError(f"Target count mismatch: {stem} {label}")
        aba = maxdiff(row["direct_nll"], actual["restored_nll"])
        if aba > 0.0001:
            raise ValueError(f"A-B-A restore failed: {stem} {label}: {aba}")
        if actual["imported_nll"][0] != row["direct_nll"][0]:
            raise ValueError(f"Unchanged first target mismatch: {stem} {label}")
        delta = [a - b for a, b in zip(actual["imported_nll"][1:], row["direct_nll"][1:])]
        entry = {"label": label, "kind": kind, "source_state_sha256": sha(source),
                 "aba_max_token_delta": aba,
                 "mean_state_affected_excess_nll": statistics.mean(delta),
                 "mean_absolute_state_affected_delta": statistics.mean(abs(x) for x in delta),
                 "max_absolute_state_affected_delta": max(abs(x) for x in delta)}
        result["conditions"].append(entry)
        print(f"{stem} {label}: mean={entry['mean_state_affected_excess_nll']:.6g} "
              f"mean_abs={entry['mean_absolute_state_affected_delta']:.6g} aba={aba}", flush=True)
    return result


def main():
    frozen = json.loads((HERE / "CONTROLLED_STATE_FREEZE.json").read_text())
    if sha(HERE / "controlled_state.cpp") != frozen["program_source_sha256"] or sha(HERE / "controlled_state") != frozen["program_binary_sha256"]:
        raise ValueError("Controlled-state code changed")
    if sha(SCORER / "frozen_manifest.json") != frozen["scorer_manifest_sha256"] or sha(PREVIOUS / "gate_summary.json") != frozen["prior_gate_summary_sha256"]:
        raise ValueError("Scorer or prior gate changed")
    if sha(ROOT / "analysis/bonsai2/dose_calibration/DOSE_FREEZE.json") != frozen["dose_freeze_sha256"]:
        raise ValueError("Dose freeze changed")
    prior_freeze = json.loads((PREVIOUS / "STATE_GATE_FREEZE.json").read_text())
    if sha(PREVIOUS / "state_gate") != prior_freeze["state_gate_binary_sha256"]:
        raise ValueError("Source capture binary changed")
    for relative, digest in frozen["linked_cuda_library_sha256"].items():
        if sha(ROOT / relative) != digest:
            raise ValueError(f"Linked library changed: {relative}")
    for item in frozen["existing_state_files"].values():
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"Existing state changed: {item['path']}")
    for arm, model in frozen["models"].items():
        if sha(ROOT / model["model_path"]) != model["model_sha256"]:
            raise ValueError(f"Model changed: {arm}")
    for doc in (frozen["destination_document"], frozen["unrelated_source_document"]):
        if sha(ROOT / doc["ids_path"]) != doc["ids_file_sha256"]:
            raise ValueError(f"Document IDs changed: {doc['id']}")
    baseline = {r["case_id"]: r["gold_nll_nats"] for r in map(json.loads,
        (SCORER / "baseline_results.jsonl").read_text().splitlines())}
    variants = {arm: {r["case_id"]: r["nll"] for r in map(json.loads,
        (ROOT / f"analysis/bonsai2/heldout_nll/{frozen['models'][arm]['name']}.jsonl").read_text().splitlines())}
        for arm in ("alpha", "mlp")}
    old_captures = prior_artifacts(HERE / "capture_summary.json")
    old_summary = prior_artifacts(HERE / "summary.json")
    captures = []
    for h in frozen["histories"]:
        captures.append(capture("mlp", ROOT / frozen["models"]["mlp"]["model_path"],
            frozen["destination_document"], h, variants["mlp"][f"pride_and_prejudice-p30000-h{h}"], old_captures))
        captures.append(capture("unrelated", ROOT / frozen["models"]["original"]["model_path"],
            frozen["unrelated_source_document"], h, baseline[f"dracula-p30000-h{h}"], old_captures))
    (HERE / "capture_summary.json").write_text(json.dumps({"captures": captures}, indent=2) + "\n")
    comparisons = []
    for h in frozen["histories"]:
        for arm in ("original", "alpha", "mlp"):
            model = frozen["models"][arm]
            case = f"pride_and_prejudice-p30000-h{h}"
            reference = baseline[case] if arm == "original" else variants[arm][case]
            result = compare(arm, ROOT / model["model_path"], frozen["destination_document"], h,
                             frozen["comparisons"][f"{arm}-h{h}"], reference, old_summary)
            comparisons.append(result)
            if arm == "original":
                positive = next(x for x in result["conditions"] if x["label"] == "unrelated_recurrent")
                threshold = frozen["controls"]["unrelated_recurrent_mean_absolute_token_nll_delta_min"]
                if positive["mean_absolute_state_affected_delta"] < threshold:
                    raise ValueError(f"Unrelated-history control below threshold at h={h}")
    summary = {"freeze_sha256": sha(HERE / "CONTROLLED_STATE_FREEZE.json"),
               "capture_summary_sha256": sha(HERE / "capture_summary.json"),
               "controls_passed": True, "comparisons": comparisons,
               "interpretation": frozen["interpretation"]}
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

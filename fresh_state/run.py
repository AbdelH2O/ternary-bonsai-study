"""Run the frozen fresh-document state assay in four gated host-CUDA phases."""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SCORER = ROOT / "analysis/bonsai2/scorer/position_scorer"
STATE_GATE = ROOT / "analysis/bonsai2/validation_gates/state_transplant/state_gate"
CONTROLLED = ROOT / "analysis/bonsai2/controlled_state/controlled_state"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def maxdiff(a, b):
    if len(a) != len(b):
        raise ValueError("NLL vector lengths differ")
    return max(abs(x - y) for x, y in zip(a, b))


def load_freeze():
    frozen = json.loads((HERE / "FRESH_STATE_FREEZE.json").read_text())
    if sha(HERE / "DESIGN.md") != frozen["design_sha256"]:
        raise ValueError("Design changed after freeze")
    if sha(ROOT / "analysis/bonsai2/controlled_state/CONTROLLED_STATE_FREEZE.json") != frozen["prior_controlled_state_freeze_sha256"]:
        raise ValueError("Prior state gate changed")
    if sha(ROOT / "analysis/bonsai2/dose_calibration/DOSE_FREEZE.json") != frozen["dose_freeze_sha256"]:
        raise ValueError("Dose freeze changed")
    for name, digest in {**frozen["program_hashes"], **frozen["linked_cuda_library_sha256"]}.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Program or Prism library changed: {name}")
    for arm, model in frozen["models"].items():
        if sha(ROOT / model["model_path"]) != model["model_sha256"]:
            raise ValueError(f"Model changed: {arm}")
        if model.get("patch_manifest_path") and sha(ROOT / model["patch_manifest_path"]) != model["patch_manifest_sha256"]:
            raise ValueError(f"Patch manifest changed: {arm}")
    by_doc = {}
    for doc in frozen["documents"]:
        if sha(ROOT / doc["source_path"]) != doc["source_sha256"] or sha(ROOT / doc["ids_path"]) != doc["ids_file_sha256"]:
            raise ValueError(f"Book or token IDs changed: {doc['id']}")
        by_doc[doc["id"]] = doc
    expected = "".join(f"{c['case_id']}\t{by_doc[c['document_id']]['ids_path']}\t"
                       f"{c['target_position']}\t{c['history_length']}\t{c['target_length']}\n"
                       for c in frozen["cases"])
    if (HERE / "cases.tsv").read_text() != expected:
        raise ValueError("Frozen cases changed")
    return frozen, by_doc


def verify_gate(path, frozen):
    row = json.loads(path.read_text())
    if row["freeze_sha256"] != sha(HERE / "FRESH_STATE_FREEZE.json") or not row["passed"]:
        raise ValueError(f"Prerequisite gate failed: {path}")
    return row


def reference_rows(frozen):
    result = {}
    for arm in frozen["models"]:
        path = HERE / f"reference-{arm}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != len(frozen["cases"]) or [r["case_id"] for r in rows] != [c["case_id"] for c in frozen["cases"]]:
            raise ValueError(f"Reference identities differ: {arm}")
        for row, case in zip(rows, frozen["cases"]):
            if (row["p"], row["h"], row["m"], row["n_batch"], row["n_ubatch"], len(row["nll"])) != (
                    case["target_position"], case["history_length"], case["target_length"], 512, 512, 96):
                raise ValueError(f"Reference profile differs: {arm} {case['case_id']}")
        result[arm] = {row["case_id"]: row["nll"] for row in rows}
    return result


def reference(frozen):
    for arm, model in frozen["models"].items():
        output = HERE / f"reference-{arm}.jsonl"
        if not output.exists():
            temp = output.with_suffix(".partial.jsonl")
            with output.with_suffix(".log").open("w") as log:
                result = subprocess.run([str(SCORER), "score", str(ROOT / model["model_path"]),
                                         str(HERE / "cases.tsv"), str(temp), "512", "512"],
                                        cwd=ROOT, stdout=log, stderr=log)
            if result.returncode:
                raise RuntimeError(f"Reference scorer failed: {output.with_suffix('.log')}")
            os.replace(temp, output)
        print(f"Reference {arm}: {output.name}", flush=True)
    reference_rows(frozen)
    write_json(HERE / "reference_gate.json", {"freeze_sha256": sha(HERE / "FRESH_STATE_FREEZE.json"),
        "passed": True, "raw_sha256": {arm: sha(HERE / f"reference-{arm}.jsonl") for arm in frozen["models"]}})


def capture_paths(case_id, arm):
    stem = f"{case_id}-{arm}"
    base = HERE / "captures"
    return (base / f"{stem}-full.bin", base / f"{stem}-recurrent.bin",
            base / f"{stem}.json", base / f"{stem}.meta.json")


def checked_capture(case, arm, model, ids, reference_nll, frozen):
    full, recurrent, raw, meta = capture_paths(case["case_id"], arm)
    if meta.exists():
        old = json.loads(meta.read_text())
        if (old["freeze_sha256"] != sha(HERE / "FRESH_STATE_FREEZE.json") or
                old["full_sha256"] != sha(full) or old["recurrent_sha256"] != sha(recurrent) or
                old["raw_sha256"] != sha(raw)):
            raise ValueError(f"Captured artifact changed: {case['case_id']} {arm}")
    else:
        temp_full = full.with_suffix(".partial.bin")
        temp_recurrent = recurrent.with_suffix(".partial.bin")
        temp_raw = raw.with_suffix(".partial.json")
        with raw.with_suffix(".log").open("w") as log:
            result = subprocess.run([str(STATE_GATE), "capture", str(model), str(ids),
                str(case["target_position"]), str(case["history_length"]), "96",
                str(temp_full), str(temp_recurrent), str(temp_raw)],
                cwd=ROOT, stdout=log, stderr=log)
        if result.returncode:
            raise RuntimeError(f"State capture failed: {raw.with_suffix('.log')}")
        row = json.loads(temp_raw.read_text())
        validate_capture(row, case, model, ids, reference_nll, frozen)
        os.replace(temp_full, full)
        os.replace(temp_recurrent, recurrent)
        os.replace(temp_raw, raw)
        write_json(meta, {"freeze_sha256": sha(HERE / "FRESH_STATE_FREEZE.json"),
            "full_sha256": sha(full), "recurrent_sha256": sha(recurrent), "raw_sha256": sha(raw),
            "model_sha256": frozen["models"][arm]["model_sha256"],
            "ids_file_sha256": sha(ids)})
    row = json.loads(raw.read_text())
    checks = validate_capture(row, case, model, ids, reference_nll, frozen)
    return {"case_id": case["case_id"], "arm": arm, "full_path": str(full.relative_to(ROOT)),
            "full_sha256": sha(full), "full_bytes": full.stat().st_size,
            "recurrent_path": str(recurrent.relative_to(ROOT)),
            "recurrent_sha256": sha(recurrent), "recurrent_bytes": recurrent.stat().st_size,
            "raw_path": str(raw.relative_to(ROOT)), "raw_sha256": sha(raw), "checks": checks}


def validate_capture(row, case, model, ids, reference_nll, frozen):
    if (row["mode"], row["model_path"], row["ids_path"], row["p"], row["h"], row["m"]) != (
            "capture", str(model), str(ids), case["target_position"], case["history_length"], 96):
        raise ValueError(f"Capture identity differs: {case['case_id']}")
    checks = {"direct_reference_max": maxdiff(row["direct_nll"], reference_nll),
              "full_roundtrip_max": maxdiff(row["direct_nll"], row["full_roundtrip_nll"]),
              "recurrent_roundtrip_max": maxdiff(row["direct_nll"], row["partial_roundtrip_nll"])}
    if max(checks.values()) > frozen["gates"]["identity_max_token_nll_delta"]:
        raise ValueError(f"State identity failed: {case['case_id']}: {checks}")
    return checks


def capture(frozen, by_doc):
    gate = verify_gate(HERE / "reference_gate.json", frozen)
    if gate["raw_sha256"] != {arm: sha(HERE / f"reference-{arm}.jsonl") for arm in frozen["models"]}:
        raise ValueError("Reference raw scores changed")
    refs = reference_rows(frozen)
    records = []
    for case in frozen["cases"]:
        ids = ROOT / by_doc[case["document_id"]]["ids_path"]
        for arm, model in frozen["models"].items():
            record = checked_capture(case, arm, ROOT / model["model_path"], ids,
                                     refs[arm][case["case_id"]], frozen)
            records.append(record)
            print(f"Captured {case['case_id']} {arm}: {record['checks']}", flush=True)
    write_json(HERE / "capture_gate.json", {"freeze_sha256": sha(HERE / "FRESH_STATE_FREEZE.json"),
        "passed": True, "captures": records})


def verify_captures(frozen):
    gate = verify_gate(HERE / "capture_gate.json", frozen)
    if len(gate["captures"]) != 72:
        raise ValueError("Incomplete state capture gate")
    result = {}
    for item in gate["captures"]:
        for kind in ("full", "recurrent"):
            path = ROOT / item[f"{kind}_path"]
            if sha(path) != item[f"{kind}_sha256"] or path.stat().st_size != item[f"{kind}_bytes"]:
                raise ValueError(f"State file changed: {path}")
        if sha(ROOT / item["raw_path"]) != item["raw_sha256"]:
            raise ValueError(f"Capture raw changed: {item['case_id']} {item['arm']}")
        result[item["case_id"], item["arm"]] = item
    return result


def comparison_paths(case_id, destination, stage):
    stem = f"{case_id}-{destination}-{stage}"
    base = HERE / "comparisons"
    return base / f"{stem}.json", base / f"{stem}.meta.json"


def checked_comparison(case, destination, specs, frozen, by_doc, refs):
    raw, meta = comparison_paths(case["case_id"], destination, specs[0][3])
    model = ROOT / frozen["models"][destination]["model_path"]
    ids = ROOT / by_doc[case["document_id"]]["ids_path"]
    sources = {label: sha(path) for label, _, path, _ in specs}
    if meta.exists():
        old = json.loads(meta.read_text())
        if (old["freeze_sha256"] != sha(HERE / "FRESH_STATE_FREEZE.json") or
                old["raw_sha256"] != sha(raw) or old["source_sha256"] != sources):
            raise ValueError(f"Comparison artifact changed: {raw}")
    else:
        temp = raw.with_suffix(".partial.json")
        cmd = [str(CONTROLLED), str(model), str(ids), str(case["target_position"]),
               str(case["history_length"]), "96", str(temp)]
        for label, kind, path, _ in specs:
            cmd.extend((label, kind, str(path)))
        with raw.with_suffix(".log").open("w") as log:
            result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=log)
        if result.returncode:
            raise RuntimeError(f"State comparison failed: {raw.with_suffix('.log')}")
        row = json.loads(temp.read_text())
        validate_comparison(row, case, destination, specs, model, ids,
                            refs[destination][case["case_id"]], frozen)
        os.replace(temp, raw)
        write_json(meta, {"freeze_sha256": sha(HERE / "FRESH_STATE_FREEZE.json"),
                          "raw_sha256": sha(raw), "source_sha256": sources})
    row = json.loads(raw.read_text())
    metrics = validate_comparison(row, case, destination, specs, model, ids,
                                  refs[destination][case["case_id"]], frozen)
    return {"case_id": case["case_id"], "destination": destination,
            "stage": specs[0][3], "raw_path": str(raw.relative_to(ROOT)),
            "raw_sha256": sha(raw), "metrics": metrics}


def validate_comparison(row, case, destination, specs, model, ids, reference_nll, frozen):
    if (row["model_path"], row["ids_path"], row["p"], row["h"], row["m"]) != (
            str(model), str(ids), case["target_position"], case["history_length"], 96):
        raise ValueError(f"Comparison identity differs: {case['case_id']} {destination}")
    if len(row["conditions"]) != len(specs):
        raise ValueError("Comparison condition count differs")
    direct_delta = maxdiff(row["direct_nll"], reference_nll)
    if direct_delta > frozen["gates"]["identity_max_token_nll_delta"]:
        raise ValueError(f"Comparison direct differs: {case['case_id']} {destination}: {direct_delta}")
    metrics = []
    for condition, (label, kind, path, _) in zip(row["conditions"], specs):
        if (condition["label"], condition["kind"], condition["state_path"]) != (label, kind, str(path)):
            raise ValueError(f"Comparison source differs: {case['case_id']} {label}")
        aba = maxdiff(row["direct_nll"], condition["restored_nll"])
        if aba > frozen["gates"]["identity_max_token_nll_delta"]:
            raise ValueError(f"A-B-A restore failed: {case['case_id']} {label}: {aba}")
        if condition["imported_nll"][0] != row["direct_nll"][0]:
            raise ValueError(f"Imported state affected target 0: {case['case_id']} {label}")
        delta = [x-y for x, y in zip(condition["imported_nll"][1:], row["direct_nll"][1:])]
        metrics.append({"label": label, "kind": kind, "source_state_sha256": sha(path),
                        "aba_max_token_delta": aba,
                        "mean_excess_nll": statistics.mean(delta),
                        "mean_absolute_nll_delta": statistics.mean(abs(x) for x in delta)})
    return metrics


def positive(frozen, by_doc):
    captures = verify_captures(frozen)
    refs = reference_rows(frozen)
    names = [d["id"] for d in frozen["documents"]]
    records = []
    for case in frozen["cases"]:
        current = names.index(case["document_id"])
        next_doc = names[(current + 1) % len(names)]
        source_id = f"{next_doc}-p{case['target_position']}-h{case['history_length']}"
        source = ROOT / captures[source_id, "original"]["recurrent_path"]
        spec = [("unrelated_recurrent", "partial", source, "positive")]
        item = checked_comparison(case, "original", spec, frozen, by_doc, refs)
        sensitivity = item["metrics"][0]["mean_absolute_nll_delta"]
        if sensitivity < frozen["gates"]["positive_mean_absolute_token_nll_delta_min"]:
            raise ValueError(f"Insensitive unrelated-history control: {case['case_id']}: {sensitivity}")
        item["source_document"] = next_doc
        records.append(item)
        print(f"Positive {case['case_id']}: mean absolute delta {sensitivity:.6f}", flush=True)
    write_json(HERE / "positive_gate.json", {"freeze_sha256": sha(HERE / "FRESH_STATE_FREEZE.json"),
        "passed": True, "comparisons": records})


def treatment(frozen, by_doc):
    positive_gate = verify_gate(HERE / "positive_gate.json", frozen)
    if len(positive_gate["comparisons"]) != 24:
        raise ValueError("Positive control gate incomplete")
    captures = verify_captures(frozen)
    refs = reference_rows(frozen)
    records = []
    for case in frozen["cases"]:
        case_id = case["case_id"]
        for destination in ("original", "alpha", "mlp"):
            sources = ("alpha", "mlp") if destination == "original" else ("original",)
            specs = []
            for source_arm in sources:
                for kind, suffix in (("partial", "recurrent"), ("full", "full")):
                    path = ROOT / captures[case_id, source_arm][f"{suffix}_path"]
                    specs.append((f"{source_arm}_{suffix}", kind, path, "treatment"))
            item = checked_comparison(case, destination, specs, frozen, by_doc, refs)
            records.append(item)
            print(f"Compared {case_id} {destination}: " + ", ".join(
                f"{x['label']}={x['mean_excess_nll']:+.6f}" for x in item["metrics"]), flush=True)
    write_json(HERE / "treatment_gate.json", {"freeze_sha256": sha(HERE / "FRESH_STATE_FREEZE.json"),
        "passed": True, "comparisons": records})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("reference", "capture", "positive", "treatment"))
    args = parser.parse_args()
    frozen, by_doc = load_freeze()
    if args.phase == "reference":
        reference(frozen)
    elif args.phase == "capture":
        capture(frozen, by_doc)
    elif args.phase == "positive":
        positive(frozen, by_doc)
    else:
        treatment(frozen, by_doc)


if __name__ == "__main__":
    main()

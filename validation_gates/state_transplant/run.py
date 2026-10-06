"""Run full/partial same-model identity checks before any cross-model state import."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
EXECUTABLE = HERE / "state_gate"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def maxdiff(left, right):
    if len(left) != len(right):
        raise ValueError("NLL vector lengths differ")
    return max(abs(a - b) for a, b in zip(left, right))


def execute(mode, model, ids, p, h, m, full, partial, output):
    if output.exists():
        row = json.loads(output.read_text())
        if row["mode"] != mode or row["model_path"] != str(model) or row["ids_path"] != str(ids) or (
                row["p"], row["h"], row["m"]) != (p, h, m):
            raise ValueError(f"Stale saved result: {output}")
        if mode == "capture" and (not full.exists() or not partial.exists()):
            raise ValueError(f"State file missing for saved result: {output}")
        return row
    temporary = output.with_suffix(".partial.json")
    log = output.with_suffix(".log")
    cmd = [str(EXECUTABLE), mode, str(model), str(ids), str(p), str(h), str(m),
           str(full), str(partial), str(temporary)]
    with log.open("w") as stream:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=stream)
    if proc.returncode:
        raise RuntimeError(f"State gate failed; see {log}")
    row = json.loads(temporary.read_text())
    if row["mode"] != mode or row["model_path"] != str(model) or row["ids_path"] != str(ids) or (
            row["p"], row["h"], row["m"]) != (p, h, m):
        raise ValueError(f"State-gate output identity mismatch: {output}")
    temporary.rename(output)
    return row


def main():
    frozen = json.loads((HERE / "STATE_GATE_FREEZE.json").read_text())
    if sha(EXECUTABLE) != frozen["state_gate_binary_sha256"] or sha(HERE / "state_gate.cpp") != frozen["state_gate_source_sha256"]:
        raise ValueError("State-gate binary/source changed")
    for relative, digest in frozen["linked_cuda_library_sha256"].items():
        if sha(ROOT / relative) != digest:
            raise ValueError(f"Linked CUDA library changed: {relative}")
    if sha(ROOT / "analysis/bonsai2/dose_calibration/DOSE_FREEZE.json") != frozen["dose_freeze_sha256"]:
        raise ValueError("Dose freeze changed")
    original = ROOT / frozen["original_model_path"]
    variant = ROOT / frozen["variant"]["model_path"]
    if sha(original) != frozen["original_model_sha256"] or sha(variant) != frozen["variant"]["model_sha256"]:
        raise ValueError("Checkpoint changed")
    doc = frozen["document"]
    ids = ROOT / doc["ids_path"]
    if sha(ids) != doc["ids_file_sha256"]:
        raise ValueError("Document IDs changed")
    prior_original = {r["case_id"]: r for r in map(json.loads,
        (ROOT / "analysis/bonsai2/scorer/baseline_results.jsonl").read_text().splitlines())}
    prior_variant = {r["case_id"]: r for r in map(json.loads,
        (ROOT / f"analysis/bonsai2/heldout_nll/{frozen['variant']['name']}.jsonl").read_text().splitlines())}
    records = []
    for case in frozen["cases"]:
        p, h, m = case["target_position"], case["history_length"], case["target_length"]
        for arm, model, prior in (("original", original, prior_original),
                                  ("alpha", variant, prior_variant)):
            stem = f"{arm}-h{h}"
            full = HERE / f"{stem}-full.bin"
            partial = HERE / f"{stem}-recurrent.bin"
            row = execute("capture", model, ids, p, h, m, full, partial, HERE / f"{stem}.json")
            if len(row["direct_nll"]) != m:
                raise ValueError(f"Target count mismatch: {stem}")
            full_diff = maxdiff(row["direct_nll"], row["full_roundtrip_nll"])
            partial_diff = maxdiff(row["direct_nll"], row["partial_roundtrip_nll"])
            reference = prior[case["case_id"]].get("gold_nll_nats", prior[case["case_id"]].get("nll"))
            baseline_diff = maxdiff(row["direct_nll"], reference)
            item = {"arm": arm, "history": h, "full_roundtrip_max_token_delta": full_diff,
                    "partial_overlay_max_token_delta": partial_diff,
                    "prior_scorer_max_token_delta": baseline_diff,
                    "full_state_bytes": full.stat().st_size, "full_state_sha256": sha(full),
                    "recurrent_state_bytes": partial.stat().st_size, "recurrent_state_sha256": sha(partial)}
            records.append(item)
            print(json.dumps(item), flush=True)
            if max(full_diff, partial_diff, baseline_diff) > 0.0001:
                (HERE / "gate_summary.json").write_text(json.dumps({"passed": False, "captures": records,
                    "decision": "Stop before cross-model import: identity or scorer alignment failed"}, indent=2) + "\n")
                raise ValueError(f"Identity gate failed: {stem}")
    cross = []
    for case in frozen["cases"]:
        p, h, m = case["target_position"], case["history_length"], case["target_length"]
        source_partial = HERE / f"alpha-h{h}-recurrent.bin"
        row = execute("transplant", original, ids, p, h, m,
                      HERE / f"original-h{h}-full.bin", source_partial,
                      HERE / f"alpha-state-into-original-h{h}.json")
        original_row = json.loads((HERE / f"original-h{h}.json").read_text())
        destination_delta = maxdiff(row["direct_nll"], original_row["direct_nll"])
        if destination_delta > 0.0001:
            raise ValueError(f"Destination direct rerun differed: h={h}")
        delta = [a - b for a, b in zip(row["transplanted_nll"], row["direct_nll"])]
        item = {"history": h, "destination_direct_rerun_max_token_delta": destination_delta,
                "transplanted_minus_original_mean_nll": sum(delta) / len(delta),
                "transplanted_minus_original_max_abs_token_nll": max(abs(x) for x in delta),
                "source_recurrent_state_sha256": sha(source_partial)}
        cross.append(item)
        print(json.dumps(item), flush=True)
    summary = {"passed": True, "freeze_sha256": sha(HERE / "STATE_GATE_FREEZE.json"),
               "captures": records, "cross_model_imports": cross,
               "decision": "same-model identity passed; cross-model import is feasible but exploratory, not a mechanism estimate"}
    (HERE / "gate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

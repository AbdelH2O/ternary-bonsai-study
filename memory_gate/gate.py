"""Freeze, run, and analyze the original-model 1K/12K recurrent-memory gate."""

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import struct
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PARENT = ROOT / "analysis/bonsai2/fresh_state"
CONTROLLED = ROOT / "analysis/bonsai2/controlled_state/controlled_state"
TCRIT5 = 2.5705818366147395


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def maxdiff(a, b):
    if len(a) != len(b):
        raise ValueError("NLL vector lengths differ")
    return max(abs(x - y) for x, y in zip(a, b))


def header(path):
    with path.open("rb") as stream:
        raw = stream.read(20)
    if len(raw) != 20:
        raise ValueError(f"Short state header: {path}")
    fields = struct.unpack("<IIIII", raw)
    if fields[:3] != (0xAF143CD8, 0, 1) or fields[4] != 0:
        raise ValueError(f"Unexpected pinned-Prism state header: {path}: {fields}")
    return fields


def parent_records():
    prior = json.loads((PARENT / "FRESH_STATE_FREEZE.json").read_text())
    cap = json.loads((PARENT / "capture_gate.json").read_text())
    refgate = json.loads((PARENT / "reference_gate.json").read_text())
    if not cap["passed"] or not refgate["passed"]:
        raise ValueError("Prior capture or reference gate failed")
    parent_sha = digest(PARENT / "FRESH_STATE_FREEZE.json")
    if cap["freeze_sha256"] != parent_sha or refgate["freeze_sha256"] != parent_sha:
        raise ValueError("Prior capture/reference freeze mismatch")
    by_capture = {(x["case_id"], x["arm"]): x for x in cap["captures"]}
    by_doc = {d["id"]: d for d in prior["documents"]}
    by_case = {c["case_id"]: c for c in prior["cases"]}
    refs = {r["case_id"]: r for r in map(json.loads, (PARENT / "reference-original.jsonl").read_text().splitlines())}
    cases = []
    for doc in prior["documents"]:
        for p in prior["positions"]:
            short = by_case[f"{doc['id']}-p{p}-h1024"]
            long = by_case[f"{doc['id']}-p{p}-h12288"]
            if short["target_ids_sha256"] != long["target_ids_sha256"] or short["target_length"] != 96:
                raise ValueError("Nonidentical nested target IDs")
            s = by_capture[short["case_id"], "original"]
            l = by_capture[long["case_id"], "original"]
            if header(ROOT / s["recurrent_path"])[3] != 1023 or header(ROOT / l["recurrent_path"])[3] != 12287:
                raise ValueError("Unexpected recurrent-cell positions")
            cases.append({"case_id": long["case_id"], "document_id": doc["id"], "position": p,
                          "ids_path": doc["ids_path"], "ids_sha256": doc["ids_file_sha256"],
                          "target_ids_sha256": long["target_ids_sha256"],
                          "short_state_path": s["recurrent_path"], "short_state_sha256": s["recurrent_sha256"],
                          "long_state_path": l["recurrent_path"], "long_state_sha256": l["recurrent_sha256"],
                          "reference_nll": refs[long["case_id"]]["nll"]})
    if len(cases) != 12:
        raise ValueError("Expected twelve paired spans")
    return prior, cases


def freeze():
    target = HERE / "MEMORY_GATE_FREEZE.json"
    if target.exists():
        raise ValueError("Freeze already exists; it must not be replaced")
    prior, cases = parent_records()
    paths = [HERE / "DESIGN.md", HERE / "gate.py", PARENT / "FRESH_STATE_FREEZE.json",
             PARENT / "capture_gate.json", PARENT / "reference-original.jsonl",
             CONTROLLED, ROOT / "analysis/bonsai2/controlled_state/controlled_state.cpp"]
    paths += [ROOT / p for p in prior["linked_cuda_library_sha256"]]
    paths += [ROOT / prior["models"]["original"]["model_path"]]
    paths += [ROOT / c["ids_path"] for c in cases]
    paths += [ROOT / c[k] for c in cases for k in ("short_state_path", "long_state_path")]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in dict.fromkeys(paths)}
    for case in cases:
        for key, sha_key in (("ids_path", "ids_sha256"), ("short_state_path", "short_state_sha256"),
                             ("long_state_path", "long_state_sha256")):
            if hashes[case[key]] != case[sha_key]:
                raise ValueError(f"Parent artifact changed: {case['case_id']} {key}")
    for path, expected in prior["linked_cuda_library_sha256"].items():
        if hashes[path] != expected:
            raise ValueError(f"Pinned library changed: {path}")
    if hashes[prior["models"]["original"]["model_path"]] != prior["models"]["original"]["model_sha256"]:
        raise ValueError("Original checkpoint changed")
    write_json(target, {"purpose": "same-model nested recurrent-state memory gate", "runtime": prior["runtime_release"],
                        "parent_freeze_sha256": hashes[str((PARENT / "FRESH_STATE_FREEZE.json").relative_to(ROOT))],
                        "hashes": hashes, "model_path": prior["models"]["original"]["model_path"],
                        "cases": cases, "thresholds": {"identity_max_token_nll_delta": 0.0001,
                            "position_invariance_max_token_nll_delta": 0.0001,
                            "primary_window": [1, 16], "secondary_window": [1, 95],
                            "meaningful_mean_nats_per_token": 0.005,
                            "support_min_positive_documents": 5, "t_critical_df5_95_two_sided": TCRIT5}})
    print(f"Frozen {len(cases)} paired spans: {target}", flush=True)


def load_freeze():
    path = HERE / "MEMORY_GATE_FREEZE.json"
    frozen = json.loads(path.read_text())
    for relative, expected in frozen["hashes"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Frozen input changed: {relative}")
    if len(frozen["cases"]) != 12:
        raise ValueError("Incomplete freeze")
    return frozen


def rebase(source, dest):
    if header(source)[3] != 1023:
        raise ValueError("Expected 1K source cell position")
    if dest.exists():
        if header(dest)[3] != 12287 or dest.stat().st_size != source.stat().st_size:
            raise ValueError("Existing rebased state differs")
        verify_rebase(source, dest)
        return
    dest.parent.mkdir(exist_ok=True)
    temp = dest.with_suffix(".partial.bin")
    try:
        subprocess.run(["cp", "--reflink=always", str(source), str(temp)], check=True)
    except subprocess.CalledProcessError:
        shutil.copyfile(source, temp)
    with temp.open("r+b") as stream:
        stream.seek(12)
        stream.write(struct.pack("<i", 12287))
    if header(temp)[3] != 12287 or temp.stat().st_size != source.stat().st_size:
        raise ValueError("Rebased state failed CPU validation")
    verify_rebase(source, temp)
    os.replace(temp, dest)


def verify_rebase(source, dest):
    with source.open("rb") as left, dest.open("rb") as right:
        a = left.read(20)
        b = right.read(20)
        if a[:12] != b[:12] or a[16:] != b[16:] or b[12:16] != struct.pack("<i", 12287):
            raise ValueError("Rebased header changed outside cell position")
        while True:
            a = left.read(1024 * 1024)
            b = right.read(1024 * 1024)
            if a != b:
                raise ValueError("Rebased recurrent tensor bytes differ")
            if not a:
                return


def run():
    frozen = load_freeze()
    results = HERE / "results"
    results.mkdir(exist_ok=True)
    freeze_sha = digest(HERE / "MEMORY_GATE_FREEZE.json")
    for case in frozen["cases"]:
        case_id = case["case_id"]
        short = ROOT / case["short_state_path"]
        long = ROOT / case["long_state_path"]
        rebased = HERE / "rebased" / f"{case_id}-short-pos12287.bin"
        rebase(short, rebased)
        raw = results / f"{case_id}.json"
        meta = results / f"{case_id}.meta.json"
        logpath = results / f"{case_id}.log"
        sources = {"long_self": digest(long), "short_raw": digest(short), "short_rebased": digest(rebased)}
        if meta.exists():
            old = json.loads(meta.read_text())
            if old["freeze_sha256"] != freeze_sha or old["sources"] != sources or old["raw_sha256"] != digest(raw):
                raise ValueError(f"Completed comparison changed: {case_id}")
        else:
            temp = raw.with_suffix(".partial.json")
            cmd = [str(CONTROLLED), str(ROOT / frozen["model_path"]), str(ROOT / case["ids_path"]),
                   str(case["position"]), "12288", "96", str(temp),
                   "long_self", "partial", str(long), "short_raw", "partial", str(short),
                   "short_rebased", "partial", str(rebased)]
            with logpath.open("w") as log:
                proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=log)
            if proc.returncode:
                raise RuntimeError(f"Comparison failed; inspect {logpath}")
            validate(json.loads(temp.read_text()), case, frozen, sources, logpath)
            os.replace(temp, raw)
            write_json(meta, {"freeze_sha256": freeze_sha, "sources": sources,
                              "raw_sha256": digest(raw), "rebased_state_path": str(rebased.relative_to(ROOT))})
        metrics = validate(json.loads(raw.read_text()), case, frozen, sources, logpath)
        print(f"{case_id}: early {metrics['early_mean']:+.6f}; position max {metrics['position_max']:.8f}", flush=True)


def validate(row, case, frozen, sources, logpath):
    if (row["model_path"], row["ids_path"], row["p"], row["h"], row["m"]) != (
            str(ROOT / frozen["model_path"]), str(ROOT / case["ids_path"]), case["position"], 12288, 96):
        raise ValueError("Comparison identity differs")
    by_label = {x["label"]: x for x in row["conditions"]}
    if len(row["conditions"]) != 3 or set(by_label) != set(sources):
        raise ValueError("Condition set differs")
    direct = row["direct_nll"]
    limit = frozen["thresholds"]["identity_max_token_nll_delta"]
    if len(direct) != 96 or maxdiff(direct, case["reference_nll"]) > limit:
        raise ValueError("Direct scorer identity failed")
    for label, condition in by_label.items():
        if condition["kind"] != "partial" or digest(Path(condition["state_path"])) != sources[label]:
            raise ValueError(f"Source identity failed: {label}")
        if condition["imported_nll"][0] != direct[0] or maxdiff(condition["restored_nll"], direct) > limit:
            raise ValueError(f"Target 0 or A-B-A restore failed: {label}")
    if maxdiff(by_label["long_self"]["imported_nll"], direct) > limit:
        raise ValueError("Long-state self-import identity failed")
    pos_max = maxdiff(by_label["short_raw"]["imported_nll"][1:],
                      by_label["short_rebased"]["imported_nll"][1:])
    if pos_max > frozen["thresholds"]["position_invariance_max_token_nll_delta"]:
        raise ValueError(f"Position-rebase invariance failed: {pos_max}")
    log = logpath.read_text()
    warnings = [line for line in log.splitlines() if "non-consecutive token position" in line]
    expected = f"non-consecutive token position {12288 + 94} after 1023"
    if len(warnings) != 1 or expected not in warnings[0]:
        raise ValueError(f"Unexpected cross-history warning count/content: {warnings}")
    delta = [a - b for a, b in zip(by_label["short_rebased"]["imported_nll"],
                                    by_label["long_self"]["imported_nll"])]
    return {"early_mean": statistics.mean(delta[1:17]), "all_mean": statistics.mean(delta[1:96]),
            "position_max": pos_max, "self_max": maxdiff(by_label["long_self"]["imported_nll"], direct)}


def interval(values):
    mean = statistics.mean(values)
    half = TCRIT5 * statistics.stdev(values) / math.sqrt(6)
    return [mean - half, mean + half]


def analyze():
    frozen = load_freeze()
    values = {}
    evidence = []
    for case in frozen["cases"]:
        cid = case["case_id"]
        raw = HERE / "results" / f"{cid}.json"
        meta = json.loads((HERE / "results" / f"{cid}.meta.json").read_text())
        if meta["freeze_sha256"] != digest(HERE / "MEMORY_GATE_FREEZE.json") or meta["raw_sha256"] != digest(raw):
            raise ValueError(f"Raw result changed: {cid}")
        row = json.loads(raw.read_text())
        metrics = validate(row, case, frozen, meta["sources"], HERE / "results" / f"{cid}.log")
        values.setdefault(case["document_id"], []).append(metrics)
        evidence.append({"case_id": cid, **metrics, "raw_sha256": digest(raw), "rebased_sha256": meta["sources"]["short_rebased"]})
    if len(values) != 6 or any(len(x) != 2 for x in values.values()):
        raise ValueError("Incomplete document clusters")
    docs = [{"document_id": doc, "early_mean": statistics.mean(x["early_mean"] for x in rows),
             "all_mean": statistics.mean(x["all_mean"] for x in rows)} for doc, rows in values.items()]
    primary = [d["early_mean"] for d in docs]
    ci = interval(primary)
    mean = statistics.mean(primary)
    floor = frozen["thresholds"]["meaningful_mean_nats_per_token"]
    positives = sum(x > 0 for x in primary)
    if mean >= floor and ci[0] > 0 and positives >= frozen["thresholds"]["support_min_positive_documents"]:
        decision = "supports_recurrent_benefit_on_these_spans"
    elif ci[1] < floor:
        decision = "excludes_0p005_recurrent_benefit_on_these_spans"
    else:
        decision = "unresolved"
    secondary = [d["all_mean"] for d in docs]
    summary = {"freeze_sha256": digest(HERE / "MEMORY_GATE_FREEZE.json"), "technical_gates_passed": True,
               "decision": decision, "primary_early_mean": mean, "primary_early_t95ci": ci,
               "primary_positive_documents": positives, "secondary_all_mean": statistics.mean(secondary),
               "secondary_all_t95ci": interval(secondary), "documents": docs, "cases": evidence}
    write_json(HERE / "analysis_summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("decision", "primary_early_mean", "primary_early_t95ci",
                                                  "primary_positive_documents", "secondary_all_mean", "secondary_all_t95ci")}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze", "run", "analyze"))
    phase = parser.parse_args().phase
    if phase == "freeze":
        freeze()
    elif phase == "run":
        run()
    else:
        analyze()


if __name__ == "__main__":
    main()

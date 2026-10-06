"""Frozen R/S decomposition and pre-question same-model import assay."""

import argparse
import hashlib
import json
import math
import os
import statistics
import struct
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "analysis/bonsai2/remote_state_gate"))
import gate as old  # noqa: E402

OLD = ROOT / "analysis/bonsai2/remote_state_gate"
FREEZE = HERE / "COMPONENT_FREEZE.json"
TCRIT7 = 2.3646242510102993
R_COUNT = S_COUNT = 48
R_BYTES = 122880
S_BYTES = 3145728


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def layout(blob):
    if len(blob) != 28 + R_COUNT * (12 + R_BYTES) + S_COUNT * (12 + S_BYTES):
        raise ValueError("Wrong recurrent-state byte length")
    if struct.unpack_from("<IIIII", blob)[:3] != (0xAF143CD8, 0, 1):
        raise ValueError("Wrong recurrent-state header")
    if struct.unpack_from("<II", blob, 20) != (0, 64):
        raise ValueError("Wrong s_trans or layer count")
    off = 28
    result = {"R": [], "S": []}
    for kind, count, size in (("R", R_COUNT, R_BYTES), ("S", S_COUNT, S_BYTES)):
        for _ in range(count):
            if struct.unpack_from("<iQ", blob, off) != (0, size):
                raise ValueError(f"Wrong {kind} tensor type or row size")
            off += 12
            result[kind].append((off, off + size))
            off += size
    if off != len(blob):
        raise ValueError("Unconsumed state bytes")
    return result


def splice(own, other, kind, ranges):
    result = bytearray(own)
    for a, b in ranges[kind]:
        result[a:b] = other[a:b]
    return result


def checked_old():
    frozen = json.loads((OLD / "REMOTE_STATE_FREEZE.json").read_text())
    gate = json.loads((OLD / "capture_gate.json").read_text())
    if not gate["passed"] or gate["freeze_sha256"] != sha(OLD / "REMOTE_STATE_FREEZE.json"):
        raise ValueError("Old capture gate changed or failed")
    captures = {c["prompt_id"]: c for c in gate["captures"]}
    if len(captures) != 16 or len(frozen["cases"]) != 32:
        raise ValueError("Incomplete old gate")
    return frozen, captures


def prepare():
    if FREEZE.exists():
        raise ValueError("Experiment is already frozen")
    frozen, captures = checked_old()
    rows = {r["id"]: r for r in map(json.loads, old.PROMPTS.read_text().splitlines())}
    template = old.chat_template(old.MODEL)
    hybrid_dir = HERE / "hybrids"
    hybrid_dir.mkdir(exist_ok=True)
    boundaries = {}
    hybrid_hashes = {}
    for prompt in frozen["prompts"]:
        pid = prompt["prompt_id"]
        counterpart = next(p["prompt_id"] for p in frozen["prompts"]
                           if p["registry_id"] == prompt["registry_id"] and p["swap"] != prompt["swap"])
        own_path = ROOT / captures[pid]["recurrent_path"]
        other_path = ROOT / captures[counterpart]["recurrent_path"]
        if sha(own_path) != captures[pid]["recurrent_sha256"] or sha(other_path) != captures[counterpart]["recurrent_sha256"]:
            raise ValueError("Old recurrent capture changed")
        own_bytes, other_bytes = own_path.read_bytes(), other_path.read_bytes()
        own_layout, other_layout = layout(own_bytes), layout(other_bytes)
        if own_layout != other_layout or own_bytes[:28] != other_bytes[:28]:
            raise ValueError("Incompatible counterpart state layouts or positions")
        r_only = splice(own_bytes, other_bytes, "R", own_layout)
        s_only = splice(own_bytes, other_bytes, "S", own_layout)
        rebuilt = splice(r_only, other_bytes, "S", own_layout)
        if rebuilt != other_bytes or splice(own_bytes, own_bytes, "R", own_layout) != own_bytes:
            raise ValueError("Byte-exact component reconstruction failed")
        for label, data in (("r_only", r_only), ("s_only", s_only)):
            path = hybrid_dir / f"{pid}.{label}.bin"
            if path.exists() and path.read_bytes() != data:
                raise ValueError(f"Existing hybrid differs: {path}")
            if not path.exists():
                tmp = path.with_suffix(".partial.bin")
                tmp.write_bytes(data)
                os.replace(tmp, path)
            hybrid_hashes[str(path.relative_to(ROOT))] = sha(path)
        chat = old.render_chat(template, rows[pid]["prompt"])
        cut = chat.rfind("\nQuestion:")
        if cut < 0 or chat.find("\nQuestion:", cut + 1) >= 0:
            raise ValueError("Question boundary missing or ambiguous")
        before = old.tokenize(chat[:cut + 1])
        full_prefix = [int(v) for v in (ROOT / next(c["full_ids_file"] for c in frozen["cases"]
                                             if c["prompt_id"] == pid and c["candidate_role"] == "gold")).read_text().splitlines()]
        if before != full_prefix[:len(before)] or not 10 <= len(full_prefix) - len(before) - 9 <= 100:
            raise ValueError("Question boundary not token-aligned")
        boundaries[pid] = {"p": len(before), "answer_p": prompt["prefix_token_count"],
                           "suffix_target_count": prompt["prefix_token_count"] - len(before) + 9,
                           "before_ids_sha256": old.ids_sha(before)}
    for reg in frozen["registries"]:
        a, b = (boundaries[f"rv2-{reg:02d}-q1-long-far-{x}"]["p"] for x in ("base", "swap"))
        if a != b:
            raise ValueError(f"Pre-question positions differ for registry {reg}")
    write_json(HERE / "PREPARED.json", {"old_freeze_sha256": sha(OLD / "REMOTE_STATE_FREEZE.json"),
                                        "boundaries": boundaries, "hybrid_hashes": hybrid_hashes})
    print(f"Prepared {len(hybrid_hashes)} byte-checked hybrids and {len(boundaries)} boundaries", flush=True)


def freeze():
    if FREEZE.exists():
        raise ValueError("Freeze already exists")
    frozen, captures = checked_old()
    prepared = json.loads((HERE / "PREPARED.json").read_text())
    if prepared["old_freeze_sha256"] != sha(OLD / "REMOTE_STATE_FREEZE.json"):
        raise ValueError("Old freeze changed after prepare")
    paths = [HERE / "DESIGN.md", HERE / "experiment.py", HERE / "PREPARED.json",
             OLD / "REMOTE_STATE_FREEZE.json", OLD / "capture_gate.json",
             OLD / "analysis_summary.json", old.PROMPTS, old.PROMPT_MANIFEST,
             old.MODEL, old.CONTROLLED, old.CAPTURE]
    paths += [ROOT / c["full_ids_file"] for c in frozen["cases"]]
    paths += [ROOT / x["recurrent_path"] for x in captures.values()]
    paths += [OLD / "comparisons" / f"{c['case_id']}.json" for c in frozen["cases"]]
    paths += [ROOT / p for p in prepared["hybrid_hashes"]]
    paths += [ROOT / p for p in json.loads(old.PARENT.read_text())["linked_cuda_library_sha256"]]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in dict.fromkeys(paths)}
    for path, expected in prepared["hybrid_hashes"].items():
        if hashes[path] != expected:
            raise ValueError(f"Prepared hybrid changed: {path}")
    write_json(FREEZE, {"purpose": "exploratory R/S split plus pre-question state import",
                        "runtime_release": frozen["runtime_release"], "registries": frozen["registries"],
                        "hashes": hashes, "boundaries": prepared["boundaries"],
                        "hybrid_hashes": prepared["hybrid_hashes"],
                        "identity_tolerance_nats_per_token": 0.0001,
                        "prequestion_effect_floor_nats_per_answer": 0.1,
                        "t_critical_df7": TCRIT7})
    print(f"Frozen {len(hashes)} input hashes", flush=True)


def load_freeze():
    f = json.loads(FREEZE.read_text())
    for path, digest in f["hashes"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"Frozen input changed: {path}")
    return f


def check_vector(a, b, label, tol=0.0001):
    if len(a) != len(b) or max(abs(x - y) for x, y in zip(a, b)) > tol:
        raise ValueError(f"NLL vector mismatch: {label}")


def run(cmd, log):
    with log.open("w") as stream:
        status = subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=stream)
    if status.returncode:
        raise RuntimeError(f"Command failed ({status.returncode}); inspect {log}")
    if "non-consecutive token position" in log.read_text():
        raise ValueError(f"Position warning in {log}")


def captures():
    f = load_freeze()
    old_f, _ = checked_old()
    dest = HERE / "prequestion_captures"
    dest.mkdir(exist_ok=True)
    for prompt in old_f["prompts"]:
        pid = prompt["prompt_id"]
        gold = next(c for c in old_f["cases"] if c["prompt_id"] == pid and c["candidate_role"] == "gold")
        b = f["boundaries"][pid]
        raw, rec, meta = dest / f"{pid}.json", dest / f"{pid}.recurrent.bin", dest / f"{pid}.meta.json"
        if not meta.exists():
            temp_full = dest / f"{pid}.partial.full.bin"
            temp_rec = dest / f"{pid}.partial.recurrent.bin"
            temp_raw = dest / f"{pid}.partial.json"
            run([str(old.CAPTURE), "capture", str(old.MODEL), str(ROOT / gold["full_ids_file"]),
                 str(b["p"]), str(b["p"]), str(b["suffix_target_count"]),
                 str(temp_full), str(temp_rec), str(temp_raw)], dest / f"{pid}.log")
            data = json.loads(temp_raw.read_text())
            check_vector(data["direct_nll"], data["full_roundtrip_nll"], f"{pid} full")
            check_vector(data["direct_nll"], data["partial_roundtrip_nll"], f"{pid} recurrent")
            if old.state_pos(temp_rec) != b["p"] - 1:
                raise ValueError("Pre-question cell position wrong")
            os.replace(temp_rec, rec)
            os.replace(temp_raw, raw)
            temp_full.unlink()
            write_json(meta, {"freeze_sha256": sha(FREEZE), "raw_sha256": sha(raw),
                              "recurrent_sha256": sha(rec)})
        info = json.loads(meta.read_text())
        if info["freeze_sha256"] != sha(FREEZE) or info["raw_sha256"] != sha(raw) or info["recurrent_sha256"] != sha(rec):
            raise ValueError(f"Pre-question capture changed: {pid}")
        data = json.loads(raw.read_text())
        check_vector(data["direct_nll"], data["full_roundtrip_nll"], f"{pid} full")
        check_vector(data["direct_nll"], data["partial_roundtrip_nll"], f"{pid} recurrent")
        print(f"Captured before question: {pid}", flush=True)


def compare(which):
    f = load_freeze()
    old_f, old_captures = checked_old()
    outdir = HERE / ("postquestion_comparisons" if which == "post" else "prequestion_comparisons")
    outdir.mkdir(exist_ok=True)
    for case in old_f["cases"]:
        pid = case["prompt_id"]
        counterpart = next(p["prompt_id"] for p in old_f["prompts"]
                           if p["registry_id"] == case["registry_id"] and p["swap"] != case["swap"])
        if which == "post":
            p = case["prefix_token_count"]
            own = ROOT / old_captures[pid]["recurrent_path"]
            cross = ROOT / old_captures[counterpart]["recurrent_path"]
            sources = {"own": own, "r_only": HERE / "hybrids" / f"{pid}.r_only.bin",
                       "s_only": HERE / "hybrids" / f"{pid}.s_only.bin", "combined": cross}
            m = 9
        else:
            p = f["boundaries"][pid]["p"]
            own = HERE / "prequestion_captures" / f"{pid}.recurrent.bin"
            cross = HERE / "prequestion_captures" / f"{counterpart}.recurrent.bin"
            sources = {"own": own, "combined": cross}
            m = f["boundaries"][pid]["suffix_target_count"]
            for source_pid, source in ((pid, own), (counterpart, cross)):
                meta = json.loads((HERE / "prequestion_captures" / f"{source_pid}.meta.json").read_text())
                if meta["freeze_sha256"] != sha(FREEZE) or meta["recurrent_sha256"] != sha(source):
                    raise ValueError("Pre-question source capture changed")
        if any(old.state_pos(path) != p - 1 for path in sources.values()):
            raise ValueError(f"Source positions differ: {case['case_id']}")
        raw = outdir / f"{case['case_id']}.json"
        meta = outdir / f"{case['case_id']}.meta.json"
        if not meta.exists():
            temp = outdir / f"{case['case_id']}.partial.json"
            cmd = [str(old.CONTROLLED), str(old.MODEL), str(ROOT / case["full_ids_file"]),
                   str(p), str(p), str(m), str(temp)]
            for label, path in sources.items():
                cmd += [label, "partial", str(path)]
            run(cmd, outdir / f"{case['case_id']}.log")
            data = json.loads(temp.read_text())
            validate_compare(data, case, which, f, sources)
            os.replace(temp, raw)
            write_json(meta, {"freeze_sha256": sha(FREEZE), "raw_sha256": sha(raw),
                              "source_hashes": {label: sha(path) for label, path in sources.items()}})
        info = json.loads(meta.read_text())
        if info["freeze_sha256"] != sha(FREEZE) or info["raw_sha256"] != sha(raw) or info["source_hashes"] != {label: sha(path) for label, path in sources.items()}:
            raise ValueError(f"Comparison artifact changed: {case['case_id']}")
        validate_compare(json.loads(raw.read_text()), case, which, f, sources)
        print(f"Compared {which}: {case['case_id']}", flush=True)


def validate_compare(row, case, which, frozen, sources):
    p = case["prefix_token_count"] if which == "post" else frozen["boundaries"][case["prompt_id"]]["p"]
    m = 9 if which == "post" else frozen["boundaries"][case["prompt_id"]]["suffix_target_count"]
    if (row["p"], row["h"], row["m"], row["ids_path"], row["model_path"]) != (p, p, m, str(ROOT / case["full_ids_file"]), str(old.MODEL)):
        raise ValueError("Comparison profile changed")
    conditions = {x["label"]: x for x in row["conditions"]}
    if set(conditions) != set(sources) or len(conditions) != len(sources):
        raise ValueError("Wrong comparison conditions")
    check_vector(row["direct_nll"], conditions["own"]["imported_nll"], "own identity")
    for label, cond in conditions.items():
        if cond["state_path"] != str(sources[label]) or cond["kind"] != "partial":
            raise ValueError("Wrong condition source")
        check_vector(row["direct_nll"], cond["restored_nll"], f"{label} restoration")
        if cond["imported_nll"][0] != row["direct_nll"][0]:
            raise ValueError("First target NLL changed before import")
    if which == "post":
        previous = json.loads((OLD / "comparisons" / f"{case['case_id']}.json").read_text())
        old_conditions = {x["label"]: x for x in previous["conditions"]}
        check_vector(row["direct_nll"], previous["direct_nll"], "old direct")
        check_vector(conditions["combined"]["imported_nll"], old_conditions["counterfactual"]["imported_nll"], "old combined")
    else:
        if case["candidate_role"] == "gold":
            capture = json.loads((HERE / "prequestion_captures" / f"{case['prompt_id']}.json").read_text())
            check_vector(row["direct_nll"], capture["direct_nll"], "pre-question capture/direct")
    return conditions


def interval(values):
    mean = statistics.mean(values)
    half = TCRIT7 * statistics.stdev(values) / math.sqrt(len(values))
    return mean, [mean - half, mean + half]


def analyze():
    f = load_freeze()
    old_f, _ = checked_old()
    per_destination = []
    for prompt in old_f["prompts"]:
        pid = prompt["prompt_id"]
        record = {"prompt_id": pid, "registry_id": prompt["registry_id"], "swap": prompt["swap"]}
        for which in ("post", "pre"):
            directory = HERE / ("postquestion_comparisons" if which == "post" else "prequestion_comparisons")
            cases = {}
            for role in ("gold", "wrong"):
                case = next(c for c in old_f["cases"] if c["prompt_id"] == pid and c["candidate_role"] == role)
                raw = directory / f"{case['case_id']}.json"
                meta = json.loads((directory / f"{case['case_id']}.meta.json").read_text())
                if meta["freeze_sha256"] != sha(FREEZE) or meta["raw_sha256"] != sha(raw):
                    raise ValueError("Analysis input changed")
                sources = {x["label"]: Path(x["state_path"]) for x in json.loads(raw.read_text())["conditions"]}
                if meta["source_hashes"] != {label: sha(path) for label, path in sources.items()}:
                    raise ValueError("Analysis source hash changed")
                cases[role] = validate_compare(json.loads(raw.read_text()), case, which, f, sources)
            labels = ("own", "r_only", "s_only", "combined") if which == "post" else ("own", "combined")
            margins = {label: sum(cases["wrong"][label]["imported_nll"][-9:]) - sum(cases["gold"][label]["imported_nll"][-9:]) for label in labels}
            effects = {label: margins["own"] - margins[label] for label in labels if label != "own"}
            record[which] = {"margins": margins, "effects": effects}
        record["post"]["interaction"] = record["post"]["effects"]["combined"] - record["post"]["effects"]["r_only"] - record["post"]["effects"]["s_only"]
        per_destination.append(record)
    registries = []
    for reg in f["registries"]:
        rows = [x for x in per_destination if x["registry_id"] == f"registry-{reg:02d}"]
        if len(rows) != 2:
            raise ValueError("Incomplete registry pair")
        registries.append({"registry_id": f"registry-{reg:02d}",
                           "post": {key: statistics.mean(r["post"]["effects"].get(key, r["post"].get(key)) for r in rows) for key in ("r_only", "s_only", "combined", "interaction")},
                           "pre_combined": statistics.mean(r["pre"]["effects"]["combined"] for r in rows)})
    summary = {"freeze_sha256": sha(FREEZE), "technical_gates_passed": True,
               "destinations": per_destination, "registries": registries, "estimates": {}}
    for key in ("r_only", "s_only", "combined", "interaction"):
        mean, ci = interval([r["post"][key] for r in registries])
        summary["estimates"]["post_" + key] = {"mean": mean, "t95ci": ci}
    mean, ci = interval([r["pre_combined"] for r in registries])
    floor = f["prequestion_effect_floor_nats_per_answer"]
    decision = "supports_prequestion_state_influence_in_this_assay" if mean >= floor and ci[0] > 0 else "excludes_0p1_nat_prequestion_effect_in_this_assay" if ci[1] < floor else "unresolved"
    summary["estimates"]["pre_combined"] = {"mean": mean, "t95ci": ci,
                                              "positive_registries": sum(r["pre_combined"] > 0 for r in registries),
                                              "decision": decision}
    write_json(HERE / "analysis_summary.json", summary)
    print(json.dumps(summary["estimates"], indent=2), flush=True)


def inventory():
    f = load_freeze()
    files = [p for p in HERE.rglob("*") if p.is_file() and p.name != "ARTIFACTS.json" and ".partial" not in p.name]
    write_json(HERE / "ARTIFACTS.json", {"freeze_sha256": sha(FREEZE),
                                         "files": {str(p.relative_to(ROOT)): sha(p) for p in files}})
    print(f"Inventoried {len(files)} output files", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "freeze", "captures", "post", "pre", "analyze", "inventory"))
    phase = parser.parse_args().phase
    globals()[phase if phase != "post" and phase != "pre" else "compare"](phase) if phase in ("post", "pre") else globals()[phase]()


if __name__ == "__main__":
    main()

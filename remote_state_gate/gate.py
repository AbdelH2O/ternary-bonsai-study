"""Freeze and run the fresh remote-value recurrent-state sensitivity gate."""

import argparse
import ast
import hashlib
import json
import math
import os
import statistics
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "analysis/bonsai2"))
from chat_tokens import chat_template, render_chat  # noqa: E402

PROMPTS = ROOT / "analysis/bonsai2/prompts/retrieval_v2.jsonl"
PROMPT_MANIFEST = ROOT / "analysis/bonsai2/prompts/retrieval_v2_manifest.json"
MODEL = ROOT / "models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf"
TOKENIZER = ROOT / "bin/cuda/llama-tokenize"
SCORER = HERE / "split_scorer"
CAPTURE = ROOT / "analysis/bonsai2/validation_gates/state_transplant/state_gate"
CONTROLLED = ROOT / "analysis/bonsai2/controlled_state/controlled_state"
PARENT = ROOT / "analysis/bonsai2/fresh_state/FRESH_STATE_FREEZE.json"
FREEZE = HERE / "REMOTE_STATE_FREEZE.json"
TCRIT7 = 2.3646242510102993


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ids_sha(ids):
    return hashlib.sha256(struct.pack("<" + "i" * len(ids), *ids)).hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def tokenize(text):
    proc = subprocess.run([str(TOKENIZER), "-m", str(MODEL), "--stdin", "--ids", "--no-bos"],
                          input=text.encode(), capture_output=True, check=True)
    return ast.literal_eval(proc.stdout.decode())


def maxdiff(a, b):
    if len(a) != len(b):
        raise ValueError("NLL vector lengths differ")
    return max(abs(x - y) for x, y in zip(a, b))


def state_pos(path):
    with path.open("rb") as stream:
        raw = stream.read(20)
    if len(raw) != 20:
        raise ValueError(f"Short recurrent state: {path}")
    fields = struct.unpack("<IIIII", raw)
    if fields[:3] != (0xAF143CD8, 0, 1) or fields[4] != 0:
        raise ValueError(f"Unexpected recurrent state header: {path}")
    return fields[3]


def freeze():
    if FREEZE.exists():
        raise ValueError("Freeze already exists; refusing to replace it")
    manifest = json.loads(PROMPT_MANIFEST.read_text())
    if sha(PROMPTS) != manifest["rows_sha256"]:
        raise ValueError("Corrected prompt file changed")
    parent = json.loads(PARENT.read_text())
    if sha(MODEL) != parent["models"]["original"]["model_sha256"]:
        raise ValueError("Original checkpoint changed")
    rows = {r["id"]: r for r in map(json.loads, PROMPTS.read_text().splitlines())}
    template = chat_template(MODEL)
    ids_dir = HERE / "ids"
    ids_dir.mkdir(exist_ok=True)
    prompts = []
    cases = []
    tsv = []
    for reg in range(8, 16):
        pair = {}
        for swap in ("base", "swap"):
            prompt_id = f"rv2-{reg:02d}-q1-long-far-{swap}"
            row = rows[prompt_id]
            other = rows[f"rv2-{reg:02d}-q2-long-far-{swap}"]
            gold = row["answer"]
            wrong = row["records"][other["target_entry"] - 1]["code"]
            if gold == wrong or row["registry_id"] != f"registry-{reg:02d}":
                raise ValueError("Invalid candidate code pair")
            chat = render_chat(template, row["prompt"])
            if hashlib.sha256(chat.encode()).hexdigest() != row["chat_sha256"]:
                raise ValueError(f"Chat rendering changed: {prompt_id}")
            prefix = tokenize(chat)
            if len(prefix) != row["prompt_tokens"] or not 11900 <= len(prefix) <= 12600:
                raise ValueError(f"Prompt token count changed: {prompt_id}")
            pinfo = {"prompt_id": prompt_id, "registry_id": row["registry_id"], "swap": swap,
                     "prefix_token_count": len(prefix), "prefix_ids_sha256": ids_sha(prefix),
                     "recent_1024_ids_sha256": ids_sha(prefix[-1024:]),
                     "gold_code": gold, "wrong_code": wrong, "chat_sha256": row["chat_sha256"],
                     "prompt_sha256": row["prompt_sha256"]}
            pair[swap] = (pinfo, prefix, row)
            prompts.append(pinfo)
            for role, code in (("gold", gold), ("wrong", wrong)):
                full = tokenize(chat + code + "<|im_end|>")
                if full[:len(prefix)] != prefix or len(full) - len(prefix) != 9:
                    raise ValueError(f"Candidate tokenization misaligned: {prompt_id} {role}")
                target = full[len(prefix):]
                ids_path = ids_dir / f"{prompt_id}-{role}.ids"
                payload = "".join(f"{token}\n" for token in full)
                ids_path.write_text(payload)
                case_id = f"{prompt_id}-{role}"
                cases.append({"case_id": case_id, "prompt_id": prompt_id, "registry_id": row["registry_id"],
                              "swap": swap, "candidate_role": role, "candidate_code": code,
                              "prefix_token_count": len(prefix), "target_token_count": len(target),
                              "target_ids": target, "target_ids_sha256": ids_sha(target),
                              "full_ids_file": str(ids_path.relative_to(ROOT)), "full_ids_file_sha256": sha(ids_path)})
                tsv.append(f"{case_id}\t{ids_path.relative_to(ROOT)}\t{len(prefix)}\t{len(prefix)}\t9\n")
        base, base_ids, base_row = pair["base"]
        swapped, swap_ids, swap_row = pair["swap"]
        if (base["gold_code"], base["wrong_code"]) != (swapped["wrong_code"], swapped["gold_code"]):
            raise ValueError(f"Codes did not swap: {reg}")
        if base_row["question"] != swap_row["question"] or len(base_ids) != len(swap_ids):
            raise ValueError(f"Question or length changed: {reg}")
        if base_ids[-1024:] != swap_ids[-1024:]:
            raise ValueError(f"Recent 1K token IDs differ: {reg}")
        if base_ids == swap_ids:
            raise ValueError(f"Remote assignment did not change: {reg}")
    if len(prompts) != 16 or len(cases) != 32:
        raise ValueError("Incomplete eight-registry design")
    for pinfo in prompts:
        roles = {c["candidate_role"]: c for c in cases if c["prompt_id"] == pinfo["prompt_id"]}
        if roles["gold"]["target_ids"][0] != roles["wrong"]["target_ids"][0]:
            raise ValueError("First answer token differs between candidate codes")
    cases_path = HERE / "baseline_cases.tsv"
    cases_path.write_text("".join(tsv))
    paths = [HERE / "DESIGN.md", HERE / "gate.py", HERE / "split_scorer.cpp", HERE / "build.sh",
             PROMPTS, PROMPT_MANIFEST, MODEL, TOKENIZER,
             SCORER, CAPTURE, CONTROLLED, PARENT, cases_path]
    paths += [ROOT / c["full_ids_file"] for c in cases]
    paths += [ROOT / p for p in parent["linked_cuda_library_sha256"]]
    paths += [ROOT / p for p in ("analysis/bonsai2/chat_tokens.py",
             "analysis/bonsai2/scorer/position_scorer.cpp",
             "analysis/bonsai2/validation_gates/state_transplant/state_gate.cpp",
             "analysis/bonsai2/controlled_state/controlled_state.cpp")]
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in dict.fromkeys(paths)}
    for path, expected in parent["linked_cuda_library_sha256"].items():
        if hashes[path] != expected:
            raise ValueError(f"Pinned library changed: {path}")
    write_json(FREEZE, {"purpose": "fresh remote-value recurrent-state sensitivity gate",
                        "runtime_release": parent["runtime_release"], "registries": list(range(8, 16)),
                        "model_path": str(MODEL.relative_to(ROOT)), "hashes": hashes,
                        "prompts": prompts, "cases": cases,
                        "gates": {"baseline_min_correct": 16,
                                  "identity_max_token_nll_delta": 0.0001,
                                  "meaningful_margin_nats": 0.5,
                                  "support_min_positive_registries": 6,
                                  "t_critical_df7_two_sided_95": TCRIT7}})
    print(f"Frozen {len(prompts)} prompts, {len(cases)} candidate scores, 8 registry clusters", flush=True)


def load_freeze():
    frozen = json.loads(FREEZE.read_text())
    for path, expected in frozen["hashes"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    return frozen


def baseline_rows(frozen):
    path = HERE / "baseline.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if [x["case_id"] for x in rows] != [x["case_id"] for x in frozen["cases"]]:
        raise ValueError("Baseline case IDs/order differ")
    for row, case in zip(rows, frozen["cases"]):
        if (row["p"], row["h"], row["m"], row["n_batch"], row["n_ubatch"], len(row["nll"])) != (
                case["prefix_token_count"], case["prefix_token_count"], 9, 512, 512, 9):
            raise ValueError(f"Baseline profile changed: {case['case_id']}")
    return {r["case_id"]: r["nll"] for r in rows}


def margin(gold_nll, wrong_nll):
    return sum(wrong_nll) - sum(gold_nll)


def baseline():
    frozen = load_freeze()
    raw = HERE / "baseline.jsonl"
    if not raw.exists():
        temp = HERE / "baseline.partial.jsonl"
        with (HERE / "baseline.log").open("w") as log:
            result = subprocess.run([str(SCORER), "score", str(MODEL), str(HERE / "baseline_cases.tsv"),
                                     str(temp), "512", "512"], cwd=ROOT, stdout=log, stderr=log)
        if result.returncode:
            raise RuntimeError("Baseline scoring failed; inspect baseline.log")
        os.replace(temp, raw)
    by_id = baseline_rows(frozen)
    margins = []
    for prompt in frozen["prompts"]:
        pid = prompt["prompt_id"]
        m = margin(by_id[f"{pid}-gold"], by_id[f"{pid}-wrong"])
        margins.append({"prompt_id": pid, "registry_id": prompt["registry_id"], "swap": prompt["swap"], "margin": m})
    passed = all(x["margin"] > 0 for x in margins)
    write_json(HERE / "baseline_gate.json", {"freeze_sha256": sha(FREEZE), "raw_sha256": sha(raw),
                                                  "passed": passed, "margins": margins})
    print(f"Baseline remote-value gate: {sum(x['margin'] > 0 for x in margins)}/16 positive; passed={passed}", flush=True)
    if not passed:
        raise ValueError("Baseline remote-value sensitivity gate failed")


def checked_baseline(frozen):
    gate = json.loads((HERE / "baseline_gate.json").read_text())
    if not gate["passed"] or gate["freeze_sha256"] != sha(FREEZE) or gate["raw_sha256"] != sha(HERE / "baseline.jsonl"):
        raise ValueError("Baseline prerequisite failed or changed")
    return baseline_rows(frozen)


def capture_paths(prompt_id):
    base = HERE / "captures" / prompt_id
    return base.with_suffix(".full.bin"), base.with_suffix(".recurrent.bin"), base.with_suffix(".json"), base.with_suffix(".meta.json")


def validate_capture(row, prompt, gold_case, reference, frozen):
    model = str(ROOT / frozen["model_path"])
    ids_path = str(ROOT / gold_case["full_ids_file"])
    p = prompt["prefix_token_count"]
    if (row["mode"], row["model_path"], row["ids_path"], row["p"], row["h"], row["m"]) != (
            "capture", model, ids_path, p, p, 9):
        raise ValueError(f"Capture identity changed: {prompt['prompt_id']}")
    limit = frozen["gates"]["identity_max_token_nll_delta"]
    checks = {"direct": maxdiff(row["direct_nll"], reference),
              "full_roundtrip": maxdiff(row["direct_nll"], row["full_roundtrip_nll"]),
              "recurrent_roundtrip": maxdiff(row["direct_nll"], row["partial_roundtrip_nll"])}
    if max(checks.values()) > limit:
        raise ValueError(f"Capture gate failed: {prompt['prompt_id']}: {checks}")
    return checks


def captures():
    frozen = load_freeze()
    refs = checked_baseline(frozen)
    case_by_id = {c["case_id"]: c for c in frozen["cases"]}
    (HERE / "captures").mkdir(exist_ok=True)
    items = []
    for prompt in frozen["prompts"]:
        pid = prompt["prompt_id"]
        case = case_by_id[f"{pid}-gold"]
        full, recurrent, raw, meta = capture_paths(pid)
        if not meta.exists():
            temp_full = full.with_suffix(".partial.bin")
            temp_rec = recurrent.with_suffix(".partial.bin")
            temp_raw = raw.with_suffix(".partial.json")
            with raw.with_suffix(".log").open("w") as log:
                result = subprocess.run([str(CAPTURE), "capture", str(MODEL), str(ROOT / case["full_ids_file"]),
                                         str(prompt["prefix_token_count"]), str(prompt["prefix_token_count"]), "9",
                                         str(temp_full), str(temp_rec), str(temp_raw)],
                                        cwd=ROOT, stdout=log, stderr=log)
            if result.returncode:
                raise RuntimeError(f"Capture failed: {pid}")
            validate_capture(json.loads(temp_raw.read_text()), prompt, case, refs[case["case_id"]], frozen)
            if state_pos(temp_rec) != prompt["prefix_token_count"] - 1:
                raise ValueError(f"Wrong recurrent cell position: {pid}")
            os.replace(temp_full, full)
            os.replace(temp_rec, recurrent)
            os.replace(temp_raw, raw)
            write_json(meta, {"freeze_sha256": sha(FREEZE), "full_sha256": sha(full),
                              "recurrent_sha256": sha(recurrent), "raw_sha256": sha(raw)})
        info = json.loads(meta.read_text())
        if info["freeze_sha256"] != sha(FREEZE) or any(sha(path) != info[key] for path, key in (
                (full, "full_sha256"), (recurrent, "recurrent_sha256"), (raw, "raw_sha256"))):
            raise ValueError(f"Captured artifact changed: {pid}")
        checks = validate_capture(json.loads(raw.read_text()), prompt, case, refs[case["case_id"]], frozen)
        if state_pos(recurrent) != prompt["prefix_token_count"] - 1:
            raise ValueError(f"Recurrent state position changed: {pid}")
        items.append({"prompt_id": pid, "full_path": str(full.relative_to(ROOT)),
                      "recurrent_path": str(recurrent.relative_to(ROOT)), "full_sha256": info["full_sha256"],
                      "recurrent_sha256": info["recurrent_sha256"], "raw_sha256": info["raw_sha256"],
                      "checks": checks})
        print(f"Captured {pid}: {checks}", flush=True)
    write_json(HERE / "capture_gate.json", {"freeze_sha256": sha(FREEZE), "passed": True, "captures": items})


def checked_captures(frozen):
    gate = json.loads((HERE / "capture_gate.json").read_text())
    if not gate["passed"] or gate["freeze_sha256"] != sha(FREEZE) or len(gate["captures"]) != 16:
        raise ValueError("Capture prerequisite failed")
    result = {x["prompt_id"]: x for x in gate["captures"]}
    for x in result.values():
        if sha(ROOT / x["recurrent_path"]) != x["recurrent_sha256"] or sha(ROOT / x["full_path"]) != x["full_sha256"]:
            raise ValueError(f"Capture bytes changed: {x['prompt_id']}")
    for reg in range(8, 16):
        a = next(p for p in frozen["prompts"] if p["registry_id"] == f"registry-{reg:02d}" and p["swap"] == "base")
        b = next(p for p in frozen["prompts"] if p["registry_id"] == f"registry-{reg:02d}" and p["swap"] == "swap")
        if a["prefix_token_count"] != b["prefix_token_count"] or state_pos(ROOT / result[a["prompt_id"]]["recurrent_path"]) != state_pos(ROOT / result[b["prompt_id"]]["recurrent_path"]):
            raise ValueError("Counterfactual state positions differ")
    return result


def validate_comparison(row, case, frozen, refs, sources, logpath):
    p = case["prefix_token_count"]
    if (row["model_path"], row["ids_path"], row["p"], row["h"], row["m"]) != (
            str(MODEL), str(ROOT / case["full_ids_file"]), p, p, 9):
        raise ValueError(f"Comparison identity changed: {case['case_id']}")
    if maxdiff(row["direct_nll"], refs[case["case_id"]]) > frozen["gates"]["identity_max_token_nll_delta"]:
        raise ValueError(f"Direct baseline mismatch: {case['case_id']}")
    by_label = {c["label"]: c for c in row["conditions"]}
    if set(by_label) != {"own", "counterfactual"} or len(row["conditions"]) != 2:
        raise ValueError("Wrong comparison conditions")
    for label, cond in by_label.items():
        if cond["kind"] != "partial" or cond["state_path"] != str(sources[label][0]) or sha(sources[label][0]) != sources[label][1]:
            raise ValueError(f"Wrong state source: {case['case_id']} {label}")
        if cond["imported_nll"][0] != row["direct_nll"][0] or maxdiff(cond["restored_nll"], row["direct_nll"]) > frozen["gates"]["identity_max_token_nll_delta"]:
            raise ValueError(f"Target 0 or A-B-A failed: {case['case_id']} {label}")
    if maxdiff(by_label["own"]["imported_nll"], row["direct_nll"]) > frozen["gates"]["identity_max_token_nll_delta"]:
        raise ValueError(f"Own-state identity failed: {case['case_id']}")
    if "non-consecutive token position" in logpath.read_text():
        raise ValueError(f"Unexpected state position warning: {case['case_id']}")
    return {"own_nll": by_label["own"]["imported_nll"],
            "counterfactual_nll": by_label["counterfactual"]["imported_nll"]}


def comparisons():
    frozen = load_freeze()
    refs = checked_baseline(frozen)
    captured = checked_captures(frozen)
    dest = HERE / "comparisons"
    dest.mkdir(exist_ok=True)
    prompts = {p["prompt_id"]: p for p in frozen["prompts"]}
    for case in frozen["cases"]:
        pid = case["prompt_id"]
        prompt = prompts[pid]
        counterpart = next(p for p in frozen["prompts"] if p["registry_id"] == case["registry_id"] and p["swap"] != case["swap"])
        own = ROOT / captured[pid]["recurrent_path"]
        cross = ROOT / captured[counterpart["prompt_id"]]["recurrent_path"]
        sources = {"own": (own, captured[pid]["recurrent_sha256"]),
                   "counterfactual": (cross, captured[counterpart["prompt_id"]]["recurrent_sha256"])}
        if state_pos(own) != prompt["prefix_token_count"] - 1 or state_pos(cross) != state_pos(own):
            raise ValueError("Source and destination cell positions differ")
        raw = dest / f"{case['case_id']}.json"
        meta = dest / f"{case['case_id']}.meta.json"
        logpath = dest / f"{case['case_id']}.log"
        if not meta.exists():
            temp = raw.with_suffix(".partial.json")
            cmd = [str(CONTROLLED), str(MODEL), str(ROOT / case["full_ids_file"]),
                   str(case["prefix_token_count"]), str(case["prefix_token_count"]), "9", str(temp),
                   "own", "partial", str(own), "counterfactual", "partial", str(cross)]
            with logpath.open("w") as log:
                result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=log)
            if result.returncode:
                raise RuntimeError(f"Comparison failed: {case['case_id']}")
            validate_comparison(json.loads(temp.read_text()), case, frozen, refs, sources, logpath)
            os.replace(temp, raw)
            write_json(meta, {"freeze_sha256": sha(FREEZE), "raw_sha256": sha(raw),
                              "sources": {label: value[1] for label, value in sources.items()}})
        info = json.loads(meta.read_text())
        if info["freeze_sha256"] != sha(FREEZE) or info["raw_sha256"] != sha(raw) or info["sources"] != {label: value[1] for label, value in sources.items()}:
            raise ValueError(f"Comparison artifact changed: {case['case_id']}")
        checked = validate_comparison(json.loads(raw.read_text()), case, frozen, refs, sources, logpath)
        print(f"Compared {case['case_id']}: sum NLL own {sum(checked['own_nll']):.4f}, counterfactual {sum(checked['counterfactual_nll']):.4f}", flush=True)


def t_interval(values):
    mean = statistics.mean(values)
    half = TCRIT7 * statistics.stdev(values) / math.sqrt(len(values))
    return [mean - half, mean + half]


def analyze():
    frozen = load_freeze()
    refs = checked_baseline(frozen)
    captured = checked_captures(frozen)
    prompts = {p["prompt_id"]: p for p in frozen["prompts"]}
    scored = {}
    for case in frozen["cases"]:
        pid = case["prompt_id"]
        counterpart = next(p for p in frozen["prompts"] if p["registry_id"] == case["registry_id"] and p["swap"] != case["swap"])
        sources = {"own": (ROOT / captured[pid]["recurrent_path"], captured[pid]["recurrent_sha256"]),
                   "counterfactual": (ROOT / captured[counterpart["prompt_id"]]["recurrent_path"], captured[counterpart["prompt_id"]]["recurrent_sha256"])}
        base = HERE / "comparisons" / case["case_id"]
        raw = base.with_suffix(".json")
        meta = json.loads(base.with_suffix(".meta.json").read_text())
        if meta["freeze_sha256"] != sha(FREEZE) or meta["raw_sha256"] != sha(raw):
            raise ValueError(f"Raw comparison changed: {case['case_id']}")
        scored[case["case_id"]] = validate_comparison(json.loads(raw.read_text()), case, frozen, refs,
                                                       sources, base.with_suffix(".log"))
    destinations = []
    for prompt in frozen["prompts"]:
        pid = prompt["prompt_id"]
        gold, wrong = scored[f"{pid}-gold"], scored[f"{pid}-wrong"]
        own = margin(gold["own_nll"], wrong["own_nll"])
        cross = margin(gold["counterfactual_nll"], wrong["counterfactual_nll"])
        destinations.append({"prompt_id": pid, "registry_id": prompt["registry_id"], "swap": prompt["swap"],
                             "own_margin": own, "counterfactual_margin": cross, "own_minus_counterfactual": own - cross})
    registries = []
    for reg in frozen["registries"]:
        rows = [d for d in destinations if d["registry_id"] == f"registry-{reg:02d}"]
        if len(rows) != 2 or {r["swap"] for r in rows} != {"base", "swap"}:
            raise ValueError("Incomplete base/swap registry pair")
        registries.append({"registry_id": f"registry-{reg:02d}",
                           "mean_own_minus_counterfactual": statistics.mean(r["own_minus_counterfactual"] for r in rows),
                           "base_effect": next(r["own_minus_counterfactual"] for r in rows if r["swap"] == "base"),
                           "swap_effect": next(r["own_minus_counterfactual"] for r in rows if r["swap"] == "swap")})
    values = [r["mean_own_minus_counterfactual"] for r in registries]
    mean, ci = statistics.mean(values), t_interval(values)
    positive = sum(v > 0 for v in values)
    floor = frozen["gates"]["meaningful_margin_nats"]
    if mean >= floor and ci[0] > 0 and positive >= frozen["gates"]["support_min_positive_registries"]:
        decision = "supports_state_carried_remote_value_in_this_assay"
    elif ci[1] < floor:
        decision = "excludes_0p5_nat_state_margin_effect_in_this_assay"
    else:
        decision = "unresolved"
    summary = {"freeze_sha256": sha(FREEZE), "technical_gates_passed": True,
               "baseline_gate_passed": True, "decision": decision,
               "primary_mean_margin_nats": mean, "primary_t95ci": ci,
               "positive_registries": positive, "destinations": destinations, "registries": registries}
    write_json(HERE / "analysis_summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("decision", "primary_mean_margin_nats", "primary_t95ci", "positive_registries")}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze", "baseline", "captures", "comparisons", "analyze"))
    phase = parser.parse_args().phase
    if phase == "freeze":
        freeze()
    elif phase == "baseline":
        baseline()
    elif phase == "captures":
        captures()
    elif phase == "comparisons":
        comparisons()
    else:
        analyze()


if __name__ == "__main__":
    main()

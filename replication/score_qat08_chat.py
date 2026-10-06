"""Matched QAT08 harness for the frozen chat amendment; reruns resume by shard/item.

export, curve, heldout, gsm8k, analyze. No scoring/selection changes in old experiments.
"""
from __future__ import annotations

import collections
import concurrent.futures
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import gsm8k_17b as gen
from prepare_qat08_chat import OUT, PRIOR, ROOT, WORK, wilson
from qat_08b_chat import ARM, verify
from score_17b import book_means, mean, paired_items, sha, t_interval

SCORES = WORK / "scores"
SCORER = ROOT / "work/qwen3_17b/score_17b"
CKPTS = ("init", "t8m", "t16m", "t33m", "final")


def protocol():
    return json.loads((OUT / "protocol.json").read_text())


def model_path(arm):
    if arm in protocol()["arms"]:
        path = ROOT / protocol()["arms"][arm]["file"]
    else:
        path = WORK / f"{arm}.gguf"
    if arm.startswith(ARM):
        export = json.loads((OUT / f"export_{path.stem}.json").read_text())
        assert sha(path) == export["sha256"]
        assert export["protocol_sha256"] == sha(OUT / "protocol.json")
    return path


def case_file(key):
    p = protocol()
    entry = {"validation": p["book_cases"]["validation"], "heldout": p["book_cases"]["heldout"],
             "mmlu": p["mmlu"], "retrieval": p["retrieval"]}[key]
    path = OUT / entry["file"]
    assert sha(path) == entry["sha256"]
    return path


def checked_rows(path, expected):
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert [r["id"] for r in rows] == expected, path
    assert all(all(math.isfinite(x) for x in r["values"]) for r in rows)
    return {r["id"]: r["values"] for r in rows}


def score(arm, key):
    SCORES.mkdir(parents=True, exist_ok=True)
    target = SCORES / f"{arm}-{key}.jsonl"
    model = model_path(arm)
    lines = case_file(key).read_text().splitlines()
    ids = [l.split("\t", 1)[0] for l in lines]
    if not target.exists() and arm != ARM and key in ("mmlu", "retrieval") and arm in protocol()["arms"]:
        source = ROOT / "work/qat08/scores" / target.name
        expected = protocol()["comparator_output_sha256"][str(source.relative_to(ROOT))]
        assert sha(source) == expected
        shutil.copyfile(source, target)
    if not target.exists():
        shard_dir = SCORES / f"{arm}-{key}"
        shard_dir.mkdir(exist_ok=True)
        files = []
        for start in range(0, len(lines), 256):
            piece = shard_dir / f"{start:05d}.jsonl"
            subset = lines[start:start+256]
            expected = ids[start:start+256]
            if not piece.exists():
                cases = shard_dir / f"{start:05d}.tsv"
                cases.write_text("\n".join(subset) + "\n")
                tmp = piece.with_suffix(".partial")
                with piece.with_suffix(".log").open("w") as log:
                    subprocess.run([str(SCORER), str(model), str(cases), str(tmp), "99", "248077", "model"],
                                   stderr=log, check=True)
                checked_rows(tmp, expected)
                tmp.replace(piece)
            checked_rows(piece, expected)
            files.append(piece)
            print(f"{arm} {key}: {min(start+256,len(lines))}/{len(lines)}", flush=True)
        tmp = target.with_suffix(".partial")
        with tmp.open("w") as f:
            for piece in files:
                f.write(piece.read_text())
        tmp.replace(target)
    return checked_rows(target, ids)


def export():
    assert (WORK / ARM / "DONE").exists()
    for ckpt in CKPTS:
        subprocess.run([sys.executable, str(ROOT / "qat_08b_chat.py"), "export", "--ckpt", ckpt], check=True)


def curve():
    arms = ["fp", "gptq", "qat_free", "qat_42"] + [f"{ARM}-{c}" for c in CKPTS]
    result = {a: mean(book_means(score(a, "validation")).values()) for a in arms}
    (OUT / "curve.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def heldout():
    for key in ("heldout", "mmlu", "retrieval"):
        for arm in protocol()["heldout_arms"]["books_mmlu_retrieval"]:
            score(arm, key)
            print(arm, key, "complete", flush=True)


def gsm8k():
    SCORES.mkdir(parents=True, exist_ok=True)
    p = protocol()
    path = OUT / p["gsm8k"]["file"]
    assert sha(path) == p["gsm8k"]["sha256"]
    items = [json.loads(l) for l in path.read_text().splitlines()]
    directory = OUT / "gsm8k"
    directory.mkdir(exist_ok=True)
    for arm in p["heldout_arms"]["gsm8k"]:
        target = directory / f"{arm}.jsonl"
        if arm != ARM and not target.exists():
            source = PRIOR / "gsm8k" / target.name
            assert sha(source) == p["comparator_output_sha256"][str(source.relative_to(ROOT))]
            shutil.copyfile(source, target)
        if target.exists():
            rows = [json.loads(l) for l in target.read_text().splitlines()]
            assert [r["id"] for r in rows] == [it["id"] for it in items]
            continue
        journal = target.with_suffix(".partial.jsonl")
        done = {r["id"]: r for r in map(json.loads, journal.read_text().splitlines())} if journal.exists() else {}
        pending = [it for it in items if it["id"] not in done]
        # Same QAT08 generation path and 8 slots; durable journal adds resume only.
        if pending:
            with gen.serve(model_path(arm), "model", SCORES / f"{arm}-gsm8k-server.log"):
                with journal.open("a", buffering=1) as f, concurrent.futures.ThreadPoolExecutor(gen.SLOTS) as pool:
                    for it, output in zip(pending, pool.map(lambda it: gen.complete(it["prompt_ids"]), pending)):
                        pred = gen.parse(output["text"])
                        row = {"id": it["id"], **output, "gold": it["gold"], "pred": pred,
                               "correct": gen.correct(pred, it["gold"])}
                        f.write(json.dumps(row) + "\n")
                        done[it["id"]] = row
                        if len(done) % 64 == 0:
                            print(f"{arm} GSM8K {len(done)}/{len(items)}", flush=True)
        assert len(done) == len(items)
        tmp = target.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(done[it["id"]]) + "\n" for it in items))
        tmp.replace(target)


def analyze():
    p = protocol()
    arms = p["heldout_arms"]["books_mmlu_retrieval"]
    books = {a: book_means(score(a, "heldout")) for a in arms}
    names = list(books["fp"])
    book_vs = lambda a, b: t_interval([books[a][k] - books[b][k] for k in names])
    mm, rt, generations, diagnostics = {}, {}, {}, {}
    for arm in arms:
        rows = score(arm, "mmlu")
        preds, correct, margins, masses = [], [], [], []
        for case, v in rows.items():
            gold = "ABCD".index(case.rsplit("/", 1)[1])
            wrong = max(x for i, x in enumerate(v) if i != gold)
            correct.append(100.0 * (v[gold] > wrong))  # same tie rule as QAT08
            margins.append(v[gold] - wrong)
            preds.append("ABCD"[max(range(4), key=lambda i: v[i])])
            masses.append(sum(math.exp(x) for x in v))
        mm[arm] = {"accuracy": mean(correct), "margin": mean(margins),
                   "ci": wilson(sum(x > 0 for x in correct), len(correct)), "items": correct,
                   "letters": dict(collections.Counter(preds)), "mean_letter_probability_mass": mean(masses)}
        by = {}
        for case, v in score(arm, "retrieval").items():
            reg, label, _ = case.split("/")
            by.setdefault(reg, {})[label] = sum(v)
        rt[arm] = [by[r]["wrong"]-by[r]["gold"] for r in sorted(by)]
        path = OUT / "gsm8k" / f"{arm}.jsonl"
        rows = [json.loads(l) for l in path.read_text().splitlines()]
        assert len(rows) == p["gsm8k"]["items"]
        assert all(gen.correct(gen.parse(r["text"]), r["gold"]) == r["correct"] for r in rows)
        generations[arm] = rows
        diagnostics[arm] = {"accuracy": wilson(sum(r["correct"] for r in rows), len(rows)),
            "naturally_terminated_correct": wilson(sum(r["correct"] and r["stop_type"] == "eos" for r in rows), len(rows)),
            "stop_types": dict(collections.Counter(r["stop_type"] for r in rows)),
            "parse_fallback": sum(r.get("parse_fallback", False) for r in rows),
            "mean_output_tokens": mean(r["tokens"] or 0 for r in rows)}
    comparisons = {}
    for base in ("fp", "gptq", "qat_free", "qat_42"):
        comparisons[base] = {"book_nll": book_vs(ARM, base),
            "mmlu_accuracy_points": paired_items(mm[ARM]["items"], mm[base]["items"]),
            "gsm8k_accuracy_points": paired_items([100*r["correct"] for r in generations[ARM]],
                                                   [100*r["correct"] for r in generations[base]]),
            "retrieval_margin": t_interval([x-y for x,y in zip(rt[ARM], rt[base])])}
    rules = p["decision"]
    gates = {"mmlu": mm[ARM]["ci"]["lo"] > rules["mmlu_lower_bound_points"],
             "gsm8k": diagnostics[ARM]["accuracy"]["lo"] > rules["gsm8k_lower_bound_points"],
             "gsm8k_without_nontermination_credit": diagnostics[ARM]["naturally_terminated_correct"]["lo"] > rules["gsm8k_lower_bound_points"],
             "book_cost": comparisons["qat_42"]["book_nll"]["hi"] <= rules["allowed_book_nll_cost"]}
    chat = all(gates[k] for k in ("mmlu", "gsm8k", "gsm8k_without_nontermination_credit"))
    result = {"protocol_sha256": sha(OUT / "protocol.json"), "gates": gates,
        "chat_recovered": chat, "amendment_success": chat and gates["book_cost"],
        "book_nll": {a: mean(books[a].values()) for a in arms}, "book_nll_by_book": books,
        "mmlu": {a: {k: v for k,v in mm[a].items() if k != "items"} for a in arms},
        "gsm8k": diagnostics, "retrieval": {a: {"accuracy": 100*mean(x>0 for x in rt[a]),
                                                   "margin": mean(rt[a])} for a in arms},
        "chat_minus_comparators": comparisons, "curve": json.loads((OUT / "curve.json").read_text()),
        "models": {a: {"file": str(model_path(a).relative_to(ROOT)), "sha256": sha(model_path(a))} for a in arms},
        "raw_output_sha256": {str(f.relative_to(ROOT)): sha(f) for f in sorted(SCORES.glob("*.jsonl"))},
        "gsm8k_output_sha256": {a: sha(OUT / "gsm8k" / f"{a}.jsonl") for a in arms}}
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("gates", "chat_recovered", "amendment_success", "book_nll")}, indent=2))


if __name__ == "__main__":
    verify(check_data=False)
    {"export": export, "curve": curve, "heldout": heldout, "gsm8k": gsm8k, "analyze": analyze}[sys.argv[1]]()

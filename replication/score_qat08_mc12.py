"""Matched harness for QAT08-MC12 (results/qat08_mc12/protocol.json); resumes by shard and item.

FP, C and M outputs on the reused sets (MMLU-Redux, binding, retrieval, GSM8K) are copied from the sealed MCU run
after hash checks; the fresh books and MMLU-fresh2 are scored for every model; MC12 is scored on everything.

    python score_qat08_mc12.py curve | heldout | gsm8k | analyze
"""
from __future__ import annotations

import concurrent.futures
import json
import shutil
import sys
from pathlib import Path

import gsm8k_17b as gen
from diag_metrics import binding_metrics, letter_metrics, wilson
from diag_run import score_cases
from score_17b import book_means, mean, paired_items, sha, t_interval

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/qat08_mc12"
WORK = ROOT / "work/qat08_mc12"
SCORES = WORK / "scores"
MCU_SCORES = ROOT / "work/qat08_mcu/scores"
MCU_GSM8K = ROOT / "results/qat08_mcu/gsm8k"
REUSED_KEYS = ("mmlu", "binding", "retrieval")
CKPTS = ("init", "t8m", "t16m", "t33m", "final")


def protocol() -> dict:
    return json.loads((OUT / "protocol.json").read_text())


def names(p: dict) -> list[str]:
    return [*p["comparators"], *p["arms"]]


def model_path(name: str, ckpt: str = "final") -> Path:
    p = protocol()
    if name in p["comparators"]:
        path = ROOT / p["comparators"][name]["file"]
        assert sha(path) == p["comparators"][name]["sha256"], name
        return path
    path = WORK / f"{name}-{ckpt}.gguf"
    record = json.loads((OUT / f"export_{name}-{ckpt}.json").read_text())
    assert sha(path) == record["sha256"] and record["protocol_sha256"] == sha(OUT / "protocol.json"), path
    return path


def case_path(key: str) -> Path:
    c = protocol()["cases"]
    entry = {"validation": c["book_cases"]["validation"], "heldout": c["book_cases"]["heldout"], "mmlu": c["mmlu"],
             "mmlu_fresh2": c["mmlu_fresh2"], "binding": c["binding"], "retrieval": c["retrieval"]}[key]
    path = OUT / entry["file"]
    assert sha(path) == entry["sha256"], key
    return path


def durable_copy(source: Path, target: Path) -> None:
    import os
    shutil.copyfile(source, target)
    with target.open("rb") as f:
        os.fsync(f.fileno())
    fd = os.open(target.parent, os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)


def score(name: str, key: str, ckpt: str = "final") -> dict:
    p = protocol()
    SCORES.mkdir(parents=True, exist_ok=True)
    label = name if ckpt == "final" else f"{name}-{ckpt}"
    target = SCORES / f"{label}-{key}.jsonl"
    if name in p["comparators"] and key in REUSED_KEYS:
        source = f"work/qat08_mcu/scores/{name}-{key}.jsonl"
        assert sha(ROOT / source) == p["reused_outputs"]["scores"][source], source
        if not target.exists():
            durable_copy(ROOT / source, target)
        assert sha(target) == p["reused_outputs"]["scores"][source], target
    return score_cases(model_path(name, ckpt), case_path(key), target)


def curve() -> None:
    p = protocol()
    result = {n: mean(book_means(score(n, "validation")).values()) for n in p["comparators"]}
    for arm in p["arms"]:
        for ckpt in CKPTS:
            result[f"{arm}-{ckpt}"] = mean(book_means(score(arm, "validation", ckpt)).values())
    (OUT / "curve.json").write_text(json.dumps(result, indent=2) + "\n")


def heldout() -> None:
    p = protocol()
    for key in ("heldout", "mmlu_fresh2", "mmlu", "binding", "retrieval"):
        for name in names(p):
            score(name, key)
            print(name, key, "scored", flush=True)


def check_gsm8k(rows: list[dict], items: list[dict]) -> list[dict]:
    """Exact frozen IDs in order (no duplicates or gaps), expected gold, and the parse and correctness reproduced."""
    assert [r["id"] for r in rows] == [it["id"] for it in items], "GSM8K rows differ from the frozen items (IDs or order)"
    for r, it in zip(rows, items):
        assert r["gold"] == it["gold"], f"GSM8K gold changed: {r['id']}"
        assert r["pred"] == gen.parse(r["text"]), f"GSM8K parse does not reproduce: {r['id']}"
        assert r["correct"] == gen.correct(r["pred"], it["gold"]), f"GSM8K correctness does not reproduce: {r['id']}"
    return rows


def gsm8k_items(p: dict) -> list[dict]:
    path = OUT / p["cases"]["gsm8k"]["file"]
    assert sha(path) == p["cases"]["gsm8k"]["sha256"], "GSM8K cases changed"
    return [json.loads(l) for l in path.read_text().splitlines()]


def load_gsm8k(name: str, p: dict) -> list[dict]:
    target = OUT / "gsm8k" / f"{name}.jsonl"
    if name in p["comparators"]:
        assert sha(target) == p["reused_outputs"]["gsm8k"][f"results/qat08_mcu/gsm8k/{name}.jsonl"], f"copied {name} GSM8K changed"
    return check_gsm8k([json.loads(l) for l in target.read_text().splitlines()], gsm8k_items(p))


def gsm8k() -> None:
    p = protocol()
    items = gsm8k_items(p)
    directory = OUT / "gsm8k"
    directory.mkdir(exist_ok=True)
    for name in names(p):
        target = directory / f"{name}.jsonl"
        if name in p["comparators"]:
            source = f"results/qat08_mcu/gsm8k/{name}.jsonl"
            assert sha(ROOT / source) == p["reused_outputs"]["gsm8k"][source], source
            if not target.exists():
                durable_copy(ROOT / source, target)
            load_gsm8k(name, p)
            continue
        if target.exists():
            load_gsm8k(name, p)
            continue
        journal = directory / f"{name}.partial.jsonl"
        done = {r["id"]: r for r in map(json.loads, journal.read_text().splitlines())} if journal.exists() else {}
        pending = [it for it in items if it["id"] not in done]
        if pending:
            gen.PORT = 8096
            with gen.serve(model_path(name), "model", WORK / f"{name}-gsm8k-server.log"):
                with journal.open("a", buffering=1) as f, concurrent.futures.ThreadPoolExecutor(gen.SLOTS) as pool:
                    for it, out in zip(pending, pool.map(lambda it: gen.complete(it["prompt_ids"]), pending)):
                        pred = gen.parse(out["text"])
                        row = {"id": it["id"], **out, "gold": it["gold"], "pred": pred, "correct": gen.correct(pred, it["gold"])}
                        f.write(json.dumps(row) + "\n")
                        done[it["id"]] = row
        assert len(done) == len(items)
        tmp = target.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(done[it["id"]]) + "\n" for it in items))
        tmp.replace(target)
        load_gsm8k(name, p)


def raw_hits(rows: dict) -> list[float]:
    """Per-item uncalibrated hit (gold letter strictly highest), in row order."""
    out = []
    for case_id, v in rows.items():
        g = "ABCD".index(case_id.rsplit("/", 1)[1])
        out.append(100.0 * (v[g] > max(x for k, x in enumerate(v) if k != g)))
    return out


def decide_mc12(summary: dict, rules: dict) -> dict:
    a = summary["arms"]["MC12"]
    eff = a["mmlu_pairwise_vs_M"]
    material = eff["mean"] >= rules["material_pairwise_points"] and eff["lo"] > 0
    fresh = a["mmlu_fresh2_pairwise_vs_M"]["lo"] > rules["fresh_lo_above"]
    c, r = rules["costs"], rules["recovery"]
    costs = (a["books_vs_M"]["hi"] <= c["books_vs_M_hi"] and a["books_vs_C"]["hi"] <= c["books_vs_C_hi"]
             and a["gsm8k_vs_M"]["lo"] >= c["gsm8k_vs_M_lo"] and a["retrieval_accuracy"] >= c["retrieval_min"])
    floor = (a["mmlu"]["lo"] > r["mmlu_lo"] and a["mmlu_fresh2"]["lo"] > r["fresh_lo"] and a["gsm8k"]["lo"] > r["gsm8k_lo"]
             and a["gsm8k_strict"]["lo"] > r["gsm8k_lo"])
    if material and fresh and costs and floor:
        rec = "increase_mc_to_12.5"
    elif eff["hi"] < 0:
        rec = "keep_m_6.25_mc12_worse"
    else:
        rec = "keep_m_6.25_inconclusive"
    return {"material_mc_effect": material, "fresh_corroborated": fresh, "costs_ok": costs, "recovery_floor": floor,
            "binding_raw_target_met": a["binding_raw"] >= rules["binding_raw_target"],
            "binding_raw_vs_M": a["binding_raw_vs_M"], "recommendation": rec}


def analyze() -> None:
    p = protocol()
    ns = names(p)
    books = {n: book_means(score(n, "heldout")) for n in ns}
    raw = {k: {n: score(n, k) for n in ns} for k in ("mmlu", "mmlu_fresh2", "binding")}
    letters = {n: letter_metrics(raw["mmlu"][n]) for n in ns}
    fresh = {n: letter_metrics(raw["mmlu_fresh2"][n]) for n in ns}
    binding = {n: binding_metrics(raw["binding"][n]) for n in ns}
    gsm = {n: load_gsm8k(n, p) for n in ns}
    retrieval = {}
    for n in ns:
        by = {}
        for case, v in score(n, "retrieval").items():
            reg, label, _ = case.split("/")
            by.setdefault(reg, {})[label] = sum(v)
        retrieval[n] = [by[r]["wrong"] - by[r]["gold"] for r in sorted(by)]
    book_names = list(books["fp"])
    pct = lambda xs: [100 * x for x in xs]
    vs_book = lambda n, ref: t_interval([books[n][b] - books[ref][b] for b in book_names]) if n != ref else None
    summary = {"arms": {}}
    for n in ns:
        strict = [r["correct"] and r["stop_type"] == "eos" for r in gsm[n]]
        summary["arms"][n] = {
            "books": mean(books[n].values()), "books_vs_M": vs_book(n, "M"), "books_vs_C": vs_book(n, "C"),
            "mmlu": wilson(round(letters[n]["raw_accuracy"] * letters[n]["items"] / 100), letters[n]["items"]),
            "mmlu_calibrated_accuracy": letters[n]["calibrated_accuracy"], "mmlu_pairwise": letters[n]["pairwise"],
            "mmlu_pairwise_ci": [letters[n]["pairwise_lo"], letters[n]["pairwise_hi"]],
            "mmlu_pairwise_vs_M": paired_items(pct(letters[n]["pairwise_items"]), pct(letters["M"]["pairwise_items"])),
            "mmlu_pairwise_vs_C": paired_items(pct(letters[n]["pairwise_items"]), pct(letters["C"]["pairwise_items"])),
            "letter_mass": letters[n]["letter_mass"], "letters": letters[n]["predicted"],
            "mmlu_fresh2": wilson(round(fresh[n]["raw_accuracy"] * fresh[n]["items"] / 100), fresh[n]["items"]),
            "mmlu_fresh2_pairwise": fresh[n]["pairwise"], "mmlu_fresh2_pairwise_ci": [fresh[n]["pairwise_lo"], fresh[n]["pairwise_hi"]],
            "mmlu_fresh2_pairwise_vs_M": paired_items(pct(fresh[n]["pairwise_items"]), pct(fresh["M"]["pairwise_items"])),
            "binding_raw": binding[n]["raw_accuracy"], "binding_raw_wilson": binding[n]["raw_wilson"],
            "binding_calibrated": binding[n]["calibrated_accuracy"], "binding_pairwise": binding[n]["pairwise"],
            "binding_raw_vs_M": paired_items(raw_hits(raw["binding"][n]), raw_hits(raw["binding"]["M"])),
            "gsm8k": wilson(sum(r["correct"] for r in gsm[n]), len(gsm[n])), "gsm8k_strict": wilson(sum(strict), len(gsm[n])),
            "gsm8k_vs_M": paired_items([100 * r["correct"] for r in gsm[n]], [100 * r["correct"] for r in gsm["M"]]),
            "gsm8k_vs_C": paired_items([100 * r["correct"] for r in gsm[n]], [100 * r["correct"] for r in gsm["C"]]),
            "gsm8k_eos": sum(r["stop_type"] == "eos" for r in gsm[n]),
            "retrieval_accuracy": 100 * mean(x > 0 for x in retrieval[n]), "retrieval_margin": mean(retrieval[n]),
        }
    decision = decide_mc12(summary, p["decision"])
    result = {"protocol_sha256": sha(OUT / "protocol.json"), "summary": summary, "decision": decision,
              "curve": json.loads((OUT / "curve.json").read_text()), "book_nll_by_book": books,
              "models": {n: {"file": str(model_path(n).relative_to(ROOT)), "sha256": sha(model_path(n))} for n in ns},
              "raw_output_sha256": {str(f.relative_to(ROOT)): sha(f) for f in sorted(SCORES.glob("*.jsonl"))},
              "gsm8k_output_sha256": {n: sha(OUT / "gsm8k" / f"{n}.jsonl") for n in ns}}
    tmp = OUT / "results.tmp"
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    tmp.replace(OUT / "results.json")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    {"curve": curve, "heldout": heldout, "gsm8k": gsm8k, "analyze": analyze}[sys.argv[1]]()

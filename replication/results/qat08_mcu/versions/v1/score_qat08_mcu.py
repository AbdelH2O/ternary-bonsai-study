"""Matched harness for QAT08-MCU (results/qat08_mcu/protocol.json); resumes by shard and item.

    python score_qat08_mcu.py export | curve | heldout | gsm8k | analyze
"""
from __future__ import annotations

import concurrent.futures
import json
import shutil
import subprocess
import sys
from pathlib import Path

import gsm8k_17b as gen
from diag_metrics import binding_metrics, letter_metrics, wilson
from diag_run import score_cases
from score_17b import book_means, mean, paired_items, sha, t_interval

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/qat08_mcu"
WORK = ROOT / "work/qat08_mcu"
SCORES = WORK / "scores"
CHAT_OUT = ROOT / "results/qat08_chat"
CHAT_NAME = {"fp": "fp", "qat_42": "qat_42", "C": "qat_chat_42"}


def protocol() -> dict:
    return json.loads((OUT / "protocol.json").read_text())


def ckpts(p: dict, arm: str) -> list[str]:
    tags = [tag for _, tag in p["training"]["checkpoints"]]
    return (["init"] if p["arms"][arm]["init"] == "folded" else []) + tags


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
             "mmlu_fresh": c["mmlu_fresh"], "binding": c["binding"], "retrieval": c["retrieval"]}[key]
    path = OUT / entry["file"]
    assert sha(path) == entry["sha256"], key
    return path


def score(name: str, key: str, ckpt: str = "final") -> dict:
    SCORES.mkdir(parents=True, exist_ok=True)
    label = name if ckpt == "final" else f"{name}-{ckpt}"
    target = SCORES / f"{label}-{key}.jsonl"
    if not target.exists() and name in CHAT_NAME and key in ("mmlu", "retrieval"):
        source = ROOT / f"work/qat08_chat/scores/{CHAT_NAME[name]}-{key}.jsonl"
        expected = json.loads((CHAT_OUT / "results.json").read_text())["raw_output_sha256"][str(source.relative_to(ROOT))]
        assert sha(source) == expected, source
        shutil.copyfile(source, target)
    return score_cases(model_path(name, ckpt), case_path(key), target)


def export() -> None:
    p = protocol()
    for arm in p["arms"]:
        for ckpt in ckpts(p, arm):
            subprocess.run([sys.executable, str(ROOT / "qat_08b_mcu.py"), "export", "--arm", arm, "--ckpt", ckpt], check=True)


def curve() -> None:
    p = protocol()
    result = {name: mean(book_means(score(name, "validation")).values()) for name in p["comparators"]}
    for arm in p["arms"]:
        for ckpt in ckpts(p, arm):
            result[f"{arm}-{ckpt}"] = mean(book_means(score(arm, "validation", ckpt)).values())
    (OUT / "curve.json").write_text(json.dumps(result, indent=2) + "\n")


def heldout() -> None:
    p = protocol()
    for key in ("heldout", "mmlu", "mmlu_fresh", "binding", "retrieval"):
        for name in [*p["comparators"], *p["arms"]]:
            score(name, key)
            print(name, key, "scored", flush=True)


def gsm8k() -> None:
    p = protocol()
    items = [json.loads(l) for l in (OUT / p["cases"]["gsm8k"]["file"]).read_text().splitlines()]
    directory = OUT / "gsm8k"
    directory.mkdir(exist_ok=True)
    reused = json.loads((CHAT_OUT / "results.json").read_text())["gsm8k_output_sha256"]
    for name in [*p["comparators"], *p["arms"]]:
        target = directory / f"{name}.jsonl"
        if name in CHAT_NAME and not target.exists():
            source = CHAT_OUT / "gsm8k" / f"{CHAT_NAME[name]}.jsonl"
            assert sha(source) == reused[CHAT_NAME[name]], source
            shutil.copyfile(source, target)
        if target.exists():
            continue
        journal = directory / f"{name}.partial.jsonl"
        done = {r["id"]: r for r in map(json.loads, journal.read_text().splitlines())} if journal.exists() else {}
        pending = [it for it in items if it["id"] not in done]
        if pending:
            gen.PORT = 8095
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


def decide_mcu(summary: dict, rules: dict, branch: str, arms: list[str]) -> dict:
    out = {"arms": {}}
    for arm in arms:
        a = summary["arms"][arm]
        effect = a["mmlu_pairwise_vs_C"]["mean"] >= rules["effect_points"] and a["mmlu_pairwise_vs_C"]["lo"] > 0
        book_gain = a["books_vs_C"]["mean"] <= rules["book_gain"] and a["books_vs_C"]["hi"] < 0
        costs = (a["books_vs_C"]["hi"] <= rules["costs"]["book_hi"] and a["gsm8k_vs_C"]["lo"] >= rules["costs"]["gsm8k_lo"]
                 and a["retrieval_accuracy"] >= rules["costs"]["retrieval_min"])
        recovered = (a["mmlu"]["lo"] > rules["recovery"]["mmlu_lo"] and a["gsm8k"]["lo"] > rules["recovery"]["gsm8k_lo"]
                     and a["gsm8k_strict"]["lo"] > rules["recovery"]["gsm8k_lo"])
        v = {"effect": effect, "book_gain": book_gain, "costs_ok": costs, "recovered": recovered,
             "fresh_confirmed": a["mmlu_fresh"]["lo"] > rules["fresh_recovery_lo"],
             "format_only": a["letter_mass"] >= rules["format_only"]["letter_mass_min"] and a["mmlu_pairwise"] < rules["format_only"]["pairwise_below"]}
        if arm in ("M", "MU"):
            v["H_M"] = effect and a["mmlu_fresh_pairwise_lo"] > rules["fresh_pairwise_lo_above"] and a["binding_calibrated"] >= rules["binding_min"]
        if arm in ("U", "MU"):
            v["H_U"] = book_gain or effect
        if arm == "B":
            v["H_B"] = book_gain or effect
        out["arms"][arm] = v
    winners = sorted((a for a, v in out["arms"].items() if v["recovered"] and v["fresh_confirmed"] and v["costs_ok"]),
                     key=lambda a: -summary["arms"][a]["mmlu_pairwise"])
    if winners:
        rec = "rent_1p7b_with_recipe"
    elif any(out["arms"][a]["effect"] for a in out["arms"] if a in ("M", "MU")):
        rec = "iterate_0p8b_mc_data"
    elif "B" in out["arms"] and branch == "knowledge_lost" and \
            (out["arms"]["B"]["effect"] or summary["arms"]["B"]["mmlu_fresh_pairwise_lo"] > rules["ladder_fresh_pairwise_lo_above"]) and \
            not any(out["arms"][a]["effect"] for a in out["arms"] if a in ("U", "MU")):
        rec = "rent_1p7b_token_ladder"
    elif any(v["effect"] or v["book_gain"] for v in out["arms"].values()):
        rec = "iterate_0p8b"
    else:
        rec = "stop_data_iteration_0p8b"
    return {**out, "recommendation": rec, "recipe_arms": winners}


def analyze() -> None:
    p = protocol()
    names = [*p["comparators"], *p["arms"]]
    books = {n: book_means(score(n, "heldout")) for n in names}
    letters = {n: letter_metrics(score(n, "mmlu")) for n in names}
    fresh = {n: letter_metrics(score(n, "mmlu_fresh")) for n in names}
    binding = {n: binding_metrics(score(n, "binding")) for n in names}
    gsm = {n: [json.loads(l) for l in (OUT / "gsm8k" / f"{n}.jsonl").read_text().splitlines()] for n in names}
    retrieval = {}
    for n in names:
        by = {}
        for case, v in score(n, "retrieval").items():
            reg, label, _ = case.split("/")
            by.setdefault(reg, {})[label] = sum(v)
        retrieval[n] = [by[r]["wrong"] - by[r]["gold"] for r in sorted(by)]
    book_names = list(books["fp"])
    pct = lambda xs: [100 * x for x in xs]
    summary = {"arms": {}}
    for n in names:
        strict_hits = [r["correct"] and r["stop_type"] == "eos" for r in gsm[n]]
        summary["arms"][n] = {
            "books": mean(books[n].values()),
            "books_vs_C": t_interval([books[n][b] - books["C"][b] for b in book_names]) if n != "C" else {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 12},
            "mmlu": wilson(round(letters[n]["raw_accuracy"] * letters[n]["items"] / 100), letters[n]["items"]),
            "mmlu_calibrated_accuracy": letters[n]["calibrated_accuracy"], "mmlu_pairwise": letters[n]["pairwise"],
            "mmlu_pairwise_ci": [letters[n]["pairwise_lo"], letters[n]["pairwise_hi"]],
            "mmlu_pairwise_vs_C": paired_items(pct(letters[n]["pairwise_items"]), pct(letters["C"]["pairwise_items"])),
            "letter_mass": letters[n]["letter_mass"], "letters": letters[n]["predicted"],
            "mmlu_fresh": wilson(round(fresh[n]["raw_accuracy"] * fresh[n]["items"] / 100), fresh[n]["items"]),
            "mmlu_fresh_pairwise": fresh[n]["pairwise"], "mmlu_fresh_pairwise_lo": fresh[n]["pairwise_lo"],
            "binding_calibrated": binding[n]["calibrated_accuracy"], "binding_raw": binding[n]["raw_accuracy"],
            "gsm8k": wilson(sum(r["correct"] for r in gsm[n]), len(gsm[n])),
            "gsm8k_strict": wilson(sum(strict_hits), len(gsm[n])),
            "gsm8k_vs_C": paired_items([100 * r["correct"] for r in gsm[n]], [100 * r["correct"] for r in gsm["C"]]),
            "gsm8k_eos": sum(r["stop_type"] == "eos" for r in gsm[n]),
            "retrieval_accuracy": 100 * mean(x > 0 for x in retrieval[n]), "retrieval_margin": mean(retrieval[n]),
            "retrieval_vs_C": t_interval([x - y for x, y in zip(retrieval[n], retrieval["C"])]) if n != "C" else None,
        }
    decision = decide_mcu(summary, p["decision"], p["diagnostics"]["branch"], list(p["arms"]))
    result = {"protocol_sha256": sha(OUT / "protocol.json"), "summary": summary, "decision": decision,
              "curve": json.loads((OUT / "curve.json").read_text()), "book_nll_by_book": books,
              "models": {n: {"file": str(model_path(n).relative_to(ROOT)), "sha256": sha(model_path(n))} for n in names},
              "raw_output_sha256": {str(f.relative_to(ROOT)): sha(f) for f in sorted(SCORES.glob("*.jsonl"))},
              "gsm8k_output_sha256": {n: sha(OUT / "gsm8k" / f"{n}.jsonl") for n in names}}
    tmp = OUT / "results.tmp"
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    tmp.replace(OUT / "results.json")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    {"export": export, "curve": curve, "heldout": heldout, "gsm8k": gsm8k, "analyze": analyze}[sys.argv[1]]()

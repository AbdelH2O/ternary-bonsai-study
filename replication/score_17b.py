"""Score, select and analyze the frozen 1.7B comparison (results/q17b/protocol.json).

Run in order with the project venv (CUDA host):
    python3 score_17b.py validation     # all arms, three validation books
    python3 score_17b.py select         # method and Prism reference; refuses to overwrite
    python3 score_17b.py heldout        # books, MMLU-Redux, retrieval for every arm
    python3 score_17b.py gsm8k          # greedy generation for the GSM8K arms
    python3 score_17b.py analyze
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from scipy import stats

import gsm8k_17b

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/q17b"
WORK = ROOT / "work/qwen3_17b"
SCORES = WORK / "scores"
SCORER = WORK / "score_17b"
NORM_VOCAB = 151669
PRISM = "prism_gguf/Ternary-Bonsai-1.7B-PQ2_0.gguf"
ARMS = {"fp": ("base_tied-f32.gguf", "model"), "prism": (PRISM, "model"), "prism_noyarn": (PRISM, "none"),
        "absmax": ("base-pq2.gguf", "model"), "ls": ("base-pq2-ls.gguf", "model"),
        "gptq": ("base-pq2-gptq.gguf", "model"), "gptqh": ("folded-pq2-gptq.gguf", "model")}
BUILT = ("ls", "gptq", "gptqh")
BANDS = {"early": slice(0, 64), "mid": slice(64, 448), "late": slice(448, 960)}


def sha(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def protocol() -> dict:
    return json.loads((OUT / "protocol.json").read_text())


def case_file(key: str) -> Path:
    p = protocol()
    entry = {"validation": p["book_cases"]["validation"], "heldout": p["book_cases"]["heldout"],
             "mmlu": p["mmlu"], "retrieval": p["retrieval"]}[key]
    path = OUT / entry["file"]
    assert sha(path) == entry["sha256"], key
    return path


def check_builds() -> None:
    for arm in BUILT:
        build = json.loads((OUT / f"build_{arm}.json").read_text())
        assert build["protocol_sha256"] == sha(OUT / "protocol.json"), arm
        assert build["sha256"] == sha(WORK / ARMS[arm][0]), arm


def score(arm: str, key: str) -> dict[str, list[float]]:
    SCORES.mkdir(exist_ok=True)
    out = SCORES / f"{arm}-{key}.jsonl"
    if not out.exists():
        model, rope = ARMS[arm]
        tmp = out.with_suffix(".partial")
        with (SCORES / f"{arm}-{key}.log").open("w") as log:
            subprocess.run([str(SCORER), str(WORK / model), str(case_file(key)), str(tmp), "99", str(NORM_VOCAB), rope],
                           stderr=log, check=True)
        tmp.rename(out)
    rows = {r["id"]: r["values"] for r in map(json.loads, out.read_text().splitlines())}
    expected = [line.split("\t", 1)[0] for line in case_file(key).read_text().splitlines()]
    assert list(rows) == expected, (arm, key)
    return rows


def book_means(rows: dict, band: slice = slice(0, 960)) -> dict[str, float]:
    by = defaultdict(list)
    for case, values in rows.items():
        assert len(values) == 960
        part = values[band]
        by[case.rsplit("-", 1)[0]].append(sum(part) / len(part))
    assert all(len(v) == 2 for v in by.values())
    return {b: sum(v) / 2 for b, v in by.items()}


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs)


def t_interval(diffs: list[float]) -> dict:
    n, d = len(diffs), mean(diffs)
    sd = math.sqrt(sum((x - d) ** 2 for x in diffs) / (n - 1))
    half = float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    return {"mean": d, "lo": d - half, "hi": d + half, "n": n}


def validation() -> None:
    check_builds()
    for arm in ARMS:
        print(f"{arm}: validation primary NLL {mean(book_means(score(arm, 'validation')).values()):.5f}", flush=True)


def select() -> None:
    if (OUT / "selection.json").exists():
        raise SystemExit("selection already written")
    check_builds()
    assert not any(SCORES.glob("*-heldout.jsonl")) and not any(SCORES.glob("*-mmlu.jsonl")) and \
        not any(SCORES.glob("*-retrieval.jsonl")) and not (OUT / "gsm8k").exists()
    v = {arm: mean(book_means(score(arm, "validation")).values()) for arm in ARMS}
    method = min((v[a], a) for a in ("gptq", "gptqh"))[1]
    reference = min((v[a], a) for a in ("prism", "prism_noyarn"))[1]
    record = {"selected_method": method, "prism_reference": reference, "validation_primary_nll": v,
              "rule": protocol()["selection"], "protocol_sha256": sha(OUT / "protocol.json"),
              "scorer_sha256": sha(SCORER), "builds_sha256": {a: sha(OUT / f"build_{a}.json") for a in BUILT},
              "heldout_scored_before_selection": False}
    (OUT / "selection.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"selected method {method}; Prism reference {reference}")


def selection() -> dict:
    s = json.loads((OUT / "selection.json").read_text())
    assert s["protocol_sha256"] == sha(OUT / "protocol.json")
    return s


def heldout() -> None:
    selection()
    for key in ("heldout", "mmlu", "retrieval"):
        for arm in ARMS:
            score(arm, key)
            print(f"{arm}: {key} scored", flush=True)


def gsm8k() -> None:
    s = selection()
    p = protocol()
    path = OUT / p["gsm8k"]["file"]
    assert sha(path) == p["gsm8k"]["sha256"]
    items = [json.loads(line) for line in path.read_text().splitlines()]
    (OUT / "gsm8k").mkdir(exist_ok=True)
    for arm in ("fp", s["prism_reference"], "gptq", "gptqh"):
        target = OUT / "gsm8k" / f"{arm}.jsonl"
        if target.exists():
            continue
        model, rope = ARMS[arm]
        with gsm8k_17b.serve(WORK / model, rope, SCORES / f"{arm}-gsm8k-server.log"):
            outs = gsm8k_17b.generate(items)
        rows = []
        for it, o in zip(items, outs):
            pred = gsm8k_17b.parse(o["text"])
            rows.append({**o, "gold": it["gold"], "pred": pred, "correct": gsm8k_17b.correct(pred, it["gold"])})
        target.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"{arm}: GSM8K accuracy {mean(r['correct'] for r in rows):.4f}", flush=True)


def paired_items(a: list[float], b: list[float]) -> dict:
    diffs = [x - y for x, y in zip(a, b)]
    n, d = len(diffs), mean(diffs)
    se = math.sqrt(sum((x - d) ** 2 for x in diffs) / (n - 1) / n)
    return {"mean": d, "lo": d - 1.96 * se, "hi": d + 1.96 * se, "n": n}


def analyze() -> None:
    p, s = protocol(), selection()
    sel, ref = s["selected_method"], s["prism_reference"]
    arms = list(ARMS)
    # books
    books = {a: score(a, "heldout") for a in arms}
    primary = {a: book_means(books[a]) for a in arms}
    bands = {a: {k: mean(book_means(books[a], sl).values()) for k, sl in BANDS.items()} for a in arms}
    names = list(primary["fp"])
    book_vs = lambda a, b: t_interval([primary[a][k] - primary[b][k] for k in names])  # noqa: E731
    # MMLU
    mmlu = {a: score(a, "mmlu") for a in arms}
    def mmlu_item(values, case):
        gold = "ABCD".index(case.rsplit("/", 1)[1])
        wrong = max(v for i, v in enumerate(values) if i != gold)
        return float(values[gold] > wrong), values[gold] - wrong
    mmlu_rows = {a: [mmlu_item(v, c) for c, v in mmlu[a].items()] for a in arms}
    mmlu_acc = {a: 100 * mean(r[0] for r in mmlu_rows[a]) for a in arms}
    mmlu_margin = {a: mean(r[1] for r in mmlu_rows[a]) for a in arms}
    # retrieval
    retr = {a: score(a, "retrieval") for a in arms}
    def margins(rows):
        by = defaultdict(dict)
        for case, values in rows.items():
            reg, label, _ = case.split("/")
            by[reg][label] = sum(values)
        return {r: v["wrong"] - v["gold"] for r, v in sorted(by.items())}
    retr_m = {a: margins(retr[a]) for a in arms}
    regs = list(retr_m["fp"])
    # GSM8K
    gsm = {}
    for f in sorted((OUT / "gsm8k").glob("*.jsonl")):
        gsm[f.stem] = [json.loads(line) for line in f.read_text().splitlines()]
    gsm_acc = {a: 100 * mean(r["correct"] for r in rows) for a, rows in gsm.items()}
    gsm_trunc = {a: sum(r["stop_type"] == "limit" for r in rows) for a, rows in gsm.items()}

    def vs(a: str, b: str) -> dict:
        out = {"book_nll": book_vs(a, b),
               "mmlu_accuracy_points": paired_items([100 * r[0] for r in mmlu_rows[a]], [100 * r[0] for r in mmlu_rows[b]]),
               "mmlu_margin": paired_items([r[1] for r in mmlu_rows[a]], [r[1] for r in mmlu_rows[b]]),
               "retrieval_margin": t_interval([retr_m[a][r] - retr_m[b][r] for r in regs])}
        if a in gsm and b in gsm:
            out["gsm8k_accuracy_points"] = paired_items([100 * r["correct"] for r in gsm[a]],
                                                        [100 * r["correct"] for r in gsm[b]])
        return out

    primary_cmp = vs(sel, ref)
    margins_pass = {"book_nll": primary_cmp["book_nll"]["hi"] <= 0.10,
                    "mmlu_accuracy": primary_cmp["mmlu_accuracy_points"]["lo"] >= -3.0,
                    "gsm8k_accuracy": primary_cmp["gsm8k_accuracy_points"]["lo"] >= -3.0,
                    "retrieval_margin": primary_cmp["retrieval_margin"]["lo"] >= -0.25}
    gap = primary_cmp["book_nll"]["mean"]
    reading = "near_prism" if gap <= 0.10 else "moderate_gap" if gap <= 0.50 else "large_gap"
    ls_ref = book_vs("ls", ref)
    closure = None
    if ls_ref["lo"] > 0 or ls_ref["hi"] < 0:
        closure = (mean(primary["ls"].values()) - mean(primary[sel].values())) / ls_ref["mean"]
    result = {
        "protocol_sha256": sha(OUT / "protocol.json"), "selection_sha256": sha(OUT / "selection.json"),
        "selected_method": sel, "prism_reference": ref,
        "book_primary_nll": {a: mean(primary[a].values()) for a in arms}, "book_band_nll": bands,
        "book_nll_by_book": primary,
        "mmlu_accuracy": mmlu_acc, "mmlu_margin": mmlu_margin,
        "retrieval_margin": {a: mean(retr_m[a].values()) for a in arms},
        "retrieval_accuracy": {a: 100 * mean(float(m > 0) for m in retr_m[a].values()) for a in arms},
        "gsm8k_accuracy": gsm_acc, "gsm8k_truncated": gsm_trunc,
        "primary_selected_minus_reference": primary_cmp, "non_inferiority_margins_pass": margins_pass,
        "non_inferior": all(margins_pass.values()), "reading": reading,
        "selected_minus_fp": vs(sel, "fp"), "reference_minus_fp": vs(ref, "fp"),
        "other_method_minus_reference": vs("gptqh" if sel == "gptq" else "gptq", ref),
        "prism_config_effect_noyarn_minus_yarn": vs("prism_noyarn", "prism"),
        "ls_minus_reference_book_nll": ls_ref, "prism_gap_closure_book_nll": closure,
        "harness_check": {"published": {"fp": {"mmlu_redux": 66.8, "gsm8k": 83.1},
                                        "prism": {"mmlu_redux": 52.9, "gsm8k": 74.2}},
                          "note": p["decision"]["harness_check"]},
        "models": {a: {"file": ARMS[a][0], "rope": ARMS[a][1], "sha256": sha(WORK / ARMS[a][0]),
                       "bytes": (WORK / ARMS[a][0]).stat().st_size} for a in arms},
        "claim_limit": p["decision"]["claim_limit"],
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    brief = {k: result[k] for k in ("book_primary_nll", "mmlu_accuracy", "retrieval_margin", "gsm8k_accuracy",
                                    "non_inferiority_margins_pass", "reading", "prism_gap_closure_book_nll")}
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    {"validation": validation, "select": select, "heldout": heldout, "gsm8k": gsm8k,
     "analyze": analyze}[sys.argv[1]]()

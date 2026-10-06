"""Export, score and analyze the frozen 0.8B training test (results/qat08/protocol.json).

    python3 score_qat08.py export       # init + all checkpoints of both arms -> PQ2_0 GGUFs
    python3 score_qat08.py curve        # validation books for every checkpoint (descriptive)
    python3 score_qat08.py heldout      # books, MMLU-Redux, retrieval for the held-out arms
    python3 score_qat08.py gsm8k
    python3 score_qat08.py analyze
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import gsm8k_17b
from score_17b import book_means, mean, paired_items, sha, t_interval

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/qat08"
QAT = ROOT / "work/qat08"
SCORES = QAT / "scores"
SCORER = ROOT / "work/qwen3_17b/score_17b"
NORM_VOCAB = 248077
CKPTS = ("t8m", "t16m", "t33m", "final")
TRAINED = {"qat_free": "absmean", "qat_42": "top86"}


def protocol() -> dict:
    return json.loads((OUT / "protocol.json").read_text())


def model_path(arm: str) -> Path:
    p = protocol()["arms"]
    if arm in ("fp", "gptq"):
        return ROOT / p[arm]["file"]
    if arm.endswith("_init"):
        return QAT / f"{'qat_free' if arm == 'absmean_init' else 'qat_42'}-init.gguf"
    if arm in TRAINED:
        return QAT / f"{arm}-final.gguf"
    return QAT / f"{arm}.gguf"   # e.g. qat_free-t8m


def case_file(key: str) -> Path:
    p = protocol()
    entry = {"validation": p["book_cases"]["validation"], "heldout": p["book_cases"]["heldout"],
             "mmlu": p["mmlu"], "retrieval": p["retrieval"]}[key]
    path = OUT / entry["file"]
    assert sha(path) == entry["sha256"], key
    return path


def score(arm: str, key: str) -> dict[str, list[float]]:
    SCORES.mkdir(parents=True, exist_ok=True)
    out = SCORES / f"{arm}-{key}.jsonl"
    if not out.exists():
        tmp = out.with_suffix(".partial")
        with (SCORES / f"{arm}-{key}.log").open("w") as log:
            subprocess.run([str(SCORER), str(model_path(arm)), str(case_file(key)), str(tmp), "99", str(NORM_VOCAB),
                            "model"], stderr=log, check=True)
        tmp.rename(out)
    rows = {r["id"]: r["values"] for r in map(json.loads, out.read_text().splitlines())}
    assert list(rows) == [line.split("\t", 1)[0] for line in case_file(key).read_text().splitlines()], (arm, key)
    return rows


def export() -> None:
    for arm in TRAINED:
        assert (QAT / arm / "DONE").exists(), f"{arm} has not finished training"
        for ckpt in ("init",) + CKPTS:
            if not (QAT / f"{arm}-{ckpt}.gguf").exists():
                subprocess.run([sys.executable, str(ROOT / "qat_08b.py"), "export", "--arm", arm, "--ckpt", ckpt],
                               cwd=ROOT, check=True)


def curve() -> None:
    arms = ["fp", "gptq"] + [f"{a}-{c}" for a in TRAINED for c in ("init",) + CKPTS]
    res = {}
    for arm in arms:
        res[arm] = mean(book_means(score(arm, "validation")).values())
        print(f"{arm}: validation NLL {res[arm]:.4f}", flush=True)
    (OUT / "curve.json").write_text(json.dumps(res, indent=2) + "\n")


def heldout() -> None:
    for key in ("heldout", "mmlu", "retrieval"):
        for arm in protocol()["heldout_arms"]["books_mmlu_retrieval"]:
            score(arm, key)
            print(f"{arm}: {key} scored", flush=True)


def gsm8k() -> None:
    p = protocol()
    path = OUT / p["gsm8k"]["file"]
    assert sha(path) == p["gsm8k"]["sha256"]
    items = [json.loads(line) for line in path.read_text().splitlines()]
    (OUT / "gsm8k").mkdir(exist_ok=True)
    for arm in p["heldout_arms"]["gsm8k"]:
        target = OUT / "gsm8k" / f"{arm}.jsonl"
        if target.exists():
            continue
        with gsm8k_17b.serve(model_path(arm), "model", SCORES / f"{arm}-gsm8k-server.log"):
            outs = gsm8k_17b.generate(items)
        rows = [{**o, "gold": it["gold"], "pred": gsm8k_17b.parse(o["text"]),
                 "correct": gsm8k_17b.correct(gsm8k_17b.parse(o["text"]), it["gold"])} for it, o in zip(items, outs)]
        target.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"{arm}: GSM8K {mean(r['correct'] for r in rows):.4f}", flush=True)


def analyze() -> None:
    p = protocol()
    arms = p["heldout_arms"]["books_mmlu_retrieval"]
    books = {a: book_means(score(a, "heldout")) for a in arms}
    names = list(books["fp"])
    bvs = lambda a, b: t_interval([books[a][k] - books[b][k] for k in names])  # noqa: E731

    def mmlu_rows(a):
        rows = []
        for case, v in score(a, "mmlu").items():
            gold = "ABCD".index(case.rsplit("/", 1)[1])
            wrong = max(x for i, x in enumerate(v) if i != gold)
            rows.append((100.0 * (v[gold] > wrong), v[gold] - wrong))
        return rows

    mm = {a: mmlu_rows(a) for a in arms}

    def margins(a):
        by = {}
        for case, v in score(a, "retrieval").items():
            reg, label, _ = case.split("/")
            by.setdefault(reg, {})[label] = sum(v)
        return [by[r]["wrong"] - by[r]["gold"] for r in sorted(by)]

    rt = {a: margins(a) for a in arms}
    gsm = {f.stem: [json.loads(l) for l in f.read_text().splitlines()] for f in sorted((OUT / "gsm8k").glob("*.jsonl"))}
    primary = bvs("qat_free", "gptq")
    fp_m, g_m, q_m = (mean(books[a].values()) for a in ("fp", "gptq", "qat_free"))
    closure = (g_m - q_m) / (g_m - fp_m)
    budget = bvs("qat_42", "qat_free")
    curve = json.loads((OUT / "curve.json").read_text())
    tail_drop = curve["qat_free-t33m"] - curve["qat_free-final"]
    helps = primary["mean"] <= -0.25 and primary["hi"] < 0

    def mmlu_acc_ci(a):
        xs = [r[0] for r in mm[a]]
        d = paired_items(xs, [0.0] * len(xs))
        return d

    capability = {a: {"mmlu_lo_above_25": mmlu_acc_ci(a)["lo"] > 25.0,
                      "gsm8k_above_0": a in gsm and mean(r["correct"] for r in gsm[a]) > 0} for a in TRAINED}
    budget_reading = ("costs little" if abs(budget["mean"]) <= 0.05 else "costs" if budget["mean"] > 0 else "helps")
    result = {
        "protocol_sha256": sha(OUT / "protocol.json"),
        "book_nll": {a: mean(books[a].values()) for a in arms}, "book_nll_by_book": books,
        "mmlu_accuracy": {a: mean(r[0] for r in mm[a]) for a in arms},
        "mmlu_margin": {a: mean(r[1] for r in mm[a]) for a in arms},
        "retrieval_margin": {a: mean(rt[a]) for a in arms},
        "retrieval_accuracy": {a: 100 * mean(float(x > 0) for x in rt[a]) for a in arms},
        "gsm8k_accuracy": {a: 100 * mean(r["correct"] for r in rows) for a, rows in gsm.items()},
        "gsm8k_truncated": {a: sum(r["stop_type"] == "limit" for r in rows) for a, rows in gsm.items()},
        "primary_qat_free_minus_gptq": primary, "training_helps": helps, "gap_closure": closure,
        "budget_qat42_minus_qatfree": {**budget, "reading": budget_reading},
        "qat_free_minus_fp": bvs("qat_free", "fp"), "qat_42_minus_fp": bvs("qat_42", "fp"),
        "init_effects": {"absmean_init_minus_gptq": bvs("absmean_init", "gptq"),
                         "top86_init_minus_gptq": bvs("top86_init", "gptq")},
        "mmlu_vs_fp": {a: paired_items([r[0] for r in mm[a]], [r[0] for r in mm["fp"]]) for a in arms if a != "fp"},
        "capability": capability, "validation_curve": curve, "qat_free_t33m_to_final_drop": tail_drop,
        "recommend_1p7b_rental": helps and (closure >= 0.5 or tail_drop >= 0.02),
        "models": {a: {"file": str(model_path(a).relative_to(ROOT)), "sha256": sha(model_path(a))} for a in arms},
        "claim_limit": p["decision"]["claim_limit"],
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("book_nll", "mmlu_accuracy", "retrieval_accuracy", "gsm8k_accuracy",
                                             "primary_qat_free_minus_gptq", "training_helps", "gap_closure",
                                             "budget_qat42_minus_qatfree", "capability", "recommend_1p7b_rental")},
                     indent=2))


if __name__ == "__main__":
    {"export": export, "curve": curve, "heldout": heldout, "gsm8k": gsm8k, "analyze": analyze}[sys.argv[1]]()

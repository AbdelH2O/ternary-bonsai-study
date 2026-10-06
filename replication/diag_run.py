"""Stages of the read-out diagnostics (results/diag_readout/plan.json).

GPU (unsandboxed systemd unit `diag-readout`, require approval.json): score, generate, kl.
CPU: analyze, decide (exclusive decision.json), report (DIAG_READOUT.md). Every GPU stage resumes.
"""
from __future__ import annotations

import concurrent.futures
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from diag_cases import CASES, LETTER_IDS, OUT, PREFIX, SUFFIX, WORK, verify
from diag_metrics import binding_metrics, branch, cloze_metrics, letter_metrics, parse_mc_answer, regions, summary, wilson
from score_17b import paired_items, sha

ROOT = Path(__file__).resolve().parent
SCORER = ROOT / "work/qwen3_17b/score_17b"
SCORES = WORK / "scores"
SHARD = 512


def approved() -> dict:
    plan = verify()
    approval = json.loads((OUT / "approval.json").read_text())
    assert approval["approved_by_user"] is True and approval["plan_sha256"] == sha(OUT / "plan.json"), \
        "GPU stages need the user's approval bound to plan.json"
    return plan


def timed(stage: str):
    """Context manager recording stage seconds and enforcing the frozen 2.0 GPU-hour ceiling."""
    import contextlib

    @contextlib.contextmanager
    def run():
        log = WORK / "timing.jsonl"
        WORK.mkdir(parents=True, exist_ok=True)
        used = sum(json.loads(l)["seconds"] for l in log.read_text().splitlines()) if log.exists() else 0.0
        budget = approved()["budget_gpu_hours"] * 3600
        assert used < budget, f"diagnostic GPU ceiling reached ({used/3600:.2f} h)"
        t0 = time.time()
        try:
            yield
        finally:
            with log.open("a") as f:
                f.write(json.dumps({"stage": stage, "seconds": time.time() - t0, "time": time.time()}) + "\n")
    return run()


def checked_rows(path: Path, expected: list[str]) -> dict[str, list[float]]:
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert [r["id"] for r in rows] == expected, path
    assert all(all(math.isfinite(x) for x in r["values"]) for r in rows), path
    return {r["id"]: r["values"] for r in rows}


def score_cases(model: Path, cases: Path, target: Path) -> dict[str, list[float]]:
    lines = cases.read_text().splitlines()
    ids = [l.split("\t", 1)[0] for l in lines]
    if not target.exists():
        shards = target.with_suffix("")
        shards.mkdir(parents=True, exist_ok=True)
        pieces = []
        for start in range(0, len(lines), SHARD):
            piece = shards / f"{start:06d}.jsonl"
            expected = ids[start:start + SHARD]
            if not piece.exists():
                subset = shards / f"{start:06d}.tsv"
                subset.write_text("\n".join(lines[start:start + SHARD]) + "\n")
                tmp = piece.with_suffix(".partial")
                with piece.with_suffix(".log").open("w") as log:
                    subprocess.run([str(SCORER), str(model), str(subset), str(tmp), "99", "248077", "model"],
                                   stderr=log, check=True)
                checked_rows(tmp, expected)
                tmp.replace(piece)
            checked_rows(piece, expected)
            pieces.append(piece)
        tmp = target.with_suffix(".partial")
        tmp.write_text("".join(p.read_text() for p in pieces))
        tmp.replace(target)
    return checked_rows(target, ids)


def score() -> None:
    plan = approved()
    SCORES.mkdir(parents=True, exist_ok=True)
    with timed("score"):
        for key in plan["sets"]:
            for arm, entry in plan["arms"].items():
                score_cases(ROOT / entry["file"], OUT / plan["files"][key]["file"], SCORES / f"{arm}-{key}.jsonl")
                print(f"{arm} {key} scored", flush=True)


def generate() -> None:
    import gsm8k_17b as gen
    plan = approved()
    items = [json.loads(l) for l in (OUT / plan["files"]["generative"]["file"]).read_text().splitlines()]
    directory = OUT / "generative"
    directory.mkdir(exist_ok=True)
    gen.PORT, gen.N_PREDICT = 8093, 1024
    with timed("generate"):
        for arm in plan["generative_arms"]:
            target = directory / f"{arm}.jsonl"
            if target.exists():
                continue
            journal = directory / f"{arm}.partial.jsonl"
            done = {r["id"]: r for r in map(json.loads, journal.read_text().splitlines())} if journal.exists() else {}
            pending = [it for it in items if it["id"] not in done]
            if pending:
                with gen.serve(ROOT / plan["arms"][arm]["file"], "model", WORK / f"{arm}-server.log"):
                    with journal.open("a", buffering=1) as f, concurrent.futures.ThreadPoolExecutor(gen.SLOTS) as pool:
                        for it, out in zip(pending, pool.map(lambda it: gen.complete(it["prompt_ids"]), pending)):
                            pred = parse_mc_answer(out["text"])
                            row = {"id": it["id"], **out, "gold": it["gold"], "pred": pred, "correct": pred == it["gold"]}
                            f.write(json.dumps(row) + "\n")
                            done[it["id"]] = row
            assert len(done) == len(items)
            tmp = target.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(done[it["id"]]) + "\n" for it in items))
            tmp.replace(target)


def position_kl(hs, ht, es, et, letter_ids: list[int], chunk: int = 64):
    """Per-position KL(teacher||student) over the full vocabulary in position chunks (bounded memory),
    plus the 4-letter renormalized KL at the last position. Same arithmetic as qat_08b.kl_chunk."""
    import torch
    per = []
    for i in range(0, hs.shape[0], chunk):
        ls = (hs[i:i + chunk].to(es.dtype) @ es.T).float().log_softmax(-1)
        lt = (ht[i:i + chunk].to(et.dtype) @ et.T).float().log_softmax(-1)
        per.append((lt.exp() * (lt - ls)).sum(-1))
    s4, t4 = ls[-1, letter_ids].log_softmax(-1), lt[-1, letter_ids].log_softmax(-1)
    return torch.cat(per), (t4.exp() * (t4 - s4)).sum().item()


def kl() -> None:
    import torch
    import qat_08b as qat
    plan = approved()
    cases = {l.split("\t")[0]: list(map(int, l.split("\t")[2].split()))
             for l in (OUT / plan["files"]["letter"]["file"]).read_text().splitlines()}
    directory = OUT / "kl"
    directory.mkdir(exist_ok=True)
    with timed("kl"):
        teacher = qat.load_teacher()
        et = teacher.lm_head.weight
        for arm, spec in plan["kl_arms"].items():
            target = directory / f"{arm}.jsonl"
            if target.exists():
                continue
            latents = torch.load(ROOT / spec["latents"], map_location="cpu")
            student, _, _ = qat.build_student(spec["rule"], latents)
            student.eval()
            es = student.lm_head.weight
            rows = []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for cid in plan["kl_ids"]:
                    ids = cases[cid]
                    assert ids[:3] == PREFIX and ids[-9:] == SUFFIX, cid
                    x = torch.tensor([ids], device=qat.DEV)
                    hs = student.model(input_ids=x).last_hidden_state[0]
                    ht = teacher.model(input_ids=x).last_hidden_state[0]
                    per, letter4 = position_kl(hs.to(torch.bfloat16), ht.to(torch.bfloat16), es, et, LETTER_IDS)
                    rec = {"id": cid, **{k: per[a:b].mean().item() for k, (a, b) in regions(len(ids), len(PREFIX), len(SUFFIX)).items()}}
                    rec["letter4"] = letter4
                    rows.append(rec)
            tmp = target.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
            tmp.replace(target)
            del student
            torch.cuda.empty_cache()


def _load(path: Path) -> dict[str, list[float]]:
    return {r["id"]: r["values"] for r in map(json.loads, path.read_text().splitlines())}


def analyze(plan: dict | None = None, write: bool = True) -> dict:
    plan = plan or verify()
    metric = {"letter": letter_metrics, "binding": binding_metrics, "cloze_chat": cloze_metrics, "cloze_raw": cloze_metrics}
    full = {arm: {key: metric[key](_load(SCORES / f"{arm}-{key}.jsonl")) for key in plan["sets"]} for arm in plan["arms"]}
    res = {"plan_sha256": sha(OUT / "plan.json") if write else None,
           "arms": {arm: {key: summary(m) for key, m in sets.items()} for arm, sets in full.items()},
           "trajectory": {}, "generative": {}, "kl": {}}
    for run, ckpts in plan["trajectory"].items():
        first, last = ckpts[0], ckpts[-1]
        res["trajectory"][run] = {
            "points": {c: {k: full[c][k]["pairwise"] for k in ("letter", "cloze_chat", "cloze_raw", "binding")} for c in ckpts},
            "final_minus_t8m": {k: paired_items([100*x for x in full[last][k]["pairwise_items"]],
                                                [100*x for x in full[first][k]["pairwise_items"]])
                                for k in ("letter", "cloze_chat", "cloze_raw")}}
    for arm in plan["generative_arms"]:
        rows = [json.loads(l) for l in (OUT / "generative" / f"{arm}.jsonl").read_text().splitlines()]
        res["generative"][arm] = {"accuracy": wilson(sum(r["correct"] for r in rows), len(rows)),
                                  "parsed": sum(r["pred"] is not None for r in rows), "items": len(rows)}
    for arm in plan["kl_arms"]:
        rows = [json.loads(l) for l in (OUT / "kl" / f"{arm}.jsonl").read_text().splitlines()]
        res["kl"][arm] = {k: sum(r[k] for r in rows) / len(rows) for k in ("question", "template", "letter", "letter4")}
    if write:
        tmp = OUT / "results.tmp"
        tmp.write_text(json.dumps(res, indent=2) + "\n")
        tmp.replace(OUT / "results.json")
    return res


def decide() -> None:
    plan = verify()
    res = json.loads((OUT / "results.json").read_text())
    assert res["plan_sha256"] == sha(OUT / "plan.json")
    decision = {**branch(res, plan["rule"]), "results_sha256": sha(OUT / "results.json"),
                "plan_sha256": sha(OUT / "plan.json"), "rule": plan["rule"], "rule_text": plan["rule_text"]}
    with (OUT / "decision.json").open("x") as f:
        json.dump(decision, f, indent=2)
        f.write("\n")
    print(json.dumps(decision, indent=2))


def report() -> None:
    res = json.loads((OUT / "results.json").read_text())
    dec = json.loads((OUT / "decision.json").read_text())
    lines = ["# Read-out diagnostics for the QAT08 chat student — results", "",
             f"**Decision (predeclared rule): `{dec['branch']}`; next experiment arms {dec['arms_required']}"
             f"{' + optional ' + str(dec['arms_optional']) if dec['arms_optional'] else ''}.**", "",
             *[f"- {r}" for r in dec["reasons"]], "",
             "Spec: [review](reviews/REVIEW_QAT08_CHAT_opus55.md) sections 4-6; frozen [plan](results/diag_readout/plan.json); "
             "[results](results/diag_readout/results.json); [decision](results/diag_readout/decision.json). "
             "Development data only (MMLU validation, synthetic registries). Evidence tags: 🟢 measured; 🟡 inferred; 🟣 proposed.", "",
             "## 🟢 Per-arm signal (pairwise gold-vs-wrong %, 50 = no information)", "",
             "| Arm | Letter raw acc | Letter calibrated acc | Letter pairwise [95%] | Letter mass | Cloze chat pairwise | Cloze raw pairwise | Binding calibrated acc |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, s in res["arms"].items():
        l, cc, cr, b = s["letter"], s["cloze_chat"], s["cloze_raw"], s["binding"]
        lines.append(f"| {arm} | {l['raw_accuracy']:.1f} | {l['calibrated_accuracy']:.1f} | {l['pairwise']:.1f} "
                     f"[{l['pairwise_lo']:.1f}, {l['pairwise_hi']:.1f}] | {l['letter_mass']:.3f} | {cc['pairwise']:.1f} "
                     f"[{cc['pairwise_lo']:.1f}, {cc['pairwise_hi']:.1f}] | {cr['pairwise']:.1f} [{cr['pairwise_lo']:.1f}, "
                     f"{cr['pairwise_hi']:.1f}] | {b['calibrated_accuracy']:.1f} |")
    lines += ["", "## 🟢 Trajectory (final minus t8m, pairwise points, paired item interval)", "",
              "| Run | Letter | Cloze chat | Cloze raw |", "|---|---:|---:|---:|"]
    for run, t in res["trajectory"].items():
        d = t["final_minus_t8m"]
        lines.append(f"| {run} | " + " | ".join(f"{d[k]['mean']:+.2f} [{d[k]['lo']:+.2f}, {d[k]['hi']:+.2f}]"
                                                for k in ("letter", "cloze_chat", "cloze_raw")) + " |")
    lines += ["", "## 🟢 Generative multiple choice and positional KL", "",
              "| Arm | Generative acc [95%] | Parsed |", "|---|---:|---:|"]
    for arm, g in res["generative"].items():
        a = g["accuracy"]
        lines.append(f"| {arm} | {a['mean']:.1f} [{a['lo']:.1f}, {a['hi']:.1f}] | {g['parsed']}/{g['items']} |")
    lines += ["", "| Arm | KL question | KL template | KL answer position | 4-letter KL |", "|---|---:|---:|---:|---:|"]
    for arm, k in res["kl"].items():
        lines.append(f"| {arm} | {k['question']:.3f} | {k['template']:.3f} | {k['letter']:.3f} | {k['letter4']:.3f} |")
    lines += ["", "## Interpretation", "", "🟡 (Written by the executing agent from the tables above; inferred, not measured.)", ""]
    (ROOT / "DIAG_READOUT.md").write_text("\n".join(lines) + "\n")
    print("wrote DIAG_READOUT.md")


if __name__ == "__main__":
    {"score": score, "generate": generate, "kl": kl, "analyze": analyze, "decide": decide, "report": report}[sys.argv[1]]()

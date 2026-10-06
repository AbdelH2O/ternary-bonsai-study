"""Score, select and analyze the frozen GPTQ amendment of the 0.8B pilot.

Run from the repository root, in order (CPU scorer, same as the original pilot):
    python3 analysis/bonsai2/replication/score_gptq_pilot.py validation
    python3 analysis/bonsai2/replication/score_gptq_pilot.py select
    python3 analysis/bonsai2/replication/score_gptq_pilot.py heldout
    python3 analysis/bonsai2/replication/score_gptq_pilot.py analyze
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PILOT = ROOT / "results/quality_pilot"
OUT = ROOT / "results/gptq_pilot"
WORK = ROOT / "work/qwen35_08b"
SCORER = WORK / "score_quality_pilot"
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(REPO))


def model(arm: str) -> Path:
    return WORK / f"{arm.replace('-absmax', '')}.gguf"


def slices(split: str) -> list[tuple[str, str, Path]]:
    """(case id, book, slice path) for 'validation' or 'heldout' (old + new held-out books)."""
    prior = json.loads((PILOT / "protocol.json").read_text())
    new = json.loads((OUT / "protocol.json").read_text())
    rows = []
    for e in prior["entries"]:
        if e["split"] == split:
            assert sha(PILOT / e["text_file"]) == e["text_sha256"]
            rows.append((e["id"], e["book"], PILOT / e["text_file"]))
    if split == "heldout":
        for e in new["new_heldout_entries"]:
            assert sha(OUT / e["text_file"]) == e["text_sha256"]
            assert sha(REPO / e["source_path"]) == e["source_sha256"]
            rows.append((e["id"], e["book"], OUT / e["text_file"]))
    return rows


def score(arm: str, split: str) -> dict[str, dict]:
    suffix = {"validation": "gptq-quality", "heldout": "gptq-heldout"}[split]
    out, log = WORK / f"{arm}-{suffix}.jsonl", WORK / f"{arm}-{suffix}.log"
    cases = slices(split)
    if not out.exists():
        with out.open("w") as o, log.open("w") as e:
            subprocess.run([str(SCORER), rel(model(arm))] + [rel(p) for _, _, p in cases],
                           cwd=REPO, stdout=o, stderr=e, check=True)
    rows = {Path(r["file"]).stem: r for r in map(json.loads, out.read_text().splitlines())}
    assert set(rows) == {c for c, _, _ in cases}, (arm, split)
    return rows


def book_means(rows: dict[str, dict], split: str) -> dict[str, float]:
    by = defaultdict(list)
    for case, book, _ in slices(split):
        by[book].append(rows[case]["nll"])
    assert all(len(v) == 2 for v in by.values())
    return {b: sum(v) / 2 for b, v in by.items()}


def validation() -> None:
    protocol = json.loads((OUT / "protocol.json").read_text())
    for arm in protocol["arms"]:
        build = json.loads((OUT / f"build_{arm.split('-')[0]}.json").read_text())
        assert build["sha256"] == sha(model(arm)) and build["protocol_sha256"] == sha(OUT / "protocol.json")
        means = book_means(score(arm, "validation"), "validation")
        print(f"{arm}: validation mean NLL {sum(means.values()) / len(means):.5f}")


def select() -> None:
    if (OUT / "selection.json").exists():
        raise SystemExit("selection already written")
    protocol = json.loads((OUT / "protocol.json").read_text())
    prior = json.loads((PILOT / "selection.json").read_text())
    assert sha(PILOT / "selection.json") == protocol["amends"]["selection_sha256"]
    scores = dict(prior["validation_mean_nll"])
    for arm in protocol["arms"]:
        means = book_means(score(arm, "validation"), "validation")
        scores[arm] = sum(means.values()) / len(means)
    packed = sorted((v, a) for a, v in scores.items() if "pq2" in a)
    best_gptq = min((scores[a], a) for a in protocol["arms"])[1]
    selection = {"selected_arm": packed[0][1], "best_gptq_arm": best_gptq,
                 "rule": protocol["selection"]["rule"], "validation_mean_nll": scores,
                 "heldout_arms": ["base-f32", "base-pq2-ls", best_gptq],
                 "protocol_sha256": sha(OUT / "protocol.json"),
                 "builder_sha256": sha(ROOT / "make_gptq_pilot.py"),
                 "scorer_source_sha256": sha(ROOT / "score_quality_pilot.cpp"),
                 "scorer_binary_sha256": sha(SCORER),
                 "build_records_sha256": {b: sha(OUT / f"build_{b}.json") for b in ("base", "folded")},
                 "new_heldout_books_scored_before_selection": False}
    (OUT / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(f"selected {selection['selected_arm']}; best GPTQ arm {best_gptq}")


def heldout() -> None:
    selection = json.loads((OUT / "selection.json").read_text())
    assert selection["protocol_sha256"] == sha(OUT / "protocol.json")
    for arm in selection["heldout_arms"]:
        means = book_means(score(arm, "heldout"), "heldout")
        print(f"{arm}: held-out mean NLL {sum(means.values()) / len(means):.5f}")


def analyze() -> None:
    protocol = json.loads((OUT / "protocol.json").read_text())
    selection = json.loads((OUT / "selection.json").read_text())
    assert selection["protocol_sha256"] == sha(OUT / "protocol.json")
    fp, ls, gq = selection["heldout_arms"]
    rows = {arm: score(arm, "heldout") for arm in (fp, ls, gq)}
    cases = slices("heldout")
    for case, _, _ in cases:
        assert len({tuple(rows[a][case]["tokens"]) for a in rows}) == 1
    means = {arm: book_means(rows[arm], "heldout") for arm in rows}
    books = list(means[fp])
    mean = {arm: sum(means[arm].values()) / len(books) for arm in rows}
    closure = {b: (means[ls][b] - means[gq][b]) / (means[ls][b] - means[fp][b]) for b in books}
    c = (mean[ls] - mean[gq]) / (mean[ls] - mean[fp])
    diffs = [means[gq][b] - means[ls][b] for b in books]
    d = sum(diffs) / len(diffs)
    sd = math.sqrt(sum((x - d) ** 2 for x in diffs) / (len(diffs) - 1))
    half = T975[len(diffs) - 1] * sd / math.sqrt(len(diffs))
    # Reproducibility: re-scored old held-out slices must equal the original pilot's saved scores.
    repro = {}
    for arm in (fp, ls):
        old = {Path(r["file"]).stem: r["nll"] for r in
               map(json.loads, (WORK / f"{arm}-heldout.jsonl").read_text().splitlines())}
        repro[arm] = max(abs(old[k] - rows[arm][k]["nll"]) for k in old)
    gap = mean[gq] - mean[fp]
    if c >= 0.5:
        decision = "rule_1"
    elif gap > 1.0:
        decision = "rule_2"
    else:
        decision = "otherwise"
    result = {"protocol_sha256": selection["protocol_sha256"], "selection_sha256": sha(OUT / "selection.json"),
              "selected_arm": selection["selected_arm"], "compared_gptq_arm": gq,
              "heldout_books": books, "book_mean_nll": means, "mean_nll": mean,
              "gptq_minus_fp": gap, "ls_minus_fp": mean[ls] - mean[fp], "gptq_minus_ls": d,
              "gptq_minus_ls_t95_descriptive": [d - half, d + half],
              "gap_closure_C": c, "gap_closure_by_book": closure,
              "decision": decision, "decision_text": protocol["decision"][decision],
              "reproducibility_max_abs_diff_vs_original_heldout": repro,
              "models": {a: {"sha256": sha(model(a)), "bytes": model(a).stat().st_size} for a in rows},
              "byte_ceiling_pass": model(gq).stat().st_size <= protocol["byte_ceiling"],
              "paired_token_ids_identical": True,
              "claim_limit": protocol["decision"]["claim_limit"]}
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("mean_nll", "gap_closure_C", "gptq_minus_ls",
                                             "gptq_minus_ls_t95_descriptive", "decision",
                                             "reproducibility_max_abs_diff_vs_original_heldout")}, indent=2))


if __name__ == "__main__":
    {"validation": validation, "select": select, "heldout": heldout, "analyze": analyze}[sys.argv[1]]()

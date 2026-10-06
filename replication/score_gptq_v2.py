"""Score, select and analyze the frozen refinement amendment (results/gptq_v2/protocol.json).

Run from the repository root, in order (unchanged CPU scorer):
    python3 analysis/bonsai2/replication/score_gptq_v2.py validation
    python3 analysis/bonsai2/replication/score_gptq_v2.py select
    python3 analysis/bonsai2/replication/score_gptq_v2.py heldout
    python3 analysis/bonsai2/replication/score_gptq_v2.py analyze
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
OUT = ROOT / "results/gptq_v2"
WORK = ROOT / "work/qwen35_08b"
SCORER = WORK / "score_quality_pilot"
GPTQ_ARMS = ["folded-pq2-gptq", "folded-pq2-gptq42", "folded-pq2-gptqv2", "folded-pq2-gptq42v2"]
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def protocol() -> dict:
    return json.loads((OUT / "protocol.json").read_text())


def cases(split: str) -> list[tuple[str, str, Path]]:
    rows = []
    for e in protocol()["entries"]:
        if e["split"] == split:
            path = OUT / e["text_file"]
            assert sha(path) == e["text_sha256"]
            assert sha(OUT / "books" / e["book"]) == e["body_sha256"]
            rows.append((e["id"], e["book"], path))
    return rows


def score(arm: str, split: str) -> dict[str, dict]:
    out, log = WORK / f"{arm}-v2-{split}.jsonl", WORK / f"{arm}-v2-{split}.log"
    todo = cases(split)
    if not out.exists():
        with out.open("w") as o, log.open("w") as e:
            subprocess.run([str(SCORER), str((WORK / f"{arm}.gguf").relative_to(REPO))] +
                           [str(p.relative_to(REPO)) for _, _, p in todo], cwd=REPO, stdout=o, stderr=e, check=True)
    rows = {Path(r["file"]).stem: r for r in map(json.loads, out.read_text().splitlines())}
    assert set(rows) == {c for c, _, _ in todo}, (arm, split)
    return rows


def book_means(rows: dict, split: str) -> dict[str, float]:
    by = defaultdict(list)
    for case, book, _ in cases(split):
        by[book].append(rows[case]["nll"])
    assert all(len(v) == 2 for v in by.values())
    return {b: sum(v) / 2 for b, v in by.items()}


def mean(d: dict) -> float:
    return sum(d.values()) / len(d)


def check_builds() -> None:
    for arm in GPTQ_ARMS[1:]:
        build = json.loads((OUT / f"build_{arm}.json").read_text())
        assert build["protocol_sha256"] == sha(OUT / "protocol.json")
        assert build["sha256"] == sha(WORK / f"{arm}.gguf")
        assert build["bytes"] <= protocol()["byte_ceiling"]
    v1 = json.loads((ROOT / "results/gptq_pilot/build_folded.json").read_text())
    assert v1["sha256"] == sha(WORK / "folded-pq2-gptq.gguf")


def validation() -> None:
    check_builds()
    for arm in ["base-f32"] + GPTQ_ARMS:
        print(f"{arm}: validation mean NLL {mean(book_means(score(arm, 'validation'), 'validation')):.5f}")


def select() -> None:
    if (OUT / "selection.json").exists():
        raise SystemExit("selection already written")
    check_builds()
    scores = {arm: mean(book_means(score(arm, "validation"), "validation")) for arm in ["base-f32"] + GPTQ_ARMS}
    selected = min((scores[a], a) for a in GPTQ_ARMS)[1]
    record = {"selected_arm": selected, "rule": protocol()["selection"], "validation_mean_nll": scores,
              "protocol_sha256": sha(OUT / "protocol.json"), "builder_sha256": sha(ROOT / "make_gptq_v2.py"),
              "scorer_binary_sha256": sha(SCORER),
              "build_records_sha256": {a: sha(OUT / f"build_{a}.json") for a in GPTQ_ARMS[1:]},
              "heldout_scored_before_selection": any((WORK / f"{a}-v2-heldout.jsonl").exists()
                                                    for a in ["base-f32"] + GPTQ_ARMS)}
    assert not record["heldout_scored_before_selection"]
    (OUT / "selection.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"selected {selected}")


def heldout() -> None:
    selection = json.loads((OUT / "selection.json").read_text())
    assert selection["protocol_sha256"] == sha(OUT / "protocol.json")
    for arm in protocol()["heldout"]["arms"]:
        print(f"{arm}: held-out mean NLL {mean(book_means(score(arm, 'heldout'), 'heldout')):.5f}")


def interval(diffs: list[float]) -> tuple[float, float, float]:
    d = sum(diffs) / len(diffs)
    sd = math.sqrt(sum((x - d) ** 2 for x in diffs) / (len(diffs) - 1))
    half = T975[len(diffs) - 1] * sd / math.sqrt(len(diffs))
    return d, d - half, d + half


def analyze() -> None:
    p = protocol()
    selection = json.loads((OUT / "selection.json").read_text())
    assert selection["protocol_sha256"] == sha(OUT / "protocol.json")
    arms = p["heldout"]["arms"]
    rows = {a: score(a, "heldout") for a in arms}
    for case, _, _ in cases("heldout"):
        assert len({tuple(rows[a][case]["tokens"]) for a in arms}) == 1
    means = {a: book_means(rows[a], "heldout") for a in arms}
    books = list(means["base-f32"])
    sel, ref = selection["selected_arm"], "folded-pq2-gptq"
    primary = interval([means[sel][b] - means[ref][b] for b in books])
    budget_diffs = [((means["folded-pq2-gptq42"][b] - means["folded-pq2-gptq"][b]) +
                     (means["folded-pq2-gptq42v2"][b] - means["folded-pq2-gptqv2"][b])) / 2 for b in books]
    recon_diffs = [((means["folded-pq2-gptqv2"][b] - means["folded-pq2-gptq"][b]) +
                    (means["folded-pq2-gptq42v2"][b] - means["folded-pq2-gptq42"][b])) / 2 for b in books]
    budget, recon = interval(budget_diffs), interval(recon_diffs)
    best = min((mean(means[a]), a) for a in arms if a != "base-f32")
    gap = best[0] - mean(means["base-f32"])
    helps = primary[0] <= -0.10 and primary[2] < 0
    if abs(budget[0]) <= 0.05:
        budget_text = "costs little here"
    elif budget[0] > 0:
        budget_text = "costs"
    else:
        budget_text = "helps"
    result = {"protocol_sha256": selection["protocol_sha256"], "selection_sha256": sha(OUT / "selection.json"),
              "selected_arm": sel, "heldout_books": books, "book_mean_nll": means,
              "mean_nll": {a: mean(means[a]) for a in arms},
              "primary_selected_minus_v1": {"mean": primary[0], "t95": primary[1:]},
              "primary_decision": "refinements_help" if helps else "no_material_gain",
              "budget_effect_42_minus_free": {"mean": budget[0], "t95": budget[1:], "reading": budget_text},
              "reconstruction_effect_v2_minus_v1": {"mean": recon[0], "t95": recon[1:]},
              "best_heldout_arm": best[1], "best_minus_fp": gap,
              "training_entry_met": gap > 1.0,
              "models": {a: {"sha256": sha(WORK / f"{a}.gguf"), "bytes": (WORK / f"{a}.gguf").stat().st_size}
                         for a in arms},
              "claim_limit": p["decision"]["claim_limit"]}
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("mean_nll", "primary_selected_minus_v1", "primary_decision",
                                             "budget_effect_42_minus_free", "reconstruction_effect_v2_minus_v1",
                                             "best_minus_fp", "training_entry_met")}, indent=2))


if __name__ == "__main__":
    {"validation": validation, "select": select, "heldout": heldout, "analyze": analyze}[sys.argv[1]]()

"""Validate paired token IDs and summarize the frozen 0.8B NLL screen."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/quality_pilot"
WORK = ROOT / "work/qwen35_08b"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def model_path(arm: str) -> Path:
    return WORK / f"{arm.replace('-absmax', '')}.gguf"


def read(arm: str, split: str) -> dict[str, dict]:
    suffix = "quality" if split == "validation" else "heldout"
    path = WORK / f"{arm}-{suffix}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    result = {Path(row["file"]).stem: row for row in rows}
    assert len(result) == len(rows)
    return result


def main() -> None:
    protocol = json.loads((OUT / "protocol.json").read_text())
    selection = json.loads((OUT / "selection.json").read_text())
    assert selection["protocol_sha256"] == sha(OUT / "protocol.json")
    assert selection["scorer_sha256"] == sha(ROOT / "score_quality_pilot.cpp")
    assert selection["assignment_sha256"] == sha(ROOT / "make_ls_pilot.py")
    entries = {row["id"]: row for row in protocol["entries"]}
    assert len(entries) == 8
    for entry in entries.values():
        assert sha(ROOT.parent / "scorer" / entry["book"]) == entry["source_sha256"]
        assert sha(OUT / entry["text_file"]) == entry["text_sha256"]
    splits = {}
    for split in ("validation", "heldout"):
        arms = protocol["arms"] if split == "validation" else ["base-f32", selection["selected_arm"]]
        cases = {key: value for key, value in entries.items() if value["split"] == split}
        rows = {arm: read(arm, split) for arm in arms}
        for arm in arms:
            assert set(rows[arm]) == set(cases)
        for case in cases:
            token_sets = {tuple(rows[arm][case]["tokens"]) for arm in arms}
            assert len(token_sets) == 1 and len(next(iter(token_sets))) == 128
            for arm in arms:
                assert Path(rows[arm][case]["file"]).resolve() == OUT / cases[case]["text_file"]
        scores = {}
        for arm in arms:
            by_book = defaultdict(list)
            for case, row in rows[arm].items():
                by_book[cases[case]["book"]].append(row["nll"])
            assert all(len(v) == 2 for v in by_book.values())
            means = {book: sum(vals) / 2 for book, vals in by_book.items()}
            scores[arm] = {"mean_nll": sum(means.values()) / len(means),
                           "book_mean_nll": means,
                           "case_nll": {case: row["nll"] for case, row in rows[arm].items()}}
            if split == "validation":
                assert abs(scores[arm]["mean_nll"] - selection["validation_mean_nll"][arm]) < 1e-9
        splits[split] = scores
    selected = selection["selected_arm"]
    assert min((splits["validation"][a]["mean_nll"], a) for a in protocol["arms"] if "pq2" in a)[1] == selected
    heldout = splits["heldout"]
    delta = heldout[selected]["mean_nll"] - heldout["base-f32"]["mean_nll"]
    book_delta = {book: heldout[selected]["book_mean_nll"][book] - baseline
                  for book, baseline in heldout["base-f32"]["book_mean_nll"].items()}
    result = {"protocol_sha256": selection["protocol_sha256"], "selection_sha256": sha(OUT / "selection.json"),
              "source_revision": protocol["source"], "selected_arm": selected,
              "scorer_sha256": selection["scorer_sha256"],
              "models": {arm: {"bytes": model_path(arm).stat().st_size,
                               "sha256": sha(model_path(arm))} for arm in protocol["arms"]},
              "paired_token_ids_identical": True, "splits": splits,
              "heldout_selected_minus_fp_nll": delta, "heldout_book_deltas": book_delta,
              "byte_ceiling_pass": model_path(selected).stat().st_size <= protocol["byte_ceiling"],
              "nll_tolerance_descriptive_pass": delta <= selection["heldout_tolerance_nat_per_token"],
              "claim_limit": "Two independent held-out books; descriptive pilot, not a confidence interval or quality non-inferiority result."}
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"selected {selected}; held-out FP {heldout['base-f32']['mean_nll']:.5f}, "
          f"packed {heldout[selected]['mean_nll']:.5f}, delta {delta:+.5f} nat/token")


if __name__ == "__main__":
    main()

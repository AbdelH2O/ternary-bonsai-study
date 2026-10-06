"""QAT08-MCU design v2: exclude the two contaminated MMLU-fresh items and re-audit, reusing the generated data.

The user's instruction "Proceed with option 1" approved the stated exclusions before any v2 hash existed. v1 is kept
byte-for-byte in results/qat08_mcu/versions/v1 (MANIFEST.json). CPU only; no teacher generation, no training.

    python revise_qat08_mcu_v2.py retire        # unlink active v1 files whose bytes match the archive manifest
    python revise_qat08_mcu_v2.py cases         # v2 cases_record.json + mmlu_fresh.tsv (3,000 -> 2,998, no replacement)
    python revise_qat08_mcu_v2.py reuse-check   # reproduce MC screening and packing in a temporary directory; compare bytes
    python revise_qat08_mcu_v2.py accept        # explicit annotation-only acceptance, bound to the failed check and the diff
    python revise_qat08_mcu_v2.py record        # exclusive results/qat08_mcu/revision.json, embedded by freeze design()
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
import shutil
import tempfile
from pathlib import Path

import prepare_qat08_mcu as p
from prepare_qat08_mcu import FRESH_EXCLUDED, write_new
from score_17b import sha

ROOT = p.ROOT
OUT, DATA = p.OUT, p.DATA
V1 = OUT / "versions/v1"
V1_OUT = V1 / "results/qat08_mcu"
RETIRED = ["results/qat08_mcu/design.json", "results/qat08_mcu/design_approval.json", "results/qat08_mcu/cases_record.json",
           "results/qat08_mcu/cases/mmlu_fresh.tsv", "results/qat08_mcu/contamination_train.jsonl",
           "results/qat08_mcu/contamination_train_M.jsonl", "results/qat08_mcu/contamination_train_B.jsonl"]
REUSED = ["mc_prompts.jsonl", "mc_record.json", "mc_teacher.jsonl", "teacher_timing.json", "pilot_timing.json",
          "kept_mc.json", "mc_unique.u32", "train_M.u32", "train_B.u32", "data_record.json"]
PACKED = ["mc_unique.u32", "kept_mc.json", "train_M.u32", "train_B.u32", "data_record.json"]
CHECK = p.WORK / "revision_v2_reuse_check.json"      # byte-identity check; accepted=false (prompt annotations drifted)
DIFF = p.WORK / "revision_v2_screen_diff.json"        # per-row diff behind it: only max_eval_cosine differs
ACCEPT = p.WORK / "revision_v2_reuse_acceptance.json"
ANNOTATION = "max_eval_cosine"
CRITERION = ("reuse is accepted on the actual teacher inputs and training transcripts, not on prompt-file bytes: identical row "
             "count and ID order; every field except the screen annotation max_eval_cosine identical (prompt_ids, prompt, "
             "gold, source, reasoning, question, seq); identical screen exclusions and retained membership with no row at or "
             "over the 0.6 cut in either screen; all five packed outputs byte-identical; every reused file freshly "
             "hash-matched to the v1 archive manifest")
INSTRUCTION = {
    "user_instruction": "Proceed with option 1",
    "source": "user's prompt in parent bb thread thr_46yhdt5qhc (prompt history phist_jbyazi263f), relayed by that thread",
    "meaning": "approval of the concrete stated option before any v2 hash existed: exclude miscellaneous/7642/D and "
               "miscellaneous/7932/C from the frozen MMLU-fresh sample (3,000 -> 2,998, no replacement), archive v1, "
               "re-freeze, reuse the generated data after hash checks, rerun only the protocol seal. It is not an "
               "approval of the v2 design sha, which needs its own design approval; it authorizes no training."}


def manifest() -> dict:
    return json.loads((V1 / "MANIFEST.json").read_text())


def retire() -> list[str]:
    """Remove the active v1 files that v2 replaces, only where their bytes equal the archived copies."""
    files, removed = manifest()["files"], []
    for name in RETIRED:
        path = ROOT / name
        if not path.exists():
            continue
        assert sha(path) == files[name]["sha256"] == sha(ROOT / files[name]["archived_copy"]), f"{name} differs from its archive"
        path.unlink()
        removed.append(name)
    return removed


def revise_fresh_cases(v1: Path = V1_OUT, out: Path = OUT) -> dict:
    """The v1 MMLU-fresh lines minus FRESH_EXCLUDED, byte-for-byte otherwise; every other case file must be unchanged."""
    old = json.loads((v1 / "cases_record.json").read_text())
    src = v1 / old["mmlu_fresh"]["file"]
    assert sha(src) == old["mmlu_fresh"]["sha256"], "archived v1 MMLU-fresh changed"
    lines = src.read_text().splitlines(keepends=True)
    kept = [l for l in lines if l.split("\t", 1)[0] not in FRESH_EXCLUDED]
    assert len(lines) - len(kept) == len(FRESH_EXCLUDED), "every excluded id must occur exactly once in the v1 sample"
    for key, entry in old.items():
        if key != "mmlu_fresh" and isinstance(entry, dict) and "file" in entry and "sha256" in entry:
            assert sha(out / entry["file"]) == entry["sha256"], f"{key} case file changed"
    dst = out / old["mmlu_fresh"]["file"]
    with dst.open("x") as f:
        f.write("".join(kept))
    fresh = {**old["mmlu_fresh"], "sha256": sha(dst), "items": len(kept), "sampled": len(lines),
             "excluded_after_sampling": FRESH_EXCLUDED, "v1_sha256": old["mmlu_fresh"]["sha256"],
             "gold_letters": dict(collections.Counter(l.split("\t", 1)[0].rsplit("/", 1)[1] for l in kept))}
    record = {**old, "mmlu_fresh": fresh,
              "revision": {"version": 2, "supersedes_cases_record_sha256": sha(v1 / "cases_record.json"),
                           "change": "MMLU-fresh minus the two FRESH_EXCLUDED items; every other case file unchanged"}}
    write_new(out / "cases_record.json", record)
    return record


def _swap(**values):
    previous = {k: getattr(p, k) for k in values}
    for k, v in values.items():
        setattr(p, k, v)
    return previous


def reuse_check() -> dict:
    """Rerun MC screening and packing against the v2 evaluation text in a temporary directory; compare with the reused data."""
    before = {n: sha(DATA / n) for n in REUSED}
    m = manifest()
    bound = {k.rsplit("/", 1)[-1]: v for k, v in m["kept_in_place_hash_only"].items()} | \
            {k.rsplit("/", 1)[-1]: v["sha256"] for k, v in m["files"].items() if k.startswith("work/qat08_mcu/data/")}
    for n in REUSED:
        assert before[n] == bound[n], f"{n} differs from the v1 manifest"
    result = {"reused_sha256": before}
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as name:
        tmp = Path(name)
        screen_data = tmp / "screen/data"
        previous = _swap(DATA=screen_data)
        try:
            p.mc()
        finally:
            _swap(**previous)
        new, old = json.loads((screen_data / "mc_record.json").read_text()), json.loads((DATA / "mc_record.json").read_text())
        result["mc_screen"] = {
            "prompts_identical": sha(screen_data / "mc_prompts.jsonl") == before["mc_prompts.jsonl"],
            "record_identical": sha(screen_data / "mc_record.json") == before["mc_record.json"],
            "fields_differing": sorted(k for k in set(new) | set(old) if new.get(k) != old.get(k)),
            "screen_v1": old["screen"], "screen_v2": new["screen"]}
        pack_data, pack_out = tmp / "pack/data", tmp / "pack/out"
        pack_data.mkdir(parents=True)
        for n in ("mc_prompts.jsonl", "mc_record.json", "mc_teacher.jsonl"):
            shutil.copyfile(DATA / n, pack_data / n)
        shutil.copytree(OUT / "cases", pack_out / "cases")
        shutil.copyfile(V1_OUT / "design.json", pack_out / "design.json")  # pack reads only design["arms"]; identical in v2
        previous = _swap(DATA=pack_data, OUT=pack_out)
        try:
            p.pack()
        finally:
            _swap(**previous)
        result["pack"] = {n: {"v1": before[n], "reproduced": sha(pack_data / n), "identical": sha(pack_data / n) == before[n]}
                          for n in PACKED}
    after = {n: sha(DATA / n) for n in REUSED}
    result["reused_untouched"] = after == before
    result["accepted"] = (result["mc_screen"]["prompts_identical"] and all(v["identical"] for v in result["pack"].values())
                          and result["reused_untouched"])
    write_new(CHECK, result)
    return result


def validate_reuse(old: list[dict], new: list[dict], old_screen: dict, new_screen: dict, pack: dict,
                   reused_now: dict, archive: dict) -> dict:
    """Accept regenerated MC prompts that differ only in the screen annotation; anything else is a stop."""
    assert len(old) == len(new), "row count changed"
    assert [r["id"] for r in old] == [r["id"] for r in new], "row order or retained membership changed"
    for a, b in zip(old, new):
        assert {k: v for k, v in a.items() if k != ANNOTATION} == {k: v for k, v in b.items() if k != ANNOTATION}, \
            f"non-annotation field changed: {a['id']}"
    for key in ("dropped", "candidates", "threshold"):
        assert old_screen.get(key) == new_screen.get(key), f"screen decision changed: {key}"
    cut = old_screen.get("threshold", 0.6)
    assert all(r[ANNOTATION] < cut for r in old) and all(r[ANNOTATION] < cut for r in new), "a retained row crosses the cut"
    assert all(v["identical"] and v["v1"] == v["reproduced"] for v in pack.values()), "packed output changed"
    assert set(pack) == set(PACKED), "packed outputs incomplete"
    for name, digest in archive.items():
        assert reused_now.get(name) == digest, f"reused file differs from the archive: {name}"
    deltas = [abs(a[ANNOTATION] - b[ANNOTATION]) for a, b in zip(old, new)]
    return {"rows": len(old), "rows_with_annotation_drift": sum(d > 0 for d in deltas), "max_abs_annotation_delta": max(deltas),
            "retained_rows_at_or_over_cut": [sum(r[ANNOTATION] >= cut for r in old), sum(r[ANNOTATION] >= cut for r in new)],
            "screen_decisions_identical": True, "packed_identical": sorted(pack), "reused_hash_matched": sorted(archive)}


def accept() -> dict:
    """Separate explicit acceptance; the failed byte-identity check and the diff stay unchanged and are hash-bound."""
    check, diff = json.loads(CHECK.read_text()), json.loads(DIFF.read_text())
    assert check["accepted"] is False and diff["identical_except_max_eval_cosine"] is True
    m = manifest()
    archive = {k.rsplit("/", 1)[-1]: v for k, v in m["kept_in_place_hash_only"].items()} | \
              {k.rsplit("/", 1)[-1]: v["sha256"] for k, v in m["files"].items() if k.startswith("work/qat08_mcu/data/")}
    archive = {n: archive[n] for n in REUSED}
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as name:
        previous = _swap(DATA=Path(name) / "data")
        try:
            p.mc()
            new = [json.loads(l) for l in (p.DATA / "mc_prompts.jsonl").read_text().splitlines()]
            new_record = json.loads((p.DATA / "mc_record.json").read_text())
            regenerated = sha(p.DATA / "mc_prompts.jsonl")
        finally:
            _swap(**previous)
    old = [json.loads(l) for l in (DATA / "mc_prompts.jsonl").read_text().splitlines()]
    old_record = json.loads((DATA / "mc_record.json").read_text())
    reused_now = {n: sha(DATA / n) for n in REUSED}
    checks = validate_reuse(old, new, old_record["screen"], new_record["screen"], check["pack"], reused_now, archive)
    rec = {"accepted": True, "criterion": CRITERION, "checks": checks,
           "bound": {"reuse_check_sha256": sha(CHECK), "screen_diff_sha256": sha(DIFF), "v1_manifest_sha256": sha(V1 / "MANIFEST.json")},
           "original_prompts_sha256": reused_now["mc_prompts.jsonl"], "regenerated_prompts_sha256": regenerated,
           "revised_screen_summary": {"eval_texts": [old_record["screen"]["eval_texts"], new_record["screen"]["eval_texts"]],
                                      "cosine_quantiles_v1": old_record["screen"]["cosine_quantiles"],
                                      "cosine_quantiles_v2": new_record["screen"]["cosine_quantiles"],
                                      "dropped": new_record["screen"]["dropped"]},
           "disclosure": "the regenerated prompt file is NOT byte-identical to the original: the screen annotation "
                         "max_eval_cosine drifts on every row because the TF-IDF is fitted with two fewer evaluation texts. "
                         "The original mc_prompts.jsonl, mc_record.json, teacher responses and timings are kept unchanged and "
                         "stay bound; no gate or recipe changed",
           "ruling": "routine implementation ruling by parent thread thr_46yhdt5qhc within the user's approved option 1",
           "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_new(ACCEPT, rec)
    return rec


def record() -> dict:
    check, acceptance = json.loads(CHECK.read_text()), json.loads(ACCEPT.read_text())
    assert acceptance["accepted"] and acceptance["bound"]["reuse_check_sha256"] == sha(CHECK) and \
        acceptance["bound"]["screen_diff_sha256"] == sha(DIFF), "reuse acceptance missing or unbound; stop and report"
    rec = {"version": 2, "instruction": INSTRUCTION,
           "supersedes": {"version": 1, "design_sha256": manifest()["design_sha256"],
                          "manifest": str((V1 / "MANIFEST.json").relative_to(ROOT)), "manifest_sha256": sha(V1 / "MANIFEST.json")},
           "scope": {"changed": "MMLU-fresh 3,000 -> 2,998 (miscellaneous/7642/D, miscellaneous/7932/C removed, no replacement); "
                                "pre-freeze 13-gram audit of every evaluation set against every existing training stream; "
                                "protocol seal projects from all measured GPU time and checks the reused data hashes",
                     "unchanged": "every other case, arm, data rule, gate, threshold, model, recipe, learning rate and budget"},
           "excluded": FRESH_EXCLUDED, "reused_data_sha256": check["reused_sha256"],
           "reuse_check": {"file": str(CHECK.relative_to(ROOT)), "sha256": sha(CHECK), "accepted": check["accepted"],
                           "note": "byte-identity of regenerated prompts failed on the annotation only; kept as recorded"},
           "screen_diff": {"file": str(DIFF.relative_to(ROOT)), "sha256": sha(DIFF)},
           "reuse_acceptance": {"file": str(ACCEPT.relative_to(ROOT)), "sha256": sha(ACCEPT), "criterion": CRITERION,
                                "max_abs_annotation_delta": acceptance["checks"]["max_abs_annotation_delta"]},
           "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_new(OUT / "revision.json", rec)
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["retire", "cases", "reuse-check", "accept", "record"])
    cmd = ap.parse_args().cmd
    out = {"retire": retire, "cases": revise_fresh_cases, "reuse-check": reuse_check, "accept": accept, "record": record}[cmd]()
    print(json.dumps(out, indent=2)[:6000])

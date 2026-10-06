"""Prepare (never train) QAT08-MC12: one arm with 12.5% multiple-choice data (2 of 16 slots), otherwise M's recipe.

CPU only. The MCU multiple-choice prompts, teacher responses and packed corpus are reused unchanged after hash checks
against the sealed MCU protocol; nothing is regenerated. Writes only results/qat08_mc12 and work/qat08_mc12.

    python prepare_qat08_mc12.py stream    # work/qat08_mc12/data/train_MC12.u32 + stream_record.json
    python prepare_qat08_mc12.py robust    # prompt-robustness dev cases (4 instruction wordings) + robust_record.json
    python prepare_qat08_mc12.py cases     # fresh books, MMLU-fresh2, reused continuity sets + cases_record.json
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import json
import random
import re
import shutil
from pathlib import Path

import numpy as np

import diag_cases as dc
import freeze_17b as fz
import prepare_qat08_mcu as p
from freeze_qat08 import ngram_hashes
from prepare_qat08_chat import grams, normalize, tokenizer, write_new
from score_17b import sha

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/qat08_mc12"
WORK = ROOT / "work/qat08_mc12"
DATA = WORK / "data"
MCU_OUT, MCU_DATA = ROOT / "results/qat08_mcu", ROOT / "work/qat08_mcu/data"
MCU_PROTOCOL_SHA = "b99f4bddfd2177c1f52cb83a747eb1173ea030944a02b1e32d145f89b82a783e"
MCU_V1_FRESH = MCU_OUT / "versions/v1/results/qat08_mcu/cases/mmlu_fresh.tsv"
CHAT_STREAM = ROOT / "work/qat08_chat/data/train.u32"
M_STREAM = MCU_DATA / "train_M.u32"
FINEWEB = ROOT / "work/qat08/data/train.u32"
STREAM = DATA / "train_MC12.u32"
SEQ, TRAIN_SEQS = 1024, 64_000
MC_SLOTS = (10, 14)          # M's slot 14 plus slot 10; both FineWeb slots in M (s % 4 != 3)
FINE_SEQS = 40_000           # exactly the first 40,000 of M's 44,000 FineWeb sequences
MAX_PASSES = 4.0
SEED = 20261007              # pinned for the MC12 trainer worker (M's start-up RNG was never recorded)
FRESH2_ITEMS, FRESH2_SEED = 3000, 20261008
NEAR_DUP = p.NEAR_DUP        # 0.6, the original MC screen cut
OWN_FILES = {ROOT / n for n in ("prepare_qat08_mc12.py", "freeze_qat08_mc12.py", "robust_qat08_mc12.py",
                                "mc12_control.py", "score_qat08_mc12.py", "report_qat08_mc12.py", "test_qat08_mc12.py",
                                "QAT08_MC12.md", "QAT08_MC12_DESIGN.md")}
GUTENBERG = {  # checked by book_inventory() against every earlier experiment, MCU included
    "the_warden": (619, "The Warden"), "agnes_grey": (767, "Agnes Grey"), "jude_the_obscure": (153, "Jude the Obscure"),
    "the_woodlanders": (482, "The Woodlanders"), "the_egoist": (1684, "The Egoist"),
    "felix_holt": (40882, "Felix Holt"), "romola": (24020, "Romola"), "rob_roy": (7025, "Rob Roy"),
    "the_antiquary": (7005, "The Antiquary"), "sybil": (3760, "Sybil"), "coningsby": (7412, "Coningsby"),
    "last_chronicle_of_barset": (3045, "The Last Chronicle of Barset"),
    "humphry_clinker": (2160, "Humphry Clinker"), "mysteries_of_udolpho": (3268, "The Mysteries of Udolpho"),
    "evelina": (6053, "Evelina"),
}
RESERVE = {"joseph_andrews": (9611, "Joseph Andrews"), "roderick_random": (4085, "Roderick Random"),
           "old_mortality": (6943, "Old Mortality"), "tristram_shandy": (1079, "Tristram Shandy")}
VALIDATION = ("the_warden", "agnes_grey", "jude_the_obscure")
VARIANTS = {  # robustness: the frozen evaluation instruction plus three wordings absent from all MC training prompts
    "original": dc.INSTRUCTION,
    "give_letter": "Give the letter (A, B, C or D) corresponding to the right answer, and nothing else.",
    "pick_choice": "Pick the correct choice. Your reply should be only its letter.",
    "output_single": "Output only the single letter of the option that is correct.",
}
MCU_DATA_FILES = ("mc_prompts.jsonl", "mc_record.json", "mc_teacher.jsonl", "mc_unique.u32", "kept_mc.json",
                  "data_record.json", "teacher_timing.json", "train_M.u32")
LICENSES = {"arc": "CC-BY-SA-4.0 (allenai/ai2_arc@210d026f, train split)",
            "fineweb_edu": "ODC-By 1.0 (HuggingFaceFW/fineweb-edu, pinned shard; in-context passages and FineWeb rows)",
            "chat": "CC-BY-4.0 (nvidia/Daring-Anteater prompts; teacher responses from the chat run)",
            "gutenberg": "Project Gutenberg; public domain in the United States",
            "mmlu": "MIT (cais/mmlu dataset card)"}


def mcu_protocol() -> dict:
    path = MCU_OUT / "protocol.json"
    assert sha(path) == MCU_PROTOCOL_SHA, "the sealed MCU protocol changed"
    return json.loads(path.read_text())


def check_mcu_data() -> dict:
    """Every reused MCU data file must match the sealed MCU protocol (train_M against its data record)."""
    proto = mcu_protocol()
    digests = {n: sha(MCU_DATA / n) for n in MCU_DATA_FILES}
    for n, d in proto["data_sha256"].items():
        assert digests.get(n, sha(MCU_DATA / n)) == d, f"MCU data changed: {n}"
    assert digests["train_M.u32"] == proto["data_record"]["stream_sha256"]["train_M.u32"]
    return digests


def layout(s: int) -> str:
    r = s % 16
    return "chat" if r % 4 == 3 else "mc" if r in MC_SLOTS else "fine"


def stream_mc12(mc_unique: np.ndarray, out_path: Path, chat_stream: Path, fine_path: Path, n: int = TRAIN_SEQS) -> dict:
    """M's stream_m with a second MC slot: chat rows copied from C at s % 4 == 3; MC rows at s % 16 in MC_SLOTS
    taken as consecutive whole 1,024-token chunks of the MC corpus (one counter, never reset); FineWeb elsewhere."""
    chat = np.memmap(chat_stream, dtype=np.uint32, mode="r").reshape(-1, SEQ)
    fine = np.memmap(fine_path, dtype=np.uint32, mode="r")
    out = np.memmap(out_path, dtype=np.uint32, mode="w+", shape=(n, SEQ))
    fine_seq = mc_seq = 0
    for s in range(n):
        kind = layout(s)
        if kind == "chat":
            out[s] = chat[s]
        elif kind == "mc":
            out[s] = mc_unique[(mc_seq * SEQ + np.arange(SEQ)) % len(mc_unique)]
            mc_seq += 1
        else:
            out[s] = fine[fine_seq * SEQ:(fine_seq + 1) * SEQ]
            fine_seq += 1
    out.flush()
    del out
    return {"fineweb_seqs": fine_seq, "mc_seqs": mc_seq, "chat_seqs": n // 4}


def stream() -> dict:
    if (DATA / "stream_record.json").exists():
        raise SystemExit("MC12 stream already sealed")
    digests = check_mcu_data()
    DATA.mkdir(parents=True, exist_ok=True)
    mc_unique = np.fromfile(MCU_DATA / "mc_unique.u32", dtype=np.uint32)
    counts = stream_mc12(mc_unique, STREAM, CHAT_STREAM, FINEWEB)
    passes = counts["mc_seqs"] * SEQ / len(mc_unique)
    assert counts == {"fineweb_seqs": FINE_SEQS, "mc_seqs": 8000, "chat_seqs": 16000}, counts
    assert passes <= MAX_PASSES, passes
    write_new(DATA / "stream_record.json", {
        "stream": str(STREAM.relative_to(ROOT)), "stream_sha256": sha(STREAM), **counts, "mc_passes": passes,
        "mc_tokens": counts["mc_seqs"] * SEQ, "mc_unique_tokens": int(len(mc_unique)),
        "layout": "per 16 seqs: C's chat rows at s%4==3 (4/16), MC at s%16 in {10, 14} (2/16) as consecutive whole "
                  "1,024-token chunks of mc_unique.u32 with one counter, FineWeb elsewhere (10/16) = FineWeb seqs 0-39,999",
        "fractions": {"fineweb": 10 / 16, "chat": 4 / 16, "mc": 2 / 16},
        "sources_sha256": {**{f"work/qat08_mcu/data/{n}": d for n, d in digests.items()},
                           "work/qat08_chat/data/train.u32": sha(CHAT_STREAM), "work/qat08/data/train.u32": sha(FINEWEB)},
        "mcu_protocol_sha256": MCU_PROTOCOL_SHA, "licenses": LICENSES,
        "teacher": "reused MCU teacher responses: FP F32 GGUF, greedy, pinned Qwen3.5 template with thinking disabled, 512-token limit"})
    return json.loads((DATA / "stream_record.json").read_text())


# ---------------------------------------------------------------------------------------------------- robustness
def render_variant(tok, ids, wording: str) -> list[int]:
    """Re-render a diagnostics dev case with another instruction sentence; everything else is unchanged."""
    text = tok.decode(list(map(int, ids)))
    user = text.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
    assert user.count(dc.INSTRUCTION) == 1, "the dev case must contain the frozen instruction exactly once"
    return fz.chat(tok, user.replace(dc.INSTRUCTION, wording))


def robust() -> dict:
    if (OUT / "robust_record.json").exists():
        raise SystemExit("robustness cases already sealed")
    plan = json.loads((dc.OUT / "plan.json").read_text())
    tok, out, files = tokenizer(), OUT / "robust", {}
    out.mkdir(parents=True, exist_ok=True)
    for key, name in (("letter", "letter.tsv"), ("binding", "binding.tsv")):
        source = dc.CASES / name
        assert sha(source) == plan["files"][key]["sha256"], f"diagnostics dev case file changed: {name}"
        lines = [l.split("\t") for l in source.read_text().splitlines()]
        for variant, wording in VARIANTS.items():
            rows = [(i, kind, render_variant(tok, ids.split(), wording), extra) for i, kind, ids, extra in lines]
            if variant == "original":
                assert [r[2] for r in rows] == [list(map(int, ids.split())) for _, _, ids, _ in lines], "re-render drift"
            path = out / f"{variant}-{key}.tsv"
            files[f"{variant}-{key}"] = {"file": f"robust/{path.name}", "sha256": fz.tsv(rows, path), "cases": len(rows),
                                         "source": f"results/diag_readout/cases/{name}", "source_sha256": sha(source)}
    write_new(OUT / "robust_record.json", {"variants": VARIANTS, "files": files,
                                           "diag_plan_sha256": sha(dc.OUT / "plan.json"),
                                           "order": "same case order and option/letter order as the diagnostics dev files"})
    return json.loads((OUT / "robust_record.json").read_text())


# ---------------------------------------------------------------------------------------------------- books
@contextlib.contextmanager
def _mc12_scope():
    """Scoped swap of the MCU scanner's own-directory exclusion: only MC12 outputs are skipped, so MCU code, results
    and downloads are scanned like every other earlier experiment. The frozen helper's source is not changed."""
    saved = p.OWN_DIRS
    p.OWN_DIRS = (OUT, WORK)
    try:
        yield
    finally:
        p.OWN_DIRS = saved


def book_inventory(root: Path | None = None) -> dict:
    with _mc12_scope():
        return p.book_inventory(root, exclude=OWN_FILES)


# ---------------------------------------------------------------------------------------------------- MMLU-fresh2
def tsv_signatures(tok, path: Path) -> tuple[set, set, set, set]:
    """Normalized questions, 13-word grams, option tuples and table indices of a frozen MMLU letter-case file."""
    questions, gram_set, options, indices = set(), set(), set(), set()
    for line in path.read_text().splitlines():
        case_id, _, ids, _ = line.split("\t")
        indices.add(int(case_id.split("/")[1]))
        body = tok.decode(list(map(int, ids.split()))).split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
        question, opts = body.split("\n\n", 1)[1].rsplit("\n\n", 1)
        questions.add(normalize(question))
        gram_set |= grams(question)
        options.add(tuple(normalize(o[3:]) for o in opts.split("\n")))
    return questions, gram_set, options, indices


def mc_texts() -> tuple[list[str], list[str]]:
    """Content of every retained MC training item: the prompt content and the teacher response."""
    kept = {k["id"] for k in json.loads((MCU_DATA / "kept_mc.json").read_text())}
    prompts = [json.loads(l) for l in (MCU_DATA / "mc_prompts.jsonl").read_text().splitlines()]
    responses = {r["id"]: r["text"] for r in map(json.loads, (MCU_DATA / "mc_teacher.jsonl").read_text().splitlines())}
    contents = [p._content(r) for r in prompts if r["id"] in kept]
    teacher = [responses[r["id"]] for r in prompts if r["id"] in kept]
    return contents, teacher


def content_screen(candidates: list[tuple[int, dict]], contents: list[str], teacher: list[str]) -> tuple[list, dict]:
    """Drop fresh candidates close to retained MC training text, prompts AND teacher responses alike: exact question
    (or exact response first paragraph), shared 13-word gram, or TF-IDF cosine >= 0.6 with any reference text."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = [f"{it['question']}\n\n{dc.options_text(it['choices'])}" for _, it in candidates]
    reference = contents + teacher
    exact = {normalize(t.split("\n\n", 1)[0]) for t in reference}
    banned = set().union(*(grams(t) for t in reference))
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(reference + texts)
    C, X = vec.transform(reference), vec.transform(texts)
    best = np.zeros(len(texts))
    for i in range(0, len(texts), 1000):
        best[i:i + 1000] = (X[i:i + 1000] @ C.T).max(axis=1).toarray().ravel()
    kept, dropped = [], collections.Counter()
    for cand, text, b in zip(candidates, texts, best):
        if normalize(cand[1]["question"]) in exact:
            dropped["mc_exact_question"] += 1
        elif grams(text) & banned:
            dropped["mc_13word_gram"] += 1
        elif b >= NEAR_DUP:
            dropped["mc_tfidf_near_duplicate"] += 1
        else:
            kept.append(cand)
    q = {str(x): float(np.quantile(best, x)) for x in (0.5, 0.9, 0.99, 1.0)} if len(best) else {}
    return kept, {"dropped": dict(dropped), "tfidf_quantiles": q, "threshold": NEAR_DUP, "mc_prompt_texts": len(contents),
                  "mc_response_texts": len(teacher), "reference": "retained MC prompt contents and teacher responses"}


def majority_fractions(rows: list[tuple], stream_path: Path) -> list[float]:
    seen = np.unique(ngram_hashes(np.asarray(np.memmap(stream_path, dtype=np.uint32, mode="r"))))
    out = []
    for _, _, ids, _ in rows:
        h = ngram_hashes(np.asarray(ids, dtype=np.uint32)) if len(ids) >= 13 else np.asarray([], dtype=np.uint64)
        out.append(float((seen[np.minimum(np.searchsorted(seen, h), len(seen) - 1)] == h).mean()) if len(h) else 0.0)
    return out


AUDIT_STREAMS = ("work/qat08_mc12/data/train_MC12.u32", "work/qat08_mcu/data/train_M.u32", "work/qat08_chat/data/train.u32")


def mmlu_fresh2(tok) -> dict:
    import pyarrow.parquet as pq
    mcu = json.loads((MCU_OUT / "cases_record.json").read_text())["mmlu_fresh"]
    source = MCU_DATA.parent / "source" / p.MMLU_TEST_FILE
    assert sha(source) == mcu["source_sha256"], "MMLU test parquet changed"
    table = pq.read_table(source).to_pylist()
    questions, gram_set, options = p.redux_signatures(tok)
    banned = dc.banned_questions(tok)
    dev = {normalize(r["question"]) for r in pq.read_table(dc.WORK / "source" / dc.MMLU_FILE).to_pylist()}
    redux_all, redux_digests = p.redux_all_questions()
    assert redux_digests == json.loads((p.CHAT_OUT / "protocol.json").read_text())["mmlu"]["source_sha256"], "Redux changed"
    eligible, dropped = p.fresh_eligible(table, banned, questions, gram_set, options, dev, redux_all)
    counts = {"pool": len(table), "fresh_eligible_as_mcu": len(eligible), "dropped_as_mcu": dict(dropped)}
    assert sha(MCU_V1_FRESH) == "65b1085729d5768da520b6f8c15cc3ceb36c70445d73d02ee95ab6bcd8eff222", "archived v1 MCU fresh changed"
    hq, hg, ho, hidx = tsv_signatures(tok, MCU_V1_FRESH)  # all 3,000 v1 items, the two Hurston items included
    hist = collections.Counter()
    stage1 = []
    for i, it in eligible:
        if i in hidx:
            hist["mcu_fresh_item"] += 1
        elif normalize(it["question"]) in hq:
            hist["mcu_fresh_question"] += 1
        elif grams(it["question"]) & hg:
            hist["mcu_fresh_13word_gram"] += 1
        elif tuple(normalize(str(c)) for c in it["choices"]) in ho:
            hist["mcu_fresh_option_tuple"] += 1
        else:
            stage1.append((i, it))
    counts["dropped_historical_mcu_fresh"] = dict(hist)
    contents, teacher = mc_texts()
    stage2, mc_audit = content_screen(stage1, contents, teacher)
    counts["mc_content_screen"] = mc_audit
    rows = [dc.letter_case(tok, i, it) for i, it in stage2]
    rows = [r for r in rows if len(r[2]) <= dc.MAX_TOKENS]
    counts["over_max_tokens"] = len(stage2) - len(rows)
    worst = [0.0] * len(rows)
    for s in AUDIT_STREAMS:
        worst = [max(a, b) for a, b in zip(worst, majority_fractions(rows, ROOT / s))]
    clean = [r for r, w in zip(rows, worst) if w <= .5]
    counts["stream_majority_13gram_dropped"] = len(rows) - len(clean)
    counts["candidates_after_all_exclusions"] = len(clean)
    if len(clean) < FRESH2_ITEMS:
        raise SystemExit(f"only {len(clean)} fresh2 candidates remain after the frozen exclusions; {FRESH2_ITEMS} required")
    chosen = sorted(random.Random(FRESH2_SEED).sample(range(len(clean)), FRESH2_ITEMS))
    final = [clean[j] for j in chosen]
    return {"repo": dc.MMLU_REPO, "revision": dc.MMLU_REV, "file": "cases/mmlu_fresh2.tsv",
            "sha256": fz.tsv(final, OUT / "cases/mmlu_fresh2.tsv"), "items": len(final), "seed": FRESH2_SEED,
            "source_sha256": sha(source), "counts": counts, "license": LICENSES["mmlu"],
            "gold_letters": dict(collections.Counter(r[0].rsplit("/", 1)[1] for r in final)),
            "exclusion": "never-scored MMLU test items: MCU's Redux/dev/content filters, then every archived v1 MCU-fresh item "
                         "(3,000, the two Hurston items included) by index, question, 13-word gram and option tuple, then "
                         "content screening against all retained MC prompts and teacher responses (exact question, 13-word "
                         "gram, TF-IDF >= 0.6), then majority 13-token overlap with the MC12, M and C streams; all fixed before scoring"}


def cases() -> dict:
    if (OUT / "cases_record.json").exists():
        raise SystemExit("cases already sealed")
    assert (DATA / "stream_record.json").exists(), "build the MC12 stream first (the fresh2 stream gate needs it)"
    inventory = book_inventory()
    clash = p.book_clashes(GUTENBERG, inventory)
    assert not clash, f"books already used in earlier experiments: {json.dumps(clash, indent=1)}"
    OUT.mkdir(parents=True, exist_ok=True)
    saved = (fz.OUT, fz.BOOKS, fz.CASES, fz.GUTENBERG, fz.VALIDATION)
    fz.OUT, fz.BOOKS, fz.CASES = OUT, OUT / "books", OUT / "cases"
    fz.GUTENBERG, fz.VALIDATION = {n: g for n, (g, _) in GUTENBERG.items()}, VALIDATION
    fz.CASES.mkdir(parents=True, exist_ok=True)
    fz.BOOKS.mkdir(exist_ok=True)
    squash = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    titles = {}
    try:
        for name, (gid, expected) in GUTENBERG.items():
            path = fz.BOOKS / f"{name}.raw.txt"
            if not path.exists():
                path.write_bytes(fz.fetch(f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"))
            text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
            match = re.search(r"^Title:\s*(.+)$", text[:12000], re.M)
            title = match.group(1).strip() if match else ""
            if squash(expected) not in squash(title):
                path.rename(path.with_name(f"rejected_pg{gid}_{name}.raw.txt"))
                raise SystemExit(f"title check failed for {name} ({gid}): {title!r}. Swap in a RESERVE book, record the "
                                 "incident in work/qat08_mc12/pre_freeze_incidents.json, and rerun.")
            titles[name] = {"gutenberg_id": gid, "expected": expected, "actual": title, "raw_sha256": sha(path)}
        tok = tokenizer()
        entries, files = fz.books(tok)
    finally:
        fz.OUT, fz.BOOKS, fz.CASES, fz.GUTENBERG, fz.VALIDATION = saved
    mcu = json.loads((MCU_OUT / "cases_record.json").read_text())
    reused = {}
    for key in ("mmlu", "gsm8k", "retrieval", "binding"):
        src = MCU_OUT / mcu[key]["file"]
        assert sha(src) == mcu[key]["sha256"], key
        shutil.copyfile(src, OUT / mcu[key]["file"])
        reused[key] = {**mcu[key], "reused_from": f"results/qat08_mcu/{mcu[key]['file']}", "fresh": False}
    record = {"book_entries": entries, "book_cases": files, "titles": titles,
              "prior_book_inventory": p.inventory_record(inventory), **reused, "mmlu_fresh2": mmlu_fresh2(tok),
              "licenses": LICENSES}
    write_new(OUT / "cases_record.json", record)
    return record


def read_cases() -> dict[str, list[tuple[str, np.ndarray]]]:
    record = json.loads((OUT / "cases_record.json").read_text())
    paths = {"validation_books": record["book_cases"]["validation"], "heldout_books": record["book_cases"]["heldout"],
             **{k: record[k] for k in ("mmlu", "mmlu_fresh2", "binding", "gsm8k", "retrieval")}}
    out = {}
    for key, entry in paths.items():
        path = OUT / entry["file"]
        assert sha(path) == entry["sha256"], key
        if path.suffix == ".jsonl":
            rows = [json.loads(l) for l in path.read_text().splitlines()]
            out[key] = [(str(r["id"]), np.asarray(r["prompt_ids"], dtype=np.uint32)) for r in rows]
        else:
            out[key] = [(l.split("\t")[0], np.asarray(list(map(int, l.split("\t")[2].split())), dtype=np.uint32))
                        for l in path.read_text().splitlines()]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stream", "robust", "cases"])
    result = {"stream": stream, "robust": robust, "cases": cases}[ap.parse_args().cmd]()
    print(json.dumps(result, indent=2, default=str)[:4000])

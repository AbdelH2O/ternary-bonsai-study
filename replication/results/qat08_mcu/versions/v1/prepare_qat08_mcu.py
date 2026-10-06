"""Prepare (never train) the QAT08-MCU factorial. Outputs are isolated from every earlier experiment.

    python prepare_qat08_mcu.py cases      # CPU + network: fresh books, MMLU-fresh, binding; continuity copies
    python prepare_qat08_mcu.py mc         # CPU + network: in-context + ARC prompts, screened against every eval set
    python prepare_qat08_mcu.py pilot      # GPU (prep phase): teacher on the first 64 prompts
    python prepare_qat08_mcu.py generate   # GPU (data phase, after design approval): teacher on all prompts
    python prepare_qat08_mcu.py pack       # CPU (data phase): kept MC corpus and the streams named in design.json
"""
from __future__ import annotations

import argparse
import ast
import collections
import concurrent.futures
import json
import random
import re
import shutil
import time
from pathlib import Path

import numpy as np

import diag_cases as dc
import freeze_17b as fz
import gsm8k_17b as gen
from diag_metrics import parse_mc_answer
from prepare_qat08_chat import atomic_json, grams, normalize, tokenizer, write_new
from score_17b import sha

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/qat08_mcu"
WORK = ROOT / "work/qat08_mcu"
DATA = WORK / "data"
CHAT_DATA = ROOT / "work/qat08_chat/data"
CHAT_OUT = ROOT / "results/qat08_chat"
FINEWEB = ROOT / "work/qat08/data/train.u32"
TEACHER = ROOT / "work/qwen35_08b/base-f32.gguf"
SEQ, TRAIN_SEQS, EOS = 1024, 64_000, 248044
GUTENBERG = {  # checked against book_inventory(): no ID, file stem or title used anywhere under analysis/
    "nicholas_nickleby": (967, "Nicholas Nickleby"), "north_and_south": (4276, "North and South"),
    "woman_in_white": (583, "The Woman in White"), "daniel_deronda": (7469, "Daniel Deronda"),
    "barchester_towers": (3409, "Barchester Towers"), "way_we_live_now": (5231, "The Way We Live Now"),
    "pendennis": (7265, "Pendennis"),
    "pickwick_papers": (580, "The Pickwick Papers"), "little_dorrit": (963, "Little Dorrit"),
    "dombey_and_son": (821, "Dombey and Son"), "old_curiosity_shop": (700, "The Old Curiosity Shop"),
    "our_mutual_friend": (883, "Our Mutual Friend"), "barnaby_rudge": (917, "Barnaby Rudge"),
    "anna_karenina": (1399, "Anna Karenina"), "tom_jones": (6593, "History of Tom Jones"),
}
RESERVE = {"martin_chuzzlewit": (968, "Martin Chuzzlewit"), "the_newcomes": (7467, "The Newcomes"),
           "henry_esmond": (2511, "The History of Henry Esmond"), "waverley": (5998, "Waverley")}
VALIDATION = ("nicholas_nickleby", "north_and_south", "woman_in_white")
INCIDENTS = ROOT / "work/qat08_mcu/pre_freeze_incidents.json"
MMLU_TEST_FILE = "all/test-00000-of-00001.parquet"
FRESH_ITEMS, FRESH_SEED, BIND_SEED = 3000, 20261011, 20261010
ARC_REPO, ARC_REV = "allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453"
ARC_FILES = ("ARC-Challenge/train-00000-of-00001.parquet", "ARC-Easy/train-00000-of-00001.parquet")
PASSAGE_SEQS = (100_000, 290_000)  # beyond every QAT08/C/B training read (<96,000) and before the probe offset (300,000)
INCONTEXT_ITEMS, MC_SEED = 6000, 20261012
PARAPHRASES = (
    "Answer with only the letter (A, B, C, or D) of the correct option.",
    "Reply with just the letter of the correct answer.",
    "Choose the correct option and respond with its letter only.",
    "Which option is correct? Give only the letter.",
    "Select the best answer. Respond with a single letter: A, B, C, or D.",
)
REASON_SUFFIX = "Briefly explain your reasoning, then finish with 'Answer: X' where X is the letter of the correct option."
REASON_SHARE, NEAR_DUP, TEACHER_LIMIT = 0.2, 0.6, 512
MC_SLOT, MC_TOKENS = 14, 4_096_000
MIN_KEPT, MIN_UNIQUE, MAX_PASSES = 5000, 1_000_000, 4.0
WORD = re.compile(r"\b[a-z]{6,}\b")


def blank_item(paragraph: str, rng: random.Random) -> dict | None:
    """Fill-the-blank item answerable only by reading the passage: distractors are other once-only passage words."""
    if not 80 <= len(paragraph.split()) <= 200:
        return None
    counts = collections.Counter(w.lower() for w in re.findall(r"[A-Za-z]+", paragraph))
    unique = sorted({w for w in WORD.findall(paragraph) if counts[w] == 1})
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", paragraph) if len(s.split()) >= 12]
    rng.shuffle(sentences)
    for sentence in sentences:
        targets = [w for w in WORD.findall(sentence) if counts[w] == 1]
        if not targets:
            continue
        target = rng.choice(targets)
        pool = [w for w in unique if w != target and not re.search(rf"\b{w}\b", sentence)]
        if len(pool) < 3:
            return None
        return {"sentence": re.sub(rf"\b{target}\b", "____", sentence, count=1), "target": target,
                "distractors": rng.sample(pool, 3)}
    return None


def split_docs(ids) -> list[list[int]]:
    docs, cur = [], []
    for x in ids:
        if int(x) == EOS:
            docs.append(cur)
            cur = []
        else:
            cur.append(int(x))
    docs.append(cur)
    return [d for d in docs if d]


def _instruction(rng: random.Random) -> tuple[str, bool]:
    reasoning = rng.random() < REASON_SHARE
    return (REASON_SUFFIX if reasoning else rng.choice(PARAPHRASES)), reasoning


def incontext_items(tok, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    data = np.memmap(FINEWEB, dtype=np.uint32, mode="r")
    seqs = list(range(*PASSAGE_SEQS))
    rng.shuffle(seqs)
    items = []
    for s in seqs:
        for doc in split_docs(data[s * SEQ:(s + 1) * SEQ]):
            made = False
            for paragraph in tok.decode(doc).split("\n"):
                paragraph = paragraph.strip()
                item = blank_item(paragraph, rng)
                if item is None:
                    continue
                gold = len(items) % 4
                options = item["distractors"][:gold] + [item["target"]] + item["distractors"][gold:]
                instruction, reasoning = _instruction(rng)
                text = (f"The following is a multiple choice question about a reading passage. {instruction}\n\n"
                        f"Passage:\n{paragraph}\n\nWhich word fills the blank?\n{item['sentence']}\n\n{dc.options_text(options)}")
                ids = fz.chat(tok, text)
                if len(ids) > 768:
                    continue
                items.append({"id": f"ic{len(items):05d}", "source": "incontext", "seq": s, "gold": "ABCD"[gold],
                              "reasoning": reasoning, "prompt": text, "prompt_ids": ids})
                made = True
                break
            if len(items) == n:
                return items
            if made:
                break  # at most one item per 1,024-token sequence
    raise SystemExit(f"only {len(items)} in-context items available")


def arc_items(tok, seed: int) -> list[dict]:
    """ARC train (CC BY-SA 4.0), each 4-option item in two orders with different gold letters."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    rng = random.Random(seed + 1)
    rows = []
    for name in ARC_FILES:
        path = Path(hf_hub_download(ARC_REPO, name, revision=ARC_REV, repo_type="dataset", local_dir=WORK / "source"))
        for item in pq.read_table(path).to_pylist():
            texts, labels = list(item["choices"]["text"]), list(item["choices"]["label"])
            key = item["answerKey"]
            if len(texts) != 4 or labels not in (list("ABCD"), list("1234")) or key not in labels:
                continue
            g = labels.index(key)
            gold_text, wrong = texts[g], [t for i, t in enumerate(texts) if i != g]
            first = rng.randrange(4)
            for order, letter in enumerate((first, (first + 1 + rng.randrange(3)) % 4)):
                d = wrong[:]
                rng.shuffle(d)
                options = d[:letter] + [gold_text] + d[letter:]
                instruction, reasoning = _instruction(rng)
                text = (f"The following is a multiple choice question about science. {instruction}\n\n"
                        f"{item['question']}\n\n{dc.options_text(options)}")
                rows.append({"id": f"arc-{item['id']}-{order}", "source": "arc", "gold": "ABCD"[letter],
                             "reasoning": reasoning, "question": item["question"], "prompt": text,
                             "prompt_ids": fz.chat(tok, text)})
    return rows


def _content(item: dict) -> str:
    if "question" in item:
        return item["question"]
    return item["prompt"].split("Passage:\n", 1)[1] if "Passage:\n" in item["prompt"] else item["prompt"]


def screen(items: list[dict], eval_texts: list[str]) -> tuple[list[dict], dict]:
    """Drop MC prompts sharing a 13-word gram, an exact question, or TF-IDF cosine >= NEAR_DUP with any eval content."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    banned = set().union(*(grams(t) for t in eval_texts))
    exact = {normalize(t.split("\n\n", 1)[0]) for t in eval_texts}
    contents = [_content(it) for it in items]
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(eval_texts + contents)
    E, X = vec.transform(eval_texts), vec.transform(contents)
    best = np.zeros(len(items))
    for i in range(0, len(items), 2000):
        best[i:i + 2000] = (X[i:i + 2000] @ E.T).max(axis=1).toarray().ravel()
    kept, dropped = [], collections.Counter()
    for it, text, b in zip(items, contents, best):
        if normalize(text.split("\n\n", 1)[0]) in exact:
            dropped["exact_question"] += 1
        elif grams(text) & banned:
            dropped["shared_13word_gram"] += 1
        elif b >= NEAR_DUP:
            dropped["near_duplicate"] += 1
        else:
            kept.append({**it, "max_eval_cosine": float(b)})
    quantiles = {str(q): float(np.quantile(best, q)) for q in (0.5, 0.9, 0.99, 1.0)} if len(best) else {}
    return kept, {"dropped": dict(dropped), "cosine_quantiles": quantiles, "threshold": NEAR_DUP,
                  "eval_texts": len(eval_texts), "candidates": len(items)}


SCAN_ROOT = ROOT.parents[1]  # analysis/: every earlier experiment's scripts, saved protocols, manifests and downloads
OWN_FILES = {ROOT / n for n in ("prepare_qat08_mcu.py", "freeze_qat08_mcu.py", "qat_08b_mcu.py", "score_qat08_mcu.py",
                                "qat08_mcu_control.py", "report_qat08_mcu.py", "test_qat08_mcu.py")}
OWN_DIRS = (OUT, WORK)
SKIP_PARTS = {".cache", "__pycache__", ".venv", ".git"}
BOOK_ASSIGN = re.compile(r"GUTENBERG|BOOK|DOC|SPLIT|HELDOUT|CALIBRATION|VALIDATION|RESERVE|TEXT|TITLE", re.I)
ID_URL = re.compile(r"(?:gutenberg\.org/(?:cache/epub|ebooks|files)/|\bpg)(\d+)")
FILE_STEM = re.compile(r"([A-Za-z0-9_]+?)(?:-\d+)?(?:\.raw)?\.(?:txt|ids)\b")


def pre_freeze_incidents(path: Path = INCIDENTS) -> list[dict]:
    return json.loads(path.read_text()) if path.exists() else []


def record_incident(entry: dict, path: Path = INCIDENTS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, pre_freeze_incidents(path) + [{"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **entry}])


def _words(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", text.lower()).strip() + " "


def _stem(text: str) -> str:
    m = FILE_STEM.fullmatch(text.strip().rsplit("/", 1)[-1])
    return re.sub(r"[^a-z0-9]+", "_", (m.group(1) if m else text).lower()).strip("_")


def _literal(value, ids: set, stems: set) -> None:
    """Book inventories as literals: {stem: id}, {stem: (id, title)}, {"stem.txt": split}, [(stem, id, title)], (stems)."""
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(k, str):
                stems.add(_stem(k))
            if isinstance(v, int) and not isinstance(v, bool):
                ids.add(v)
            else:
                _literal(v, ids, stems)
    elif isinstance(value, (list, tuple, set, frozenset)):
        ids.update(v for v in value if isinstance(v, int) and not isinstance(v, bool))
        for v in value:
            if isinstance(v, str):
                stems.add(_stem(v))
            elif not isinstance(v, int):
                _literal(v, ids, stems)


def _json_books(value, ids: set, stems: set) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            if re.search(r"gutenberg|ebook", k, re.I) and isinstance(v, int):
                ids.add(v)
            if k in ("id", "book", "name", "document_id", "title") and isinstance(v, str):
                stems.add(_stem(v))
            if k in ("books", "documents", "titles") and isinstance(v, dict):
                stems.update(_stem(x) for x in v)
            _json_books(v, ids, stems)
    elif isinstance(value, list):
        for v in value:
            _json_books(v, ids, stems)


def book_inventory(root: Path | None = None, exclude: set[Path] | None = None) -> dict[str, dict]:
    """Every book any earlier experiment touched, by Gutenberg ID, file stem and text, under root (default analysis/).

    Python literals assigned to book-like names, saved JSON (gutenberg_id/ebook keys, ids, URLs, file names), Markdown
    reports, and downloaded *.txt / *.ids files on disk. This experiment's own files and outputs are excluded."""
    root = root or SCAN_ROOT
    exclude = OWN_FILES if exclude is None else exclude
    inventory = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or SKIP_PARTS & set(path.parts) or path in exclude or \
                any(d == path or d in path.parents for d in OWN_DIRS):
            continue
        ids, stems, text = set(), set(), None
        if path.suffix in (".txt", ".ids"):
            stems.add(_stem(path.name))
        elif path.suffix in (".py", ".json", ".md"):
            text = path.read_text(errors="replace")
            ids.update(int(g) for g in ID_URL.findall(text))
            stems.update(_stem(m.group(0)) for m in FILE_STEM.finditer(text))
            if path.suffix == ".py":
                try:
                    tree = ast.parse(text)
                except SyntaxError:
                    tree = None
                for node in ast.walk(tree) if tree else ():
                    if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                        if any(isinstance(t, ast.Name) and BOOK_ASSIGN.search(t.id) for t in targets):
                            try:
                                _literal(ast.literal_eval(node.value), ids, stems)
                            except (ValueError, TypeError, SyntaxError, RecursionError):
                                pass
            elif path.suffix == ".json":
                try:
                    _json_books(json.loads(text), ids, stems)
                except json.JSONDecodeError:
                    pass
        else:
            continue
        if ids or stems or text:
            inventory[str(path.relative_to(root))] = {"ids": ids, "stems": stems - {""}, "words": _words(text) if text else ""}
    return inventory


def _title_core(title: str) -> str:
    core = re.split(r"[;:]", title, 1)[0]
    return re.sub(r"^(the|a|an) ", "", _words(core).strip())


def book_clashes(books: dict[str, tuple[int, str]], inventory: dict[str, dict]) -> dict[str, list[str]]:
    """Each candidate whose ID, stem, stem phrase or title phrase appears in any earlier file, with the evidence."""
    out = {}
    for name, (gid, title) in books.items():
        phrases = {" " + _words(name.replace("_", " ")).strip() + " ", " " + _title_core(title) + " "}
        hits = []
        for path, entry in inventory.items():
            if gid in entry["ids"]:
                hits.append(f"id {gid}: {path}")
            if _stem(name) in entry["stems"]:
                hits.append(f"stem {name}: {path}")
            hits += [f"phrase{p.rstrip()!s}: {path}" for p in phrases if p.strip() and p in entry["words"]]
        if hits:
            out[name] = sorted(set(hits))
    return out


def inventory_record(inventory: dict[str, dict]) -> dict:
    return {"scan_root": str(SCAN_ROOT.relative_to(ROOT.parents[2])), "files_scanned": len(inventory),
            "gutenberg_ids": sorted({g for e in inventory.values() for g in e["ids"]}),
            "book_stems_on_disk_or_in_literals": sorted({s for e in inventory.values() for s in e["stems"]})}


def redux_signatures(tok) -> tuple[set, set, set]:
    """Normalized questions, 13-word grams and option tuples of every frozen MMLU-Redux case."""
    questions, gram_set, options = set(), set(), set()
    for line in (CHAT_OUT / "cases/mmlu.tsv").read_text().splitlines():
        text = tok.decode(list(map(int, line.split("\t")[2].split())))
        body = text.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0].split("\n\n", 1)[1]
        question, opts = body.rsplit("\n\n", 1)
        questions.add(normalize(question))
        gram_set |= grams(question)
        options.add(tuple(normalize(o[3:]) for o in opts.split("\n")))
    return questions, gram_set, options


def redux_all_questions() -> tuple[set[str], dict[str, str]]:
    """Normalized questions of every MMLU-Redux 2.0 item, whatever its error_type (the frozen cases keep only 'ok')."""
    import io
    import pyarrow as pa
    repo, rev = fz.MMLU
    listing = json.loads(fz.fetch(f"https://huggingface.co/api/datasets/{repo}/revision/{rev}"))
    subjects = sorted({x["rfilename"].split("/")[0] for x in listing["siblings"] if x["rfilename"].endswith(".arrow")})
    questions, digests = set(), {}
    for subject in subjects:
        data = fz.fetch(f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{subject}/data-00000-of-00001.arrow")
        digests[subject] = fz.digest(data)
        questions |= {normalize(r["question"]) for r in pa.ipc.open_stream(io.BytesIO(data)).read_all().to_pylist()}
    return questions, digests


def fresh_eligible(table, banned, redux_questions, redux_grams, redux_options, dev, redux_all):
    excluded_questions = redux_questions | dev
    eligible, dropped = [], collections.Counter()
    for i, it in enumerate(table):
        q = normalize(it["question"])
        if not dc.keep_item(it, banned):
            dropped["filtered_or_redux_question"] += 1
        elif q in redux_all:
            dropped["any_redux_item_including_flagged"] += 1
        elif q in excluded_questions:
            dropped["redux_or_dev_question"] += 1
        elif grams(it["question"]) & redux_grams:
            dropped["redux_13word_gram"] += 1
        elif tuple(normalize(str(c)) for c in it["choices"]) in redux_options:
            dropped["redux_option_tuple"] += 1
        else:
            eligible.append((i, it))
    return eligible, dropped


def mmlu_fresh(tok) -> dict:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    path = Path(hf_hub_download(dc.MMLU_REPO, MMLU_TEST_FILE, revision=dc.MMLU_REV, repo_type="dataset",
                                local_dir=WORK / "source"))
    table = pq.read_table(path).to_pylist()
    questions, gram_set, options = redux_signatures(tok)
    banned = dc.banned_questions(tok)
    dev = {normalize(r["question"]) for r in pq.read_table(dc.WORK / "source" / dc.MMLU_FILE).to_pylist()}
    redux_all, redux_digests = redux_all_questions()
    assert redux_digests == json.loads((CHAT_OUT / "protocol.json").read_text())["mmlu"]["source_sha256"], "Redux source changed"
    eligible, dropped = fresh_eligible(table, banned, questions, gram_set, options, dev, redux_all)
    rows = []
    for j in sorted(random.Random(FRESH_SEED).sample(range(len(eligible)), FRESH_ITEMS)):
        i, it = eligible[j]
        row = dc.letter_case(tok, i, it)
        if len(row[2]) <= dc.MAX_TOKENS:
            rows.append(row)
    return {"repo": dc.MMLU_REPO, "revision": dc.MMLU_REV, "file": "cases/mmlu_fresh.tsv",
            "sha256": fz.tsv(rows, OUT / "cases/mmlu_fresh.tsv"), "items": len(rows), "eligible": len(eligible),
            "dropped": dict(dropped), "source_sha256": sha(path), "seed": FRESH_SEED, "license": "MIT (dataset card)",
            "gold_letters": dict(collections.Counter(r[0].rsplit("/", 1)[1] for r in rows)),
            "redux_items_all_error_types": len(redux_all),
            "exclusion": "MMLU test items not in MMLU-Redux 2.0 (any error_type) by normalized question, 13-word gram or option tuple; not an MMLU validation (diagnostics dev) question"}


def cases() -> None:
    if (OUT / "cases_record.json").exists():
        raise SystemExit("cases already sealed")
    inventory = book_inventory()
    clash = book_clashes(GUTENBERG, inventory)
    assert not clash, f"books already used in earlier experiments: {json.dumps(clash, indent=1)}"
    prior = inventory_record(inventory)
    OUT.mkdir(parents=True, exist_ok=True)
    fz.OUT, fz.BOOKS, fz.CASES = OUT, OUT / "books", OUT / "cases"
    fz.GUTENBERG, fz.VALIDATION = {n: g for n, (g, _) in GUTENBERG.items()}, VALIDATION
    fz.CASES.mkdir(parents=True, exist_ok=True)
    fz.BOOKS.mkdir(exist_ok=True)
    squash = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    titles = {}
    for name, (gid, expected) in GUTENBERG.items():
        path = fz.BOOKS / f"{name}.raw.txt"
        if not path.exists():
            path.write_bytes(fz.fetch(f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"))
        text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
        match = re.search(r"^Title:\s*(.+)$", text[:12000], re.M)
        title = match.group(1).strip() if match else ""
        if squash(expected) not in squash(title):
            path.rename(path.with_name(f"rejected_pg{gid}_{name}.raw.txt"))
            raise SystemExit(f"title check failed for {name} ({gid}): {title!r}. Swap in a RESERVE book, "
                             "record the incident in work/qat08_mcu/pre_freeze_incidents.json, and rerun.")
        titles[name] = {"gutenberg_id": gid, "expected": expected, "actual": title, "raw_sha256": sha(path)}
    tok = tokenizer()
    entries, files = fz.books(tok)
    chat = json.loads((CHAT_OUT / "protocol.json").read_text())
    for key in ("mmlu", "gsm8k", "retrieval"):
        source = CHAT_OUT / chat[key]["file"]
        assert sha(source) == chat[key]["sha256"], key
        shutil.copyfile(source, OUT / chat[key]["file"])
    binding = dc.binding_cases(tok, seed=BIND_SEED)
    write_new(OUT / "cases_record.json", {
        "book_entries": entries, "book_cases": files, "titles": titles, "prior_book_inventory": prior,
        **{k: chat[k] for k in ("mmlu", "gsm8k", "retrieval")}, "mmlu_fresh": mmlu_fresh(tok),
        "binding": {"file": "cases/binding.tsv", "sha256": fz.tsv(binding, OUT / "cases/binding.tsv"),
                    "cases": len(binding), "seed": BIND_SEED}})
    print(json.dumps({"books": files, "mmlu_fresh": json.loads((OUT / "cases_record.json").read_text())["mmlu_fresh"]["items"]}, indent=2))


def read_cases() -> dict[str, list[tuple[str, np.ndarray]]]:
    record = json.loads((OUT / "cases_record.json").read_text())
    paths = {"validation_books": record["book_cases"]["validation"], "heldout_books": record["book_cases"]["heldout"],
             **{k: record[k] for k in ("mmlu", "mmlu_fresh", "binding", "gsm8k", "retrieval")}}
    result = {}
    for key, entry in paths.items():
        path = OUT / entry["file"]
        assert sha(path) == entry["sha256"], key
        if path.suffix == ".jsonl":
            rows = [json.loads(l) for l in path.read_text().splitlines()]
            result[key] = [(str(r["id"]), np.asarray(r["prompt_ids"], dtype=np.uint32)) for r in rows]
        else:
            result[key] = [(l.split("\t")[0], np.asarray(list(map(int, l.split("\t")[2].split())), dtype=np.uint32))
                           for l in path.read_text().splitlines()]
    return result


def eval_contents(tok) -> list[str]:
    """Question content (no instruction header, no template) of every evaluation and diagnostics-dev case."""
    def content(ids) -> str:
        return tok.decode(list(map(int, ids))).split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
    out = []
    for path in (OUT / "cases/mmlu.tsv", OUT / "cases/mmlu_fresh.tsv", OUT / "cases/binding.tsv", dc.CASES / "letter.tsv"):
        for line in path.read_text().splitlines():
            out.append(content(line.split("\t")[2].split()).split("\n\n", 1)[1])
    for line in (OUT / "cases/gsm8k.jsonl").read_text().splitlines():
        out.append(content(json.loads(line)["prompt_ids"]).rsplit("\nPlease reason step by step", 1)[0])
    return out


def mc() -> None:
    if (DATA / "mc_prompts.jsonl").exists():
        raise SystemExit("MC prompts already built")
    DATA.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    card = Path(hf_hub_download(ARC_REPO, "README.md", revision=ARC_REV, repo_type="dataset", local_dir=WORK / "source"))
    assert "cc-by-sa-4.0" in card.read_text(), "ARC licence changed"
    tok = tokenizer()
    items = incontext_items(tok, INCONTEXT_ITEMS, MC_SEED) + arc_items(tok, MC_SEED)
    kept, audit = screen(items, eval_contents(tok))
    kept = [kept[i] for i in random.Random(MC_SEED + 2).sample(range(len(kept)), len(kept))]
    (DATA / "mc_prompts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kept))
    write_new(DATA / "mc_record.json", {
        "items": len(kept), "by_source": dict(collections.Counter(r["source"] for r in kept)),
        "gold_letters": dict(collections.Counter(r["gold"] for r in kept)),
        "reasoning_items": sum(r["reasoning"] for r in kept), "screen": audit,
        "sources": {"incontext": {"file": "work/qat08/data/train.u32", "sha256": sha(FINEWEB), "seq_range": list(PASSAGE_SEQS),
                                  "license": "FineWeb-Edu ODC-By 1.0 (already used for QAT08 training)"},
                    "arc": {"repo": ARC_REPO, "revision": ARC_REV, "files": list(ARC_FILES), "card_sha256": sha(card),
                            "license": "CC-BY-SA-4.0", "attribution": "AI2 Reasoning Challenge (Clark et al., 2018)"}},
        "paraphrases": list(PARAPHRASES), "reason_suffix": REASON_SUFFIX, "seed": MC_SEED,
        "prompts_sha256": sha(DATA / "mc_prompts.jsonl")})
    print(json.dumps(json.loads((DATA / "mc_record.json").read_text()), indent=2))


def generate(pilot: bool = False) -> None:
    prompts = [json.loads(l) for l in (DATA / "mc_prompts.jsonl").read_text().splitlines()]
    assert sha(DATA / "mc_prompts.jsonl") == json.loads((DATA / "mc_record.json").read_text())["prompts_sha256"]
    target = DATA / "mc_teacher.jsonl"
    done = {r["id"] for r in map(json.loads, target.read_text().splitlines())} if target.exists() else set()
    pending = [r for r in (prompts[:64] if pilot else prompts) if r["id"] not in done]
    gen.PORT, gen.N_PREDICT = 8094, TEACHER_LIMIT
    t0, count, tokens = time.time(), 0, 0
    if pending:
        with gen.serve(TEACHER, "model", WORK / "teacher-server.log"):
            with target.open("a", buffering=1) as f, concurrent.futures.ThreadPoolExecutor(gen.SLOTS) as pool:
                for row, out in zip(pending, pool.map(lambda r: gen.complete(r["prompt_ids"]), pending)):
                    f.write(json.dumps({"id": row["id"], **out}) + "\n")
                    count += 1
                    tokens += out["tokens"] or 0
    atomic_json(DATA / ("pilot_timing.json" if pilot else "teacher_timing.json"), {
        "responses_this_session": count, "seconds": time.time() - t0, "tokens": tokens, "pilot": pilot,
        "model_sha256": sha(TEACHER), "prompts_sha256": sha(DATA / "mc_prompts.jsonl"),
        "generation": f"F32 FP GGUF, pinned llama-server, greedy, thinking disabled, 8 slots, {TEACHER_LIMIT} token limit"})


def stream_m(mc_unique: np.ndarray, out_path: Path, chat_stream: Path, fine_path: Path, n: int = TRAIN_SEQS) -> dict:
    """Per 16 sequences: chat rows copied from C at s % 4 == 3; one MC row at s % 16 == MC_SLOT; FineWeb elsewhere."""
    chat = np.memmap(chat_stream, dtype=np.uint32, mode="r").reshape(-1, SEQ)
    fine = np.memmap(fine_path, dtype=np.uint32, mode="r")
    out = np.memmap(out_path, dtype=np.uint32, mode="w+", shape=(n, SEQ))
    fine_seq = mc_seq = 0
    for s in range(n):
        r = s % 16
        if r % 4 == 3:
            out[s] = chat[s]
        elif r == MC_SLOT:
            out[s] = mc_unique[(mc_seq * SEQ + np.arange(SEQ)) % len(mc_unique)]
            mc_seq += 1
        else:
            out[s] = fine[fine_seq * SEQ:(fine_seq + 1) * SEQ]
            fine_seq += 1
    out.flush()
    del out
    return {"fineweb_seqs": fine_seq, "mc_seqs": mc_seq, "chat_seqs": n // 4}


def stream_b(chat_unique: np.ndarray, out_path: Path, fine_path: Path, n: int = TRAIN_SEQS,
             fine_start: int = 48_000, chat_start: int = 16_000) -> dict:
    """C's exact layout continued: the next unseen FineWeb sequences and chat windows."""
    fine = np.memmap(fine_path, dtype=np.uint32, mode="r")
    out = np.memmap(out_path, dtype=np.uint32, mode="w+", shape=(n, SEQ))
    fine_seq, chat_seq = fine_start, chat_start
    for s in range(n):
        if s % 4 == 3:
            out[s] = chat_unique[(chat_seq * SEQ + np.arange(SEQ)) % len(chat_unique)]
            chat_seq += 1
        else:
            out[s] = fine[fine_seq * SEQ:(fine_seq + 1) * SEQ]
            fine_seq += 1
    out.flush()
    del out
    return {"fineweb_range": [fine_start, fine_seq], "chat_window_range": [chat_start, chat_seq]}


def pack() -> None:
    if (DATA / "data_record.json").exists():
        raise SystemExit("data already sealed")
    design = json.loads((OUT / "design.json").read_text())
    tok = tokenizer()
    banned = set().union(*(grams(t) for t in eval_contents(tok)))
    prompts = [json.loads(l) for l in (DATA / "mc_prompts.jsonl").read_text().splitlines()]
    responses = {r["id"]: r for r in map(json.loads, (DATA / "mc_teacher.jsonl").read_text().splitlines())}
    assert set(responses) == {r["id"] for r in prompts}, "teacher generation incomplete"
    end = tok.encode("<|im_end|>\n", add_special_tokens=False)
    ids_all, kept, excluded, teacher = [], [], collections.Counter(), collections.Counter()
    for row in prompts:
        out = responses[row["id"]]
        text = out["text"].strip()
        if out["stop_type"] != "eos" or not text:
            excluded["not_naturally_terminated_or_empty"] += 1
            continue
        if "<think>" in text or "</think>" in text or "<|" in text:
            excluded["special_token_or_thinking"] += 1
            continue
        if grams(text) & banned:
            excluded["response_benchmark_overlap"] += 1
            continue
        ids = row["prompt_ids"] + tok.encode(text, add_special_tokens=False) + end
        if len(ids) > SEQ:
            excluded["conversation_over_1024"] += 1
            continue
        rendered = tok.apply_chat_template([{"role": "user", "content": row["prompt"]},
                                            {"role": "assistant", "content": text}], tokenize=False,
                                           add_generation_prompt=False, enable_thinking=False)
        assert ids == tok.encode(rendered, add_special_tokens=False), row["id"]
        pred = parse_mc_answer(text) if row["reasoning"] else (text[0] if text[0] in "ABCD" else None)
        teacher[f"{row['source']}_{'correct' if pred == row['gold'] else 'wrong_or_unparsed'}"] += 1
        ids_all.extend(ids)
        kept.append({"id": row["id"], "source": row["source"], "tokens": len(ids)})
    unique = np.asarray(ids_all, dtype=np.uint32)
    passes = MC_TOKENS / max(1, len(unique))
    assert len(kept) >= MIN_KEPT and len(unique) >= MIN_UNIQUE and passes <= MAX_PASSES, (len(kept), len(unique), passes, dict(excluded))
    unique.tofile(DATA / "mc_unique.u32")
    atomic_json(DATA / "kept_mc.json", kept)  # unsealed intermediate; data_record.json is the exclusive seal
    chat_unique = np.fromfile(CHAT_DATA / "chat_unique.u32", dtype=np.uint32)
    c = np.memmap(CHAT_DATA / "train.u32", dtype=np.uint32, mode="r").reshape(-1, SEQ)
    fine = np.memmap(FINEWEB, dtype=np.uint32, mode="r")
    assert np.array_equal(c[63999], chat_unique[(15999 * SEQ + np.arange(SEQ)) % len(chat_unique)])
    assert np.array_equal(c[63998], fine[47999 * SEQ:48000 * SEQ])
    streams = {}
    if any(a["stream"].endswith("train_M.u32") for a in design["arms"].values()):
        streams["train_M.u32"] = stream_m(unique, DATA / "train_M.u32", CHAT_DATA / "train.u32", FINEWEB)
        m = np.memmap(DATA / "train_M.u32", dtype=np.uint32, mode="r").reshape(-1, SEQ)
        assert np.array_equal(m[3::4], c[3::4]), "M must keep C's chat rows"
    if any(a["stream"].endswith("train_B.u32") for a in design["arms"].values()):
        streams["train_B.u32"] = stream_b(chat_unique, DATA / "train_B.u32", FINEWEB)
    write_new(DATA / "data_record.json", {
        "mc_kept": len(kept), "mc_unique_tokens": int(len(unique)), "mc_tokens": MC_TOKENS, "mc_passes": passes,
        "excluded": dict(excluded), "kept_by_source": dict(collections.Counter(k["source"] for k in kept)),
        "teacher_letter_accuracy": dict(teacher), "streams": streams,
        "stream_sha256": {n: sha(DATA / n) for n in streams}, "mc_unique_sha256": sha(DATA / "mc_unique.u32"),
        "kept_mc_sha256": sha(DATA / "kept_mc.json"), "mc_teacher_sha256": sha(DATA / "mc_teacher.jsonl"),
        "chat_stream_sha256": sha(CHAT_DATA / "train.u32"),
        "layout": {"M": "per 16 seqs: C's chat rows at s%4==3, MC at s%16==14, FineWeb elsewhere (45,056,000 tokens)",
                   "B": "C's [F F F C] layout continued from FineWeb seq 48,000 and chat window 16,000",
                   "U": "C's exact stream work/qat08_chat/data/train.u32"}})
    print(json.dumps(json.loads((DATA / "data_record.json").read_text()), indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cases", "mc", "pilot", "generate", "pack"])
    a = ap.parse_args()
    {"cases": cases, "mc": mc, "pilot": lambda: generate(True), "generate": generate, "pack": pack}[a.cmd]()

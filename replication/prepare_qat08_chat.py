"""Prepare (never train) the chat amendment. All outputs are isolated from QAT08.

Order: cases, prompts, pilot, generate, pack. GPU commands use systemd user units.
Only teacher generation is allowed before the user's training approval.
"""
from __future__ import annotations

import argparse
import ast
import collections
import concurrent.futures
import json
import re
import shutil
import time
import urllib.request
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

import freeze_17b as fz
import gsm8k_17b as gen
from freeze_qat08 import ngram_hashes
from score_17b import sha

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/qat08_chat"
WORK = ROOT / "work/qat08_chat"
DATA = WORK / "data"
PRIOR = ROOT / "results/qat08"
SOURCE = "nvidia/Daring-Anteater"
REVISION = "ae79f8ac44cf185fbd3250dbe46057a7f7c4ec40"
SEED = 20261002
QUOTAS = {"synthetic_conv": 6400, "synthetic_math": 1500,
          "synthetic_precise_instruction_following": 1000,
          "synthetic_complex_instruction": 1000, "synthetic_json_format_following": 100}
GUTENBERG = {
    "bleak_house": (1023, "Bleak House"),
    "return_of_the_native": (122, "The Return of the Native"),
    "mayor_of_casterbridge": (143, "The Mayor of Casterbridge"),
    "mill_on_the_floss": (6688, "The Mill on the Floss"),
    "adam_bede": (507, "Adam Bede"), "hard_times": (786, "Hard Times"),
    "david_copperfield": (766, "David Copperfield"),
    "wives_and_daughters": (4274, "Wives and Daughters"),
    "cranford": (394, "Cranford"), "villette": (9182, "Villette"),
    "tenant_of_wildfell_hall": (969, "The Tenant of Wildfell Hall"),
    "lorna_doone": (840, "Lorna Doone"), "ben_hur": (2145, "Ben-Hur"),
    "last_of_the_mohicans": (940, "The Last of the Mohicans"),
    "red_badge_of_courage": (73, "The Red Badge of Courage"),
}
VALIDATION = tuple(list(GUTENBERG)[:3])
TRAIN_TOKENS = 65_536_000


def tokenizer():
    return AutoTokenizer.from_pretrained(ROOT / "work/qwen35_08b/base", local_files_only=True)


def write_new(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def atomic_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def wilson(k, n):
    z = 1.96
    p = k / n
    den = 1 + z*z/n
    mid = (p + z*z/(2*n)) / den
    half = z * (p*(1-p)/n + z*z/(4*n*n))**0.5 / den
    return {"n": n, "correct": k, "mean": 100*p, "lo": 100*(mid-half), "hi": 100*(mid+half)}


def cases():
    if (OUT / "cases_record.json").exists():
        raise SystemExit("cases already sealed")
    # Check every earlier freeze script's literal Gutenberg inventory.
    previous = set()
    inventories = {}
    for path in sorted(ROOT.parent.rglob("freeze_*.py")):
        if path.name == "freeze_qat08_chat.py":
            continue
        values = []
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "GUTENBERG" for t in node.targets):
                values = list(ast.literal_eval(node.value).values())
                previous.update(values)
        inventories[str(path.relative_to(ROOT.parent))] = {"sha256": sha(path), "gutenberg_ids": values}
    assert not previous.intersection(g for g, title in GUTENBERG.values())
    OUT.mkdir(parents=True, exist_ok=True)
    fz.OUT, fz.BOOKS, fz.CASES = OUT, OUT / "books", OUT / "cases"
    fz.GUTENBERG = {name: g for name, (g, _) in GUTENBERG.items()}
    fz.VALIDATION = VALIDATION
    fz.CASES.mkdir(parents=True, exist_ok=True)
    titles = {}
    # Validate actual Title headers, not filenames, before constructing slices.
    fz.BOOKS.mkdir(exist_ok=True)
    for name, (gid, expected) in GUTENBERG.items():
        path = fz.BOOKS / f"{name}.raw.txt"
        if not path.exists():
            path.write_bytes(fz.fetch(f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"))
        text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
        match = re.search(r"^Title:\s*(.+)$", text[:12000], re.M)
        assert match, name
        title = match.group(1).strip()
        normalize = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
        assert normalize(expected) in normalize(title), (name, expected, title)
        titles[name] = {"gutenberg_id": gid, "expected": expected, "actual": title, "raw_sha256": sha(path)}
        print(name, gid, title, flush=True)
    entries, files = fz.books(tokenizer())
    old = json.loads((PRIOR / "protocol.json").read_text())
    for key in ("mmlu", "gsm8k", "retrieval"):
        entry = old[key]
        source = PRIOR / entry["file"]
        assert sha(source) == entry["sha256"]
        shutil.copyfile(source, OUT / entry["file"])
    write_new(OUT / "cases_record.json", {"book_entries": entries, "book_cases": files,
        "titles": titles, "prior_freeze_inventories": inventories,
        **{k: old[k] for k in ("mmlu", "gsm8k", "retrieval")}})


def read_cases():
    record = json.loads((OUT / "cases_record.json").read_text())
    paths = {"validation_books": record["book_cases"]["validation"],
             "heldout_books": record["book_cases"]["heldout"],
             **{k: record[k] for k in ("mmlu", "gsm8k", "retrieval")}}
    result = {}
    for key, entry in paths.items():
        path = OUT / entry["file"]
        assert sha(path) == entry["sha256"]
        if path.suffix == ".jsonl":
            rows = [json.loads(l) for l in path.read_text().splitlines()]
            result[key] = [(str(r["id"]), np.asarray(r["prompt_ids"], dtype=np.uint32)) for r in rows]
        else:
            result[key] = [(l.split("\t")[0], np.asarray(list(map(int, l.split("\t")[2].split())), dtype=np.uint32))
                           for l in path.read_text().splitlines()]
    return result


def normalize(s):
    return " ".join(re.findall(r"\w+", s.casefold()))


def grams(s):
    words = normalize(s).split()
    return {" ".join(words[i:i+13]) for i in range(max(0, len(words)-12))}


def benchmark_contents(tok):
    all_cases = read_cases()
    contents = []
    for key in ("mmlu", "gsm8k"):
        for _, ids in all_cases[key]:
            text = tok.decode(ids)
            text = text.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
            if key == "mmlu":
                text = text.split("\n\n", 1)[1]
            else:
                text = text.rsplit("\nPlease reason step by step", 1)[0]
            contents.append(text)
    return contents


def prompts():
    if (DATA / "prompts.jsonl").exists():
        raise SystemExit("prompts already selected")
    DATA.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    source = Path(hf_hub_download(SOURCE, "train.jsonl", revision=REVISION, repo_type="dataset",
                                  local_dir=DATA / "source"))
    card = Path(hf_hub_download(SOURCE, "README.md", revision=REVISION, repo_type="dataset",
                                local_dir=DATA / "source"))
    assert "license: cc-by-4.0" in card.read_text()
    tok = tokenizer()
    contents = benchmark_contents(tok)
    banned = set().union(*(grams(s) for s in contents))
    exact = {normalize(s) for s in contents}
    seen, pools, excluded = set(), {k: [] for k in QUOTAS}, collections.Counter()
    with source.open() as f:
        for index, line in enumerate(f):
            row = json.loads(line)
            category = row["dataset"]
            if category not in pools:
                continue
            turns = row["conversations"]
            if not turns or turns[0]["from"] != "User":
                excluded["not_first_user"] += 1
                continue
            text = turns[0]["value"].strip()
            norm = normalize(text)
            if norm in seen or not text or "<|" in text:
                excluded["duplicate_empty_or_special_token"] += 1
                continue
            if norm in exact or grams(text).intersection(banned):
                excluded["benchmark_overlap"] += 1
                continue
            ids = fz.chat(tok, text)
            if len(ids) > 384 or len(ids) < 24:
                excluded["length"] += 1
                continue
            seen.add(norm)
            pools[category].append({"id": index, "category": category, "prompt": text, "prompt_ids": ids})
    rng = np.random.default_rng(SEED)
    selected = []
    for key, quota in QUOTAS.items():
        assert len(pools[key]) >= quota, (key, len(pools[key]), quota)
        indexes = rng.permutation(len(pools[key]))[:quota]
        selected.extend(pools[key][int(i)] for i in indexes)
    selected = [selected[int(i)] for i in rng.permutation(len(selected))]
    (DATA / "prompts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in selected))
    write_new(DATA / "prompts_record.json", {"source": SOURCE, "revision": REVISION,
        "source_sha256": sha(source), "card_sha256": sha(card), "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "NVIDIA Daring-Anteater (Wang et al., HelpSteer2, 2024). Prompts only; original answers discarded.",
        "quotas": QUOTAS, "pool_sizes": {k: len(v) for k, v in pools.items()}, "excluded": dict(excluded),
        "seed": SEED, "prompts_sha256": sha(DATA / "prompts.jsonl"), "items": len(selected),
        "selection": "first User turn only; no source system message; 24..384 template tokens; seeded stratified selection",
        "benchmark_disjointness": "reject exact normalized content or ANY shared normalized 13-word gram with every MMLU/GSM8K case, excluding harness boilerplate; no semantic-paraphrase guarantee"})


def generate(pilot=False):
    prompts = [json.loads(l) for l in (DATA / "prompts.jsonl").read_text().splitlines()]
    record = json.loads((DATA / "prompts_record.json").read_text())
    assert sha(DATA / "prompts.jsonl") == record["prompts_sha256"]
    # Pilot is part of the eventual corpus, not an alternative model or recipe.
    target = DATA / "teacher.jsonl"
    done = {}
    if target.exists():
        done = {r["id"]: r for r in map(json.loads, target.read_text().splitlines())}
        assert len(done) == sum(1 for _ in target.open()), "duplicate response IDs"
    selected = prompts[:64] if pilot else prompts
    pending = [r for r in selected if r["id"] not in done]
    if not pending:
        print("all requested teacher responses already present", flush=True)
        return
    gen.PORT = 8092
    gen.N_PREDICT = 768
    WORK.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    count, tokens = 0, 0
    with gen.serve(ROOT / "work/qwen35_08b/base-f32.gguf", "model", WORK / "teacher-server.log"):
        with target.open("a", buffering=1) as f, concurrent.futures.ThreadPoolExecutor(gen.SLOTS) as pool:
            for row, out in zip(pending, pool.map(lambda r: gen.complete(r["prompt_ids"]), pending)):
                rec = {"id": row["id"], **out}
                f.write(json.dumps(rec) + "\n")
                count += 1
                tokens += out["tokens"] or 0
                if count % 64 == 0:
                    print(json.dumps({"new_responses": count, "total_responses": len(done)+count,
                                      "seconds": time.time()-t0, "tokens": tokens,
                                      "tokens_per_s": tokens/(time.time()-t0)}), flush=True)
    timing = {"responses_this_session": count, "seconds": time.time()-t0, "tokens": tokens,
              "tokens_per_s": tokens/(time.time()-t0), "pilot": pilot,
              "model_sha256": sha(ROOT / "work/qwen35_08b/base-f32.gguf"),
              "prompt_sha256": sha(DATA / "prompts.jsonl"), "thinking": "disabled in template (empty think block)",
              "generation": "F32 FP GGUF, pinned llama-server, greedy top_k=1, temperature=0, 8 slots, 768 token limit"}
    atomic_json(DATA / ("pilot_timing.json" if pilot else "teacher_timing.json"), timing)
    print(json.dumps(timing, indent=2), flush=True)


def audit_prompts():
    """Additional exact-question audit for questions too short to have a 13-word gram."""
    tok = tokenizer()
    contents = benchmark_contents(tok)
    questions = [normalize(x.split("\n\n", 1)[0]) for x in contents]
    short = sorted({q for q in questions if 4 <= len(q.split()) < 13}, key=lambda s: -len(s))
    pattern = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(s) for s in short) + r")(?!\w)")
    banned = set().union(*(grams(s) for s in contents))
    prompts = [json.loads(l) for l in (DATA / "prompts.jsonl").read_text().splitlines()]
    matches = []
    for row in prompts:
        norm = normalize(row["prompt"])
        if pattern.search(norm) or norm in questions or grams(row["prompt"]).intersection(banned):
            matches.append(row["id"])
    assert not matches, f"selected prompts overlap evaluation questions: {matches}"
    responses = [json.loads(l) for l in (DATA / "teacher.jsonl").read_text().splitlines()]
    assert len(responses) == 64
    prompts_by_id = {r["id"]: r for r in prompts}
    checked = 0
    for out in responses:
        if out["stop_type"] != "eos":
            continue
        row = prompts_by_id[out["id"]]
        text = out["text"].strip()
        rendered = tok.apply_chat_template([{"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": text}], tokenize=False,
            add_generation_prompt=False, enable_thinking=False)
        ids = row["prompt_ids"] + tok.encode(text, add_special_tokens=False) + tok.encode("<|im_end|>\n", add_special_tokens=False)
        assert tok.encode(rendered, add_special_tokens=False) == ids
        checked += 1
    shutil.copyfile(DATA / "teacher.jsonl", DATA / "pilot_responses.jsonl")
    write_new(DATA / "prompt_audit.json", {"prompts": len(prompts), "questions_checked": len(questions),
        "short_questions_checked": len(short), "overlapping_prompt_ids": matches,
        "policy": "normalized full-question equality for all questions; substring check for 4..12-word questions; any shared 13-word content gram otherwise; shared 1..3-word topics are not test-item duplicates",
        "prompts_sha256": sha(DATA / "prompts.jsonl"), "finished_pilot_template_identities_checked": checked,
        "pilot_responses_sha256": sha(DATA / "pilot_responses.jsonl")})
    print("prompt disjointness and pilot chat-template audit passed", flush=True)


def pack():
    if (DATA / "data_record.json").exists():
        raise SystemExit("mixed data already sealed")
    prompts = [json.loads(l) for l in (DATA / "prompts.jsonl").read_text().splitlines()]
    responses = {r["id"]: r for r in map(json.loads, (DATA / "teacher.jsonl").read_text().splitlines())}
    assert set(responses) == {r["id"] for r in prompts}, "teacher generation incomplete"
    tok = tokenizer()
    contents = benchmark_contents(tok)
    banned = set().union(*(grams(s) for s in contents))
    end = tok.encode("<|im_end|>\n", add_special_tokens=False)
    chats, excluded, kept = [], collections.Counter(), []
    for row in prompts:
        out = responses[row["id"]]
        text = out["text"].strip()  # the pinned Qwen template trims assistant content
        if out["stop_type"] != "eos" or not text.strip():
            excluded["not_naturally_terminated_or_empty"] += 1
            continue
        if "<think>" in text or "</think>" in text or "<|" in text:
            excluded["special_token_or_thinking"] += 1
            continue
        if grams(text).intersection(banned):
            excluded["response_benchmark_overlap"] += 1
            continue
        answer = tok.encode(text, add_special_tokens=False)
        ids = row["prompt_ids"] + answer + end
        if len(ids) > 1024:
            excluded["conversation_over_1024"] += 1
            continue
        # Assert the concatenation is the teacher's exact non-thinking chat transcript.
        rendered = tok.apply_chat_template([{"role": "user", "content": row["prompt"]},
                                           {"role": "assistant", "content": text}], tokenize=False,
                                          add_generation_prompt=False, enable_thinking=False)
        assert ids == tok.encode(rendered, add_special_tokens=False), row["id"]
        chats.extend(ids)
        kept.append({"id": row["id"], "category": row["category"], "tokens": len(ids),
                     "prompt_tokens": len(row["prompt_ids"]), "response_tokens": len(answer)})
    unique = np.asarray(chats, dtype=np.uint32)
    assert len(kept) >= 3000 and len(unique) >= 1_000_000, (len(kept), len(unique), excluded)
    unique.tofile(DATA / "chat_unique.u32")
    write_new(DATA / "kept_chats.json", kept)
    n = TRAIN_TOKENS // 1024
    train = np.memmap(DATA / "train.u32", dtype=np.uint32, mode="w+", shape=(n, 1024))
    fine = np.memmap(ROOT / "work/qat08/data/train.u32", dtype=np.uint32, mode="r")
    fine_seq, chat_seq = 0, 0
    for seq in range(n):
        if seq % 4 == 3:
            pos = (chat_seq*1024 + np.arange(1024)) % len(unique)
            train[seq] = unique[pos]
            chat_seq += 1
        else:
            train[seq] = fine[fine_seq*1024:(fine_seq+1)*1024]
            fine_seq += 1
    train.flush()
    del train
    shutil.copyfile(ROOT / "work/qat08/data/monitor.u32", DATA / "monitor.u32")
    write_new(DATA / "data_record.json", {"train_tokens": TRAIN_TOKENS, "chat_tokens": chat_seq*1024,
        "fineweb_tokens": fine_seq*1024, "mix_ratio_tokens": 0.25, "unique_chat_tokens": len(unique),
        "chat_corpus_passes": chat_seq*1024/len(unique), "kept_chats": len(kept), "excluded": dict(excluded),
        "categories": dict(collections.Counter(r["category"] for r in kept)),
        "order": "3 contiguous FineWeb 1024-token sequences, then 1 chat sequence; chat transcripts concatenated with im_end, cycled in fixed prompt order; no padding/masking; FineWeb starts at QAT08 offset 0",
        "train_sha256": sha(DATA / "train.u32"), "monitor_sha256": sha(DATA / "monitor.u32"),
        "unique_chat_sha256": sha(DATA / "chat_unique.u32"), "kept_chats_sha256": sha(DATA / "kept_chats.json"),
        "teacher_responses_sha256": sha(DATA / "teacher.jsonl"),
        "prompts_record": json.loads((DATA / "prompts_record.json").read_text()),
        "prior_fineweb_record": json.loads((ROOT / "work/qat08/data/data_record.json").read_text())})
    print(json.dumps(json.loads((DATA / "data_record.json").read_text()), indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cases", "prompts", "pilot", "audit", "generate", "pack"])
    a = ap.parse_args()
    {"cases": cases, "prompts": prompts, "pilot": lambda: generate(True), "audit": audit_prompts,
     "generate": generate, "pack": pack}[a.cmd]()

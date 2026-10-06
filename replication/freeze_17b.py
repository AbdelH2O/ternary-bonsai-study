"""Freeze the matched 1.7B comparison (Phase 1) before any arm is built or scored.

Arms: Qwen3-1.7B FP, PrismML's published Ternary-Bonsai-1.7B PQ2_0 file, and our PQ2_0 arms built
from the same Qwen3-1.7B ancestor with the same tensor policy (pinned absmax, least squares, and the
0.8B-selected GPTQ-style rule unrotated and in an H1024 folded basis). Evaluation: fresh-book NLL,
MMLU-Redux 2.0 letter logits, GSM8K greedy generation, and a synthetic retrieval margin.

    .venv/bin/python analysis/bonsai2/replication/freeze_17b.py      # refuses to overwrite
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inspect_gguf import inspect  # noqa: E402
from phase0 import align  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/q17b"
BOOKS = OUT / "books"
CASES = OUT / "cases"
WORK = ROOT / "work/qwen3_17b"
DOCS = ROOT.parent / "fresh_state/documents"
SEED = 20261001
GUTENBERG = {  # none used by any earlier experiment in this repository
    "sense_and_sensibility": 161, "call_of_the_wild": 215, "jekyll_and_hyde": 43,
    "huckleberry_finn": 76, "crime_and_punishment": 2554, "persuasion": 105, "dubliners": 2814,
    "wizard_of_oz": 55, "twenty_thousand_leagues": 164, "around_the_world": 103,
    "hound_of_the_baskervilles": 2852, "secret_garden": 113, "anne_of_green_gables": 45,
    "three_musketeers": 1257, "scarlet_letter": 33,
}
VALIDATION = ("sense_and_sensibility", "call_of_the_wild", "jekyll_and_hyde")
HELDOUT = tuple(b for b in GUTENBERG if b not in VALIDATION)
SLICE_BYTES, SLICE_TOKENS, FIRST_TARGET = 8192, 1024, 64
BANDS = {"early": [64, 128], "mid": [128, 512], "late": [512, 1024]}
CALIBRATION_BOOKS, CAL_WINDOWS, WINDOW = ("emma.txt", "little_women.txt", "middlemarch.txt"), (22, 21, 21), 512
MMLU = ("edinburgh-dawg/mmlu-redux-2.0", "372ea425445d51e1ba1188c56e5e893f8138621f")
GSM8K = ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866")
REGISTRIES, ENTRIES = 200, 30
NORM_VOCAB = 151669
PRISM = WORK / "prism_gguf/Ternary-Bonsai-1.7B-PQ2_0.gguf"
FIRST_NAMES = ("Ada Alan Alice Amara Arjun Beatrix Bruno Camila Carlos Chen Clara Dmitri Elena Emeka "
               "Farah Felix Grace Hana Hugo Ines Ivan Jonas Julia Kai Kofi Lars Leila Lucia Marco Maya "
               "Nadia Nikolai Noor Oscar Priya Rafael Rosa Samir Sofia Tariq Tessa Tomas Uma Victor "
               "Wen Yara Yusuf Zara Zoe Emil").split()
LAST_NAMES = ("Abbott Alvarez Andersen Baker Bianchi Brennan Castillo Dubois Eriksen Fischer Garcia "
              "Hansen Haddad Ibrahim Ivanova Jensen Kaur Kowalski Laurent Lindqvist Moreau Murphy "
              "Nakamura Novak Okafor Olsen Petrov Quinn Rahman Rossi Santos Schmidt Silva Tanaka "
              "Torres Varga Vogel Walsh Weber Yamamoto Zhang Ziegler Mendes Horvat Duarte Keller "
              "Lambert Morales Nguyen Park").split()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def strip_gutenberg(raw: bytes) -> bytes:
    lines = raw.decode("utf-8-sig").replace("\r\n", "\n").split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("*** START OF"))
    end = next(i for i, l in enumerate(lines) if l.startswith("*** END OF"))
    return "\n".join(lines[start + 1:end]).strip().encode("utf-8") + b"\n"


def tsv(rows: list[tuple[str, str, list[int], str]], path: Path) -> str:
    path.write_text("".join(f"{i}\t{k}\t{' '.join(map(str, ids))}\t{a}\n" for i, k, ids, a in rows))
    return digest(path.read_bytes())


def books(tok) -> tuple[list[dict], dict]:
    BOOKS.mkdir(parents=True, exist_ok=True)
    entries, rows = [], {"validation": [], "heldout": []}
    for name, gid in GUTENBERG.items():
        raw_path = BOOKS / f"{name}.raw.txt"
        if not raw_path.exists():
            raw_path.write_bytes(fetch(f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"))
        raw = raw_path.read_bytes()
        body = strip_gutenberg(raw)
        (BOOKS / f"{name}.txt").write_bytes(body)
        split = "validation" if name in VALIDATION else "heldout"
        title = raw[:4000].decode("utf-8-sig", errors="replace").split("\n")[0].strip()
        for fraction in (0.25, 0.65):
            start = body.find(b"\n", int(len(body) * fraction)) + 1
            text = body[start:start + SLICE_BYTES].decode("utf-8", errors="replace")
            ids = tok(text, add_special_tokens=False)["input_ids"]
            assert len(ids) >= SLICE_TOKENS, (name, len(ids))
            case = f"{name}-{int(100 * fraction)}"
            rows[split].append((case, "nll", ids[:SLICE_TOKENS], str(FIRST_TARGET)))
            entries.append({"id": case, "split": split, "book": name, "gutenberg_id": gid, "title_line": title,
                            "raw_sha256": digest(raw), "body_sha256": digest(body), "byte_offset": start})
    files = {s: {"file": f"cases/{s}_books.tsv", "sha256": tsv(r, CASES / f"{s}_books.tsv"), "cases": len(r)}
             for s, r in rows.items()}
    return entries, files


def calibration(tok) -> dict:
    windows, records = [], []
    for name, count in zip(CALIBRATION_BOOKS, CAL_WINDOWS):
        source = (DOCS / name).read_bytes()
        ids = tok(source.decode("utf-8"), add_special_tokens=False)["input_ids"]
        lo, hi = int(0.05 * len(ids)), int(0.95 * len(ids)) - WINDOW
        starts = np.linspace(lo, hi, count).astype(np.int64)
        windows.extend(ids[s:s + WINDOW] for s in starts)
        records.append({"book": name, "source_sha256": digest(source), "tokens": len(ids), "starts": starts.tolist()})
    array = np.asarray(windows, dtype=np.int32)
    assert array.shape == (64, WINDOW)
    np.save(OUT / "calibration_ids.npy", array)
    return {"rule": "v1 pilot rule re-tokenized with the Qwen3 tokenizer: 22/21/21 windows of 512 tokens, "
                    "evenly spaced over the 5%-95% token range of each book",
            "books": records, "ids_file": "calibration_ids.npy",
            "ids_sha256": digest((OUT / "calibration_ids.npy").read_bytes())}


def chat(tok, text: str) -> list[int]:
    prompt = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    return tok(prompt, add_special_tokens=False)["input_ids"]


def mmlu(tok) -> dict:
    repo, rev = MMLU
    listing = json.loads(fetch(f"https://huggingface.co/api/datasets/{repo}/revision/{rev}"))
    subjects = sorted({s["rfilename"].split("/")[0] for s in listing["siblings"] if s["rfilename"].endswith(".arrow")})
    letters = [tok.encode(x, add_special_tokens=False) for x in "ABCD"]
    assert all(len(x) == 1 for x in letters)
    letters = [x[0] for x in letters]
    rows, counts, sources = [], {}, {}
    for subject in subjects:
        data = fetch(f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{subject}/data-00000-of-00001.arrow")
        sources[subject] = digest(data)
        table = pa.ipc.open_stream(io.BytesIO(data)).read_all().to_pylist()
        kept = 0
        for i, item in enumerate(table):
            if item["error_type"] != "ok" or len(item["choices"]) != 4 or item["answer"] not in range(4):
                continue
            options = "\n".join(f"{l}. {c}" for l, c in zip("ABCD", item["choices"]))
            text = (f"The following is a multiple choice question about {subject.replace('_', ' ')}. "
                    f"Answer with only the letter (A, B, C, or D) of the correct option.\n\n"
                    f"{item['question']}\n\n{options}")
            ids = chat(tok, text)
            if len(ids) > 2000:
                continue
            gold = "ABCD"[item["answer"]]
            rows.append((f"{subject}/{i}/{gold}", "choice", ids, ",".join(map(str, letters))))
            kept += 1
        counts[subject] = kept
    return {"repo": repo, "revision": rev, "file": "cases/mmlu.tsv", "sha256": tsv(rows, CASES / "mmlu.tsv"),
            "items": len(rows), "subjects": counts, "source_sha256": sources, "letter_token_ids": letters,
            "filter": "error_type == 'ok', exactly 4 choices, prompt <= 2000 tokens",
            "prompt": "zero-shot, Qwen3 chat template, thinking disabled (empty think block), answer = next token",
            "scores": "accuracy = argmax over the four letter log-probabilities; margin = log p(gold) - max log p(wrong letter)"}


def gsm8k(tok) -> dict:
    repo, rev = GSM8K
    data = fetch(f"https://huggingface.co/datasets/{repo}/resolve/{rev}/main/test-00000-of-00001.parquet")
    table = pq.read_table(io.BytesIO(data)).to_pylist()
    lines = []
    for i, item in enumerate(table):
        gold = item["answer"].split("####")[-1].strip().replace(",", "")
        text = f"{item['question']}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        lines.append(json.dumps({"id": i, "gold": gold, "prompt_ids": chat(tok, text)}))
    path = CASES / "gsm8k.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return {"repo": repo, "revision": rev, "source_sha256": digest(data), "file": "cases/gsm8k.jsonl",
            "sha256": digest(path.read_bytes()), "items": len(lines),
            "generation": "llama-server /completion with prompt token IDs; temperature 0, top_k 1, "
                          "n_predict 1024, stop at end-of-generation; 8 parallel slots",
            "parse": "last \\boxed{...} content, else last number in the output; remove commas, '$', "
                     "and a trailing '.'; correct if numerically equal to the gold answer after '####'"}


def retrieval(tok) -> dict:
    rng = random.Random(SEED)
    rows = []
    for r in range(REGISTRIES):
        names = rng.sample([f"{f} {l}" for f in FIRST_NAMES for l in LAST_NAMES], ENTRIES)
        codes = rng.sample(range(1000, 10000), ENTRIES)
        q = r % ENTRIES
        w = (q + 1 + rng.randrange(ENTRIES - 1)) % ENTRIES
        body = "\n".join(f"{n}: {c}" for n, c in zip(names, codes))
        prefix = (f"Access code registry\n{body}\n\nQuestion: What is the access code for {names[q]}?\n"
                  f"Answer: The access code for {names[q]} is")
        p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
        for label, code in (("gold", codes[q]), ("wrong", codes[w])):
            c_ids = tok(f" {code}", add_special_tokens=False)["input_ids"]
            rows.append((f"r{r:03d}/{label}/q{q}", "nll", p_ids + c_ids, str(len(p_ids))))
    return {"file": "cases/retrieval.tsv", "sha256": tsv(rows, CASES / "retrieval.tsv"), "registries": REGISTRIES,
            "entries_per_registry": ENTRIES, "seed": SEED,
            "design": "raw text; 30 distinct 'First Last: NNNN' lines; the queried entry cycles through all 30 "
                      "positions; the wrong answer is another entry's code from the same registry (value swap)",
            "score": "margin = NLL(wrong code tokens) - NLL(gold code tokens), nat/answer; accuracy = margin > 0"}


def payloads(path: Path) -> dict[str, tuple[str, bytes]]:
    _, _, rows, header = inspect(path)
    out = {}
    with path.open("rb") as f:
        for r in rows:
            f.seek(align(header) + r["offset"])
            out[r["tensor"]] = (r["storage"], f.read(r["bytes"]))
    return out


def tensor_policy() -> dict:
    prism, repack, template = (payloads(p) for p in (PRISM, WORK / "prism-unpacked-pq2.gguf", WORK / "base-pq2.gguf"))
    assert set(prism) == set(repack) == set(template)
    assert all(prism[n] == repack[n] for n in prism), "repacked unpacked checkpoint differs from the published file"
    assert all(prism[n][0] == template[n][0] for n in prism)
    counts: dict[str, int] = {}
    for storage, _ in prism.values():
        counts[storage] = counts.get(storage, 0) + 1
    return {"source": "Prism's published PQ2_0 file", "tensors": len(prism), "storage_counts": counts,
            "pq2_0": "all 196 decoder projections and the tied token embedding; no separate output tensor",
            "f32": "all 1-D norms (attn, ffn, q/k norms, output norm)",
            "quantizer_command": "bin/cuda/llama-quantize --pure --token-embedding-type PQ2_0 IN-f32.gguf OUT.gguf PQ2_0 8",
            "repack_check": ("pinned converter + that command on prism-ml/Ternary-Bonsai-1.7B-unpacked reproduces all "
                             "310 tensor payloads of the published file byte for byte"),
            "prism_sha256": digest(PRISM.read_bytes())}


def main() -> None:
    if (OUT / "protocol.json").exists():
        raise SystemExit("protocol already frozen; write a versioned amendment instead")
    CASES.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(WORK / "base")
    prism_tok = WORK / "bonsai_unpacked/tokenizer.json"
    assert digest(prism_tok.read_bytes()) == digest((WORK / "base/tokenizer.json").read_bytes())
    book_entries, book_files = books(tok)
    policy = tensor_policy()
    protocol = {
        "date": "2026-10-01",
        "role": ("Phase 1 matched comparison on Qwen3-1.7B against PrismML's public Ternary-Bonsai-1.7B; "
                 "gen-1, unrotated, full attention; no Bonsai 2 recipe or quality claim"),
        "ancestor": "Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "prism_reference": {"gguf": "prism-ml/Ternary-Bonsai-1.7B-gguf@983b5dec2ff16aab79990711ba0f828a499a7e6a "
                                    "Ternary-Bonsai-1.7B-PQ2_0.gguf (scored unchanged)",
                            "unpacked": "prism-ml/Ternary-Bonsai-1.7B-unpacked@3aca840085293d026ce6f6b80fafdae937fd2eeb "
                                        "(static weight analysis only)"},
        "runtime": "prism-b10735-842b188 (bin/cuda), source archive sha256 84ec38b7...; scorer score_17b.cpp",
        "tensor_policy": policy,
        "arms": {
            "fp": ("base_tied-f32.gguf: pinned converter, F32, unquantized reference; the duplicate lm_head "
                   "(bitwise equal to embed_tokens) is dropped so the output is tied as in Prism's file"),
            "prism": "PrismML's published PQ2_0 file, unchanged, with its YaRN rope metadata (factor 4, original 8192)",
            "prism_noyarn": ("the same file with rope scaling forced to none at runtime, matching the Qwen3 ancestor's "
                             "rope configuration used by every other arm"),
            "absmax": "pinned llama-quantize PQ2_0 of base-f32.gguf with the Prism tensor policy (data-free)",
            "ls": "independent-group least-squares ternary rule (0.8B pilot 'ls'), unrotated, data-free",
            "gptq": ("0.8B-selected v1 GPTQ-style ternary rule, free zero count, unrotated: 64 calibration windows, "
                     "damping 0.01, 128-column blocks, layer-sequential, tied embedding last on final-norm inputs"),
            "gptqh": ("same rule in a folded H1024 basis: normalized signed Sylvester Hadamard, block 1024, "
                      "explicit signs from numpy default_rng(20260930 + width) for widths 2048 and 6144, "
                      "all 196 projections plus the tied embedding folded; converter schema 3"),
        },
        "calibration": calibration(tok),
        "splits": {"validation_books": list(VALIDATION), "heldout_books": list(HELDOUT)},
        "book_entries": book_entries,
        "book_cases": book_files,
        "book_scoring": {"slice": f"{SLICE_BYTES} bytes after the next newline at 25% and 65% of the stripped body",
                         "tokens": f"first {SLICE_TOKENS} Qwen3 token IDs, no special tokens",
                         "targets": f"positions {FIRST_TARGET}-{SLICE_TOKENS - 1}", "bands": BANDS,
                         "primary": "mean NLL over all targets 64-1023; book = mean of its two slices"},
        "mmlu": mmlu(tok),
        "gsm8k": gsm8k(tok),
        "retrieval": retrieval(tok),
        "scoring_backend": (f"CUDA, all layers offloaded, n_ctx 2048, n_ubatch 512; softmax over token IDs "
                            f"0..{NORM_VOCAB - 1} for every arm (Prism trims the vocabulary to {NORM_VOCAB}, "
                            "Qwen3 pads to 151936); identical token IDs for every arm"),
        "selection": {
            "method": ("between gptq and gptqh: lower mean primary NLL over the three validation books; lexical tie "
                       "break"),
            "prism_reference": ("between prism and prism_noyarn: lower mean primary NLL over the validation books; "
                                "the winner is the reference for every held-out comparison, the other is reported"),
            "timing": "both written to selection.json before any held-out metric of any arm is computed"},
        "heldout_arms": {"books_mmlu_retrieval": ["fp", "prism", "prism_noyarn", "absmax", "ls", "gptq", "gptqh"],
                         "gsm8k": ["fp", "prism_reference", "gptq", "gptqh"]},
        "pre_freeze_checks": {
            "fold_identity": ("calibration-text smoke test on GPU: base_tied-f32 vs folded-f32 per-token NLL max "
                              "|difference| 0.034 over 2,880 tokens, window means within 0.0003"),
            "gsm8k_timing": ("64 GSM8K train items (not test): fp 81.2% at 386 tok/s, prism 76.6% at 1,312 tok/s, "
                             "3/64 prism outputs hit the 1024-token limit")},
        "decision": {
            "terms": "below, 'prism' means the selected Prism reference configuration",
            "primary_comparison": "selected arm minus the selected Prism reference on each held-out metric",
            "non_inferiority_margins": {
                "book_nll": "upper 95% t bound (12 books) of selected - prism <= +0.10 nat/token",
                "mmlu_accuracy": "lower 95% paired normal bound (items) of selected - prism >= -3.0 points",
                "gsm8k_accuracy": "lower 95% paired normal bound (items) of selected - prism >= -3.0 points",
                "retrieval_margin": "lower 95% t bound (registries) of selected - prism >= -0.25 nat/answer",
            },
            "non_inferior": "all four margins pass",
            "prism_gap_closure": ("secondary, book NLL: (ls - selected) / (ls - prism), reported only if "
                                  "ls - prism has a 95% t interval excluding zero"),
            "readings": {
                "near_prism": "book NLL selected - prism <= 0.10: one-shot reconstruction nearly matches Prism here",
                "moderate_gap": "0.10 < selected - prism <= 0.50",
                "large_gap": ("selected - prism > 0.50: Prism's model is far better than one-shot reconstruction "
                              "from the same ancestor; supports a stronger ingredient such as training (inferred)"),
            },
            "harness_check": ("descriptive only: fp and prism MMLU/GSM8K next to published 66.8/52.9 and 83.1/74.2; "
                              "prompts, harness and scoring differ from EvalScope, so no agreement threshold"),
            "claim_limit": ("one gen-1 1.7B model pair; English public-domain books may be in pretraining data; "
                            "logit-scored MMLU and one GSM8K prompt differ from the publisher's harness; "
                            "deterministic methods without seeds; says nothing about Bonsai 2's recipe"),
        },
        "byte_ceiling": "each packed arm <= prism file bytes + 1,000,000 (folded metadata)",
        "static_analysis": ("descriptive, after builds: per-tensor sign agreement, zero rate, support overlap and "
                            "scale ratios for prism vs ancestor and vs our arms; layer output error under FP-input "
                            "Hessians from the calibration windows"),
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(f"froze protocol {digest((OUT / 'protocol.json').read_bytes())}: "
          f"{sum(f['cases'] for f in book_files.values())} book slices, {protocol['mmlu']['items']} MMLU items, "
          f"{protocol['gsm8k']['items']} GSM8K items, {REGISTRIES} registries")


if __name__ == "__main__":
    main()

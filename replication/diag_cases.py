"""Development-only cases and the immutable plan for the read-out diagnostics.

No MMLU-Redux, GSM8K-test or held-out book item is used. Run with the project venv:
    python diag_cases.py cases    # pinned MMLU validation -> results/diag_readout/cases/ (network: huggingface.co)
    python diag_cases.py freeze   # results/diag_readout/plan.json, exclusive; before any GPU stage
    python diag_cases.py verify
"""
from __future__ import annotations

import argparse
import collections
import json
import platform
import random
from pathlib import Path

import freeze_17b as fz
from prepare_qat08_chat import benchmark_contents, normalize, tokenizer, write_new
from score_17b import sha

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/diag_readout"
WORK = ROOT / "work/diag_readout"
CASES = OUT / "cases"
MMLU_REPO, MMLU_REV, MMLU_FILE = "cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe", "all/validation-00000-of-00001.parquet"
LETTER_IDS = [32, 33, 34, 35]
PREFIX = [248045, 846, 198]
SUFFIX = [248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]
MAX_TOKENS = 2000
BIND_SEED, GEN_SEED, KL_SEED = 20261004, 20261005, 20261006
BIND_ITEMS, BIND_ENTRIES, GEN_ITEMS, KL_ITEMS = 400, 8, 300, 200
INSTRUCTION = "Answer with only the letter (A, B, C, or D) of the correct option."
ARMS = {
    "fp": "work/qwen35_08b/base-f32.gguf",
    "gptq": "work/qwen35_08b/folded-pq2-gptq.gguf",
    **{f"qat_42-{c}": f"work/qat08/qat_42-{c}.gguf" for c in ("t8m", "t16m", "t33m", "final")},
    **{f"qat_chat_42-{c}": f"work/qat08_chat/qat_chat_42-{c}.gguf" for c in ("init", "t8m", "t16m", "t33m", "final")},
}
TRAJECTORY = {"qat_42": [f"qat_42-{c}" for c in ("t8m", "t16m", "t33m", "final")],
              "qat_chat_42": [f"qat_chat_42-{c}" for c in ("t8m", "t16m", "t33m", "final")]}
GENERATIVE_ARMS = ["fp", "qat_42-final", "qat_chat_42-final"]
KL_ARMS = {"qat_42-final": {"rule": "top86", "latents": "work/qat08/qat_42/latent_final.pt"},
           "qat_chat_42-final": {"rule": "top86", "latents": "work/qat08_chat/qat_chat_42/latent_final.pt"}}
SETS = ("letter", "cloze_chat", "cloze_raw", "binding")
RULE = {"fp_binding_min": 95.0, "fp_letter_pairwise_min": 60.0, "fp_cloze_pairwise_lo_min": 55.0, "readout_broken_below": 70.0,
        "readout_intact_at_least": 90.0, "knowledge_pairwise_lo_above": 52.0, "trajectory_rise_points": 2.0}
IMPLEMENTATION = ("diag_metrics.py", "diag_cases.py", "diag_run.py", "run_diag_readout.sh", "test_diag_readout.py",
                  "freeze_17b.py", "prepare_qat08_chat.py", "score_17b.py", "gsm8k_17b.py", "qat_08b.py",
                  "make_gptq_pilot.py", "work/qwen3_17b/score_17b", "../../../bin/cuda/llama-server")


def runtime_hashes() -> dict[str, str]:
    """The pinned runtime libraries score_17b and llama-server link against, keyed like the chat protocol."""
    return {"../../../bin/cuda/" + p.name: sha(p) for p in sorted((ROOT.parents[2] / "bin/cuda").glob("*.so*")) if p.is_file()}


def banned_questions(tok) -> set[str]:
    """Normalized question text of every frozen MMLU-Redux and GSM8K case."""
    return {normalize(x.split("\n\n", 1)[0]) for x in benchmark_contents(tok)}


def keep_item(item: dict, banned: set[str]) -> bool:
    return (len(item["choices"]) == 4 and item["answer"] in range(4)
            and all(str(c).strip() for c in item["choices"]) and normalize(item["question"]) not in banned)


def options_text(choices) -> str:
    return "\n".join(f"{l}. {c}" for l, c in zip("ABCD", choices))


def letter_case(tok, idx: int, item: dict) -> tuple[str, str, list[int], str]:
    subject = item["subject"].replace("_", " ")
    text = (f"The following is a multiple choice question about {subject}. {INSTRUCTION}\n\n"
            f"{item['question']}\n\n{options_text(item['choices'])}")
    return (f"{item['subject']}/{idx}/{'ABCD'[item['answer']]}", "choice", fz.chat(tok, text),
            ",".join(map(str, LETTER_IDS)))


def cloze_cases(tok, idx: int, item: dict, raw: bool) -> list[tuple]:
    subject = item["subject"].replace("_", " ")
    if raw:
        prefix = tok(f"Question: {item['question']}\nAnswer:", add_special_tokens=False)["input_ids"]
        encode = lambda c: tok(f" {c}", add_special_tokens=False)["input_ids"]
    else:
        prefix = fz.chat(tok, f"The following is a question about {subject}. Answer briefly.\n\n{item['question']}")
        encode = lambda c: tok(str(c), add_special_tokens=False)["input_ids"]
    gold = "ABCD"[item["answer"]]
    return [(f"{item['subject']}/{idx}/{gold}/{k}", "nll", prefix + encode(c), str(len(prefix)))
            for k, c in enumerate(item["choices"])]


def binding_cases(tok, seed: int = BIND_SEED, n: int = BIND_ITEMS, entries: int = BIND_ENTRIES) -> list[tuple]:
    """In-context multiple choice: the answer is stated in a short registry inside the prompt."""
    rng = random.Random(seed)
    pool = [f"{f} {l}" for f in fz.FIRST_NAMES for l in fz.LAST_NAMES]
    rows = []
    for i in range(n):
        names = rng.sample(pool, entries)
        codes = rng.sample(range(1000, 10000), entries)
        q = rng.randrange(entries)
        others = rng.sample([c for j, c in enumerate(codes) if j != q], 3)
        gold = i % 4
        options = others[:gold] + [codes[q]] + others[gold:]
        body = "\n".join(f"{a}: {c}" for a, c in zip(names, codes))
        text = (f"The following is a multiple choice question about an access code registry. {INSTRUCTION}\n\n"
                f"Access code registry\n{body}\n\nWhat is the access code for {names[q]}?\n\n{options_text(options)}")
        rows.append((f"bind{i:03d}/{'ABCD'[gold]}", "choice", fz.chat(tok, text), ",".join(map(str, LETTER_IDS))))
    return rows


def generative_case(tok, idx: int, item: dict) -> dict:
    subject = item["subject"].replace("_", " ")
    text = (f"The following is a multiple choice question about {subject}. Think step by step, then finish with "
            f"'Answer: X' where X is the letter (A, B, C, or D) of the correct option.\n\n"
            f"{item['question']}\n\n{options_text(item['choices'])}")
    gold = "ABCD"[item["answer"]]
    return {"id": f"{item['subject']}/{idx}/{gold}", "gold": gold, "prompt_ids": fz.chat(tok, text)}


def _tsv(name: str, rows: list[tuple]) -> dict:
    return {"file": f"cases/{name}", "sha256": fz.tsv(rows, CASES / name), "cases": len(rows)}


def cases() -> None:
    if (OUT / "cases_record.json").exists():
        raise SystemExit("cases already sealed")
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    CASES.mkdir(parents=True, exist_ok=True)
    tok = tokenizer()
    assert [tok.encode(x, add_special_tokens=False) for x in "ABCD"] == [[i] for i in LETTER_IDS]
    banned = banned_questions(tok)
    source = Path(hf_hub_download(MMLU_REPO, MMLU_FILE, revision=MMLU_REV, repo_type="dataset",
                                  local_dir=WORK / "source"))
    table = pq.read_table(source).to_pylist()
    kept, dropped = [], collections.Counter()
    for idx, item in enumerate(table):
        if not keep_item(item, banned):
            dropped["filtered_or_heldout_overlap"] += 1
            continue
        letter = letter_case(tok, idx, item)
        assert letter[2][:3] == PREFIX and letter[2][-9:] == SUFFIX, letter[0]
        chat_rows, raw_rows = cloze_cases(tok, idx, item, False), cloze_cases(tok, idx, item, True)
        if max(len(r[2]) for r in [letter, *chat_rows, *raw_rows]) > MAX_TOKENS:
            dropped["length"] += 1
            continue
        if any(int(r[3]) >= len(r[2]) for r in chat_rows + raw_rows):
            dropped["empty_option"] += 1
            continue
        kept.append((idx, item, letter, chat_rows, raw_rows))
    files = {"letter": _tsv("letter.tsv", [k[2] for k in kept]),
             "cloze_chat": _tsv("cloze_chat.tsv", [r for k in kept for r in k[3]]),
             "cloze_raw": _tsv("cloze_raw.tsv", [r for k in kept for r in k[4]]),
             "binding": _tsv("binding.tsv", binding_cases(tok))}
    order = random.Random(GEN_SEED).sample(range(len(kept)), GEN_ITEMS)
    generative = [generative_case(tok, kept[j][0], kept[j][1]) for j in order]
    (CASES / "generative.jsonl").write_text("".join(json.dumps(g) + "\n" for g in generative))
    files["generative"] = {"file": "cases/generative.jsonl", "sha256": sha(CASES / "generative.jsonl"),
                           "cases": len(generative)}
    kl_ids = [kept[j][2][0] for j in random.Random(KL_SEED).sample(range(len(kept)), KL_ITEMS)]
    write_new(OUT / "cases_record.json", {
        "source": {"repo": MMLU_REPO, "revision": MMLU_REV, "file": MMLU_FILE, "sha256": sha(source),
                   "license": "MIT (dataset card)"},
        "items_in_source": len(table), "kept_items": len(kept), "dropped": dict(dropped),
        "subjects": dict(collections.Counter(k[1]["subject"] for k in kept)),
        "gold_letters": dict(collections.Counter(k[2][0].rsplit("/", 1)[1] for k in kept)),
        "files": files, "kl_ids": kl_ids,
        "exclusion": "normalized question equality with every frozen MMLU-Redux and GSM8K case; 4 non-empty choices; <=2000 tokens"})
    print(json.dumps({"kept_items": len(kept), "dropped": dict(dropped), "files": files}, indent=2))


def freeze() -> None:
    chat_runtime = json.loads((ROOT / "results/qat08_chat/protocol.json").read_text())["runtime_sha256"]
    assert runtime_hashes() == chat_runtime, "runtime differs from the chat run's; reused comparisons would mix kernels"
    record = json.loads((OUT / "cases_record.json").read_text())
    for entry in record["files"].values():
        assert sha(OUT / entry["file"]) == entry["sha256"], entry["file"]
    plan = {
        "date": "2026-10-03",
        "role": "read-out diagnostics D1-D6 for the QAT08 chat student; development data only; no training",
        "spec": "reviews/REVIEW_QAT08_CHAT_opus55.md sections 4-6",
        "cases_record_sha256": sha(OUT / "cases_record.json"), "files": record["files"], "kl_ids": record["kl_ids"],
        "arms": {a: {"file": f, "sha256": sha(ROOT / f)} for a, f in ARMS.items()},
        "trajectory": TRAJECTORY, "generative_arms": GENERATIVE_ARMS,
        "kl_arms": {a: {**s, "sha256": sha(ROOT / s["latents"])} for a, s in KL_ARMS.items()},
        "sets": list(SETS),
        "scorer": "score_17b MODEL CASES OUT 99 248077 model (frozen QAT08 arguments)",
        "generation": "llama-server /completion, prompt token IDs, temperature 0, top_k 1, n_predict 1024, 8 slots, port 8093",
        "metrics": {
            "letter": "renormalize the 4 letter log-probs; calibrated = minus the arm's mean per letter over the set; "
                      "pairwise = mean over items of P(gold calibrated score > wrong), ties 0.5; normal 95% item interval; "
                      "raw/calibrated accuracy use the strict QAT08 rule (gold beats every wrong letter)",
            "cloze": "option score = -mean token NLL of the option given the question (chat: Qwen template, thinking off; "
                     "raw: 'Question: ...\\nAnswer:' + ' option'); pairwise and accuracy as above, no calibration",
            "binding": "letter metrics on 400 in-context registry items, gold letters exactly balanced",
            "generative": "parse the last 'Answer: X' (diag_metrics.parse_mc_answer); unparsed = wrong; Wilson interval",
            "kl": "per position KL(teacher||student) over the full vocabulary; mean over question, template and answer "
                  "positions (diag_metrics.regions with prefix 3, suffix 9) and the 4-letter renormalized KL at the answer position",
            "trajectory": "final minus t8m paired item interval (score_17b.paired_items) on pairwise item values x100",
        },
        "rule": RULE,
        "rule_text": "harness_invalid if FP binding calibrated < 95% or FP letter pairwise < 60%; else with chat-final binding b: "
                     "b < 70 -> readout_broken (M, U; optional MU); b >= 90 but FP's best cloze pairwise lower bound <= 55 (cloze uninformative) -> ambiguous (M, U; optional MU); "
                     "b >= 90 and no student cloze pairwise lower bound > 52 -> knowledge_lost (U, B); "
                     "b >= 90 with such a signal -> closed_book_readout (M, U; optional MU); otherwise ambiguous (M, U; optional MU); "
                     "add B if letter or raw-cloze final minus t8m has mean >= 2 points and lower bound > 0",
        "budget_gpu_hours": 2.0,
        "implementation_sha256": {n: sha(ROOT / n) for n in IMPLEMENTATION},
        "runtime_sha256": runtime_hashes(),
        "environment": {"python": platform.python_version()},
        "approval": "GPU stages require results/diag_readout/approval.json bound to this file's sha256, written only after the user approves",
    }
    write_new(OUT / "plan.json", plan)
    print("froze diagnostics plan", sha(OUT / "plan.json"))


def verify() -> dict:
    plan = json.loads((OUT / "plan.json").read_text())
    assert sha(OUT / "cases_record.json") == plan["cases_record_sha256"]
    for entry in plan["files"].values():
        assert sha(OUT / entry["file"]) == entry["sha256"], entry["file"]
    for arm, entry in plan["arms"].items():
        assert sha(ROOT / entry["file"]) == entry["sha256"], arm
    for arm, entry in plan["kl_arms"].items():
        assert sha(ROOT / entry["latents"]) == entry["sha256"], arm
    for name, digest in plan["implementation_sha256"].items():
        assert sha(ROOT / name) == digest, f"diagnostic dependency changed: {name}"
    assert runtime_hashes() == plan["runtime_sha256"], "pinned runtime libraries changed"
    return plan


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cases", "freeze", "verify"])
    a = ap.parse_args()
    {"cases": cases, "freeze": freeze, "verify": lambda: print("plan verified", sha(OUT / "plan.json")) or verify()}[a.cmd]()

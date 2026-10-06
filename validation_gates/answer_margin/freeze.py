"""Freeze exact chat-prefix and candidate IDs for a tiny answer-margin gate."""

import ast
import hashlib
import json
import re
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "analysis/bonsai2"))
from chat_tokens import chat_template, render_chat  # noqa: E402

MODEL = ROOT / "models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf"
TOKENIZER = ROOT / "bin/cuda/llama-tokenize"
PROMPTS = ROOT / "analysis/bonsai2/prompts/retrieval_v2.jsonl"
EOS_TEXT = "<|im_end|>"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ids_sha(ids):
    import struct
    return hashlib.sha256(struct.pack("<" + "i" * len(ids), *ids)).hexdigest()


def tokenize(text):
    proc = subprocess.run([str(TOKENIZER), "-m", str(MODEL), "--stdin", "--ids", "--no-bos"],
                          input=text.encode(), capture_output=True, check=True)
    return ast.literal_eval(proc.stdout.decode())


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    old_manifest = json.loads((ROOT / "analysis/bonsai2/prompts/retrieval_v2_manifest.json").read_text())
    if sha(PROMPTS) != old_manifest["rows_sha256"]:
        raise ValueError("Corrected prompt file changed")
    baseline = json.loads((ROOT / "analysis/bonsai2/scorer/frozen_manifest.json").read_text())
    if sha(MODEL) != baseline["model_sha256"]:
        raise ValueError("Baseline model changed")
    registry_rows = [r for r in map(json.loads, PROMPTS.read_text().splitlines())
                     if r["id"].startswith("rv2-00-")]
    selected = [r for r in registry_rows if r["question_id"] == "q1"]
    if len(selected) != 6:
        raise ValueError("Expected six q1 base/swap layouts")
    order = [f"rv2-00-q1-{layout}-{swap}"
             for swap in ("base", "swap")
             for layout in ("short-near", "long-near", "long-far")]
    rows = {r["id"]: r for r in selected}
    template = chat_template()
    frozen_cases = []
    tsv = []
    token_root = HERE / "ids"
    token_root.mkdir(exist_ok=True)
    original_candidates = None
    for item in order:
        row = rows[item]
        q2 = next(q for q in registry_rows if q["question_id"] == "q2" and
                  q["layout"] == row["layout"] and q["swap"] == row["swap"])
        other = row["records"][q2["target_entry"] - 1]
        gold, wrong = row["answer"], other["code"]
        candidate_set = {gold, wrong}
        if original_candidates is None:
            original_candidates = candidate_set
        if candidate_set != original_candidates:
            raise ValueError("Base/swap candidates changed")
        line = (f"Entry {row['target_entry']:02d}: Item "
                f"{row['records'][row['target_entry'] - 1]['name']} was assigned reference code {gold}. "
                "The clerk recorded its arrival and completed the same ordinary intake check.")
        if row["prompt"].count(line) != 1:
            raise ValueError(f"Target entry not unique: {item}")
        missing_prompt = row["prompt"].replace(line, "")
        if gold in missing_prompt or wrong not in missing_prompt:
            raise ValueError(f"Target-removal negative control invalid: {item}")
        for control, prompt in (("present", row["prompt"]), ("target-removed", missing_prompt)):
            chat = render_chat(template, prompt)
            prefix = tokenize(chat)
            if control == "present" and (len(prefix) != row["prompt_tokens"] or
                                         hashlib.sha256(chat.encode()).hexdigest() != row["chat_sha256"]):
                raise ValueError(f"Frozen server prefix mismatch: {item}")
            for role, code in (("gold", gold), ("wrong", wrong)):
                full = tokenize(chat + code + EOS_TEXT)
                if full[:len(prefix)] != prefix:
                    raise ValueError(f"Boundary tokenization changed prefix: {item} {control} {role}")
                target = full[len(prefix):]
                if len(target) < 2:
                    raise ValueError("Candidate lacks code and end-of-turn IDs")
                case_id = f"{item}-{control}-{role}"
                ids_path = token_root / f"{case_id}.ids"
                payload = "".join(f"{x}\n" for x in full)
                if ids_path.exists() and ids_path.read_text() != payload:
                    raise ValueError(f"Frozen IDs changed: {case_id}")
                ids_path.write_text(payload)
                frozen_cases.append({"case_id": case_id, "prompt_id": item,
                    "layout": row["layout"], "swap": row["swap"], "control": control,
                    "candidate_role": role, "candidate_code": code,
                    "gold_code": gold, "wrong_code": wrong,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "chat_sha256": hashlib.sha256(chat.encode()).hexdigest(),
                    "prefix_token_count": len(prefix), "prefix_ids_sha256": ids_sha(prefix),
                    "target_token_count": len(target), "target_ids_sha256": ids_sha(target),
                    "target_ids": target, "full_ids_file": str(ids_path.relative_to(ROOT)),
                    "full_ids_file_sha256": hashlib.sha256(payload.encode()).hexdigest()})
                tsv.append(f"{case_id}\t{ids_path.relative_to(ROOT)}\t{len(prefix)}\t{len(prefix)}\t{len(target)}\n")
    if len(frozen_cases) != 24:
        raise ValueError("Wrong case count")
    dose_path = ROOT / "analysis/bonsai2/dose_calibration/DOSE_FREEZE.json"
    dose = json.loads(dose_path.read_text())
    variants = [v for v in dose["selected_variants"] if v["seed"] == 20260929]
    if len(variants) != 2 or {v["arm"] for v in variants} != {"alpha", "mlp"}:
        raise ValueError("Expected one alpha and one MLP variant")
    manifest = {"purpose": "small teacher-forced gold-versus-wrong answer margin validation",
        "prompt_file_sha256": sha(PROMPTS), "model_path": str(MODEL.relative_to(ROOT)),
        "model_sha256": baseline["model_sha256"],
        "tokenizer_path": str(TOKENIZER.relative_to(ROOT)), "tokenizer_sha256": sha(TOKENIZER),
        "scorer_source_sha256": baseline["scorer_source_sha256"],
        "scorer_binary_sha256": sha(ROOT / "analysis/bonsai2/scorer/position_scorer"),
        "dose_freeze_sha256": sha(dose_path), "selected_variants": variants,
        "tokenization": "GGUF chat template; parse_special=true; add_bos=false; candidate code followed by <|im_end|>; full-string tokenization must preserve prompt prefix IDs",
        "cases": frozen_cases,
        "gates": {"baseline": "All six present-condition gold-minus-wrong log-likelihood margins positive and swap follows changed code; target-removal margin drops by at least 1 nat in at least four of six matched prompts",
                  "variants": "Only score one alpha and one MLP seed after baseline gate passes; compare exact same candidate ID strings and preserve all per-token NLLs"}}
    out = HERE / "ANSWER_MARGIN_FREEZE.json"
    encoded = json.dumps(manifest, indent=2) + "\n"
    if out.exists() and out.read_text() != encoded:
        raise ValueError("Frozen answer-margin manifest changed")
    out.write_text(encoded)
    cases_path = HERE / "cases.tsv"
    payload = "".join(tsv)
    if cases_path.exists() and cases_path.read_text() != payload:
        raise ValueError("Frozen case TSV changed")
    cases_path.write_text(payload)
    print(f"Frozen {len(frozen_cases)} candidate cases from {len(order)} prompts")


if __name__ == "__main__":
    main()

"""Freeze four new registry families for a small margin generalization check."""

import ast
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "analysis/bonsai2"))
from chat_tokens import chat_template, render_chat  # noqa: E402

PROMPTS = ROOT / "analysis/bonsai2/prompts/retrieval_v2.jsonl"
TOKENIZER = ROOT / "bin/cuda/llama-tokenize"
SCORER = ROOT / "analysis/bonsai2/scorer/position_scorer"
REGISTRIES = (4, 5, 6, 7)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids_sha(ids):
    return hashlib.sha256(struct.pack("<" + "i" * len(ids), *ids)).hexdigest()


def tokenize(model, text):
    proc = subprocess.run([str(TOKENIZER), "-m", str(model), "--stdin", "--ids", "--no-bos"],
                          input=text.encode(), capture_output=True, check=True)
    return ast.literal_eval(proc.stdout.decode())


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    ids_dir = HERE / "ids"
    ids_dir.mkdir(exist_ok=True)
    old = json.loads((ROOT / "analysis/bonsai2/prompts/retrieval_v2_manifest.json").read_text())
    if sha(PROMPTS) != old["rows_sha256"]:
        raise ValueError("Corrected prompts changed")
    scorer_manifest = json.loads((ROOT / "analysis/bonsai2/scorer/frozen_manifest.json").read_text())
    model = ROOT / scorer_manifest["model_path"]
    if sha(model) != scorer_manifest["model_sha256"]:
        raise ValueError("Original checkpoint changed")
    dose_path = ROOT / "analysis/bonsai2/dose_calibration/DOSE_FREEZE.json"
    dose = json.loads(dose_path.read_text())
    variants = [v for v in dose["selected_variants"] if v["seed"] == 20260929]
    if len(variants) != 2 or {v["arm"] for v in variants} != {"alpha", "mlp"}:
        raise ValueError("Unexpected pilot variants")
    rows = {r["id"]: r for r in map(json.loads, PROMPTS.read_text().splitlines())}
    template = chat_template()
    cases = []
    baseline_tsv = []
    variant_tsv = []
    for registry in REGISTRIES:
        for swap in ("base", "swap"):
            for layout in ("short-near", "long-near", "long-far"):
                item = f"rv2-{registry:02d}-q1-{layout}-{swap}"
                q2_item = f"rv2-{registry:02d}-q2-{layout}-{swap}"
                row, q2 = rows[item], rows[q2_item]
                gold = row["answer"]
                wrong = row["records"][q2["target_entry"] - 1]["code"]
                if gold == wrong:
                    raise ValueError(f"Candidate codes identical: {item}")
                target = row["records"][row["target_entry"] - 1]
                line = (f"Entry {row['target_entry']:02d}: Item {target['name']} was assigned "
                        f"reference code {gold}. The clerk recorded its arrival and completed "
                        "the same ordinary intake check.")
                if row["prompt"].count(line) != 1:
                    raise ValueError(f"Target entry not unique: {item}")
                removed = row["prompt"].replace(line, "")
                if gold in removed or wrong not in removed:
                    raise ValueError(f"Target-removal prompt invalid: {item}")
                for control, prompt in (("present", row["prompt"]), ("target-removed", removed)):
                    chat = render_chat(template, prompt)
                    prefix = tokenize(model, chat)
                    if control == "present" and (len(prefix) != row["prompt_tokens"] or
                        hashlib.sha256(chat.encode()).hexdigest() != row["chat_sha256"]):
                        raise ValueError(f"Frozen chat prefix mismatch: {item}")
                    for role, code in (("gold", gold), ("wrong", wrong)):
                        full = tokenize(model, chat + code + "<|im_end|>")
                        if full[:len(prefix)] != prefix:
                            raise ValueError(f"Candidate changed prefix IDs: {item} {control} {role}")
                        target_ids = full[len(prefix):]
                        if len(target_ids) != 9:
                            raise ValueError(f"Candidate length differs: {item} {control} {role}")
                        case_id = f"{item}-{control}-{role}"
                        ids_path = ids_dir / f"{case_id}.ids"
                        payload = "".join(f"{x}\n" for x in full)
                        if ids_path.exists() and ids_path.read_text() != payload:
                            raise ValueError(f"Frozen IDs changed: {case_id}")
                        ids_path.write_text(payload)
                        case = {"case_id": case_id, "prompt_id": item,
                            "registry_id": row["registry_id"], "layout": layout, "swap": swap,
                            "control": control, "candidate_role": role, "candidate_code": code,
                            "gold_code": gold, "wrong_code": wrong,
                            "chat_sha256": hashlib.sha256(chat.encode()).hexdigest(),
                            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                            "prefix_token_count": len(prefix), "prefix_ids_sha256": ids_sha(prefix),
                            "target_token_count": len(target_ids), "target_ids_sha256": ids_sha(target_ids),
                            "target_ids": target_ids, "full_ids_file": str(ids_path.relative_to(ROOT)),
                            "full_ids_file_sha256": hashlib.sha256(payload.encode()).hexdigest()}
                        cases.append(case)
                        line_tsv = f"{case_id}\t{ids_path.relative_to(ROOT)}\t{len(prefix)}\t{len(prefix)}\t9\n"
                        baseline_tsv.append(line_tsv)
                        if control == "present":
                            variant_tsv.append(line_tsv)
    if len(cases) != 96 or len(variant_tsv) != 48:
        raise ValueError("Wrong balanced case count")
    by_id = {c["case_id"]: c for c in cases}
    for registry in REGISTRIES:
        for layout in ("short-near", "long-near", "long-far"):
            stem = f"rv2-{registry:02d}-q1-{layout}"
            base = by_id[f"{stem}-base-present-gold"]
            swap = by_id[f"{stem}-swap-present-gold"]
            if base["candidate_code"] != swap["wrong_code"] or swap["candidate_code"] != base["wrong_code"]:
                raise ValueError(f"Base/swap candidates failed: {stem}")
    frozen = {"purpose": "answer-margin generalization check on four registry families outside the initial accuracy pilot",
        "registries": list(REGISTRIES), "prompt_file_sha256": sha(PROMPTS),
        "model_path": scorer_manifest["model_path"], "model_sha256": scorer_manifest["model_sha256"],
        "tokenizer_sha256": sha(TOKENIZER), "scorer_source_sha256": scorer_manifest["scorer_source_sha256"],
        "scorer_binary_sha256": sha(SCORER), "dose_freeze_sha256": sha(dose_path),
        "variants": variants, "cases": cases,
        "profile": {"n_ctx": 16384, "n_batch": 512, "n_ubatch": 512,
                    "n_seq_max": 1, "gpu_layers": 99, "flash_attention": "on"},
        "gates": {"baseline": "At least 20 of 24 target-present gold margins > 0 and at least 20 of 24 matched target-removal margins drop by >=1 nat; base/swap reverses candidate identities",
                  "variant": "Only run one alpha and one MLP seed on 48 present-target candidates each after the baseline gate; report registry-clustered and per-layout contrasts descriptively"},
        "token_policy": "exact GGUF chat prefix, no BOS, parse special, code plus <|im_end|>, 9 target IDs for every candidate"}
    manifest_path = HERE / "MARGIN_EXTENSION_FREEZE.json"
    payload = json.dumps(frozen, indent=2) + "\n"
    if manifest_path.exists() and manifest_path.read_text() != payload:
        raise ValueError("Frozen extension manifest changed")
    manifest_path.write_text(payload)
    for name, lines in (("baseline_cases.tsv", baseline_tsv), ("variant_cases.tsv", variant_tsv)):
        path = HERE / name
        payload = "".join(lines)
        if path.exists() and path.read_text() != payload:
            raise ValueError(f"Frozen TSV changed: {name}")
        path.write_text(payload)
    print("Frozen 24 prompts, 96 baseline candidates, 48 variant candidates")


if __name__ == "__main__":
    main()

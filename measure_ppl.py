"""GPU perplexity calibration on fixed 512-token chunks, without logit dumps."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
PATTERN = re.compile(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)")


def parse_log(path: Path):
    match = PATTERN.search(path.read_text())
    if not match:
        raise ValueError(f"No final perplexity in {path}")
    return float(match.group(1)), float(match.group(2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deltas", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--arms", choices=["state", "mlp"], nargs="+", default=["state", "mlp"])
    args = parser.parse_args()
    corpus = BASE / "prompts/calibration_corpus.txt"
    baseline = BASE / "runs/ppl-baseline.log"
    baseline_ppl, baseline_se = parse_log(baseline)
    env = dict(os.environ)
    binary = ROOT / "bin/cuda/llama-perplexity"
    env["LD_LIBRARY_PATH"] = str(binary.parent) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    out = BASE / "runs/ppl_summary.json"
    results = json.loads(out.read_text()) if out.exists() else {
        "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "context_tokens": 512, "chunks": 8,
        "baseline": {"ppl": baseline_ppl, "uncertainty_reported_by_binary": baseline_se},
        "candidates": {}}
    if results["corpus_sha256"] != hashlib.sha256(corpus.read_bytes()).hexdigest():
        raise ValueError("Calibration corpus changed since previous measurements")
    for delta in args.deltas:
        for arm in args.arms:
            name = f"{arm}-d{delta:02d}"
            if name in results["candidates"]:
                print(f"Skipping measured {name}", flush=True)
                continue
            model = BASE / "variants" / f"bonsai2-{name}.gguf"
            log = BASE / "runs" / f"ppl-{name}.log"
            command = [str(binary), "-m", str(model), "-ngl", "99", "-fa", "on",
                       "-c", "512", "-b", "512", "-ub", "512", "-f", str(corpus),
                       "--chunks", "8"]
            with log.open("w") as stream:
                subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                               check=True)
            ppl, se = parse_log(log)
            results["candidates"][name] = {"ppl": ppl, "uncertainty_reported_by_binary": se,
                                            "excess_nll_nats": math.log(ppl / baseline_ppl),
                                            "log": str(log)}
            print(name, ppl, "excess_nll", round(math.log(ppl / baseline_ppl), 6), flush=True)
            out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

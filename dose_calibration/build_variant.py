"""Build and byte-verify separate alpha, beta, or coverage-controlled MLP reflink variants."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import random
import struct
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from make_variants import bf16_scaled, layout, sha256

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf"
OUT = ROOT / "analysis/bonsai2/variants/dose_v3"
SOURCE_SHA = "3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1"


def choose(rows, arm: str, coverage: float, seed: int):
    by_name = {row["tensor"]: row for row in rows}
    for block in range(64):
        if block % 4 == 3:
            continue
        name = f"blk.{block}.{'ffn_up' if arm == 'mlp' else 'ssm_' + arm}.weight"
        row = by_name[name]
        total = row["elements"] // 128
        count = total if arm != "mlp" else 2 * max(1, round(total * coverage / 2))
        if count > total or count % 2:
            raise ValueError("Bad group count")
        indices = list(range(total)) if arm != "mlp" else [((2 * i + 1) * total) // (2 * count) for i in range(count)]
        if len(set(indices)) != count:
            raise ValueError("Duplicate MLP group index")
        signs = [1] * (count // 2) + [-1] * (count // 2)
        random.Random(f"dose-v3:{seed}:{arm}:{name}").shuffle(signs)
        yield row, indices, signs


def variant_name(arm: str, delta: float, coverage: float, seed: int):
    return f"{arm}-d{round(1000 * delta):03d}-c{round(1000 * coverage):03d}-s{seed}"


def specs(rows, data_start, arm, coverage, seed):
    selected = []
    details = []
    for row, indices, signs in choose(rows, arm, coverage, seed):
        stride, width = (34, 2) if arm == "mlp" else (256, 256)
        selected.extend((data_start + row["offset"] + stride * i, width, sign)
                        for i, sign in zip(indices, signs))
        details.append({"tensor": row["tensor"], "storage": row["storage"],
                        "available_groups": row["elements"] // 128, "selected_groups": len(indices),
                        "first_group": indices[0], "last_group": indices[-1],
                        "plus_groups": signs.count(1), "minus_groups": signs.count(-1)})
    selected.sort(key=lambda x: x[0])
    return selected, details


def replace(raw: bytes, width: int, sign: int, delta: float) -> bytes:
    if width == 256:
        return bf16_scaled(raw, 1 + sign * delta)
    scale = struct.unpack("<e", raw)[0]
    if not math.isfinite(scale):
        raise ValueError("Nonfinite MLP scale")
    return struct.pack("<e", scale * (1 + sign * delta))


def verify(model: Path, candidate: Path, selected, delta: float):
    if model.stat().st_size != candidate.stat().st_size:
        raise ValueError("File size changed")
    # Verify each realized patch, then compare all other bytes in 8 MiB blocks.
    changed_values = 0
    changed_bytes = 0
    with model.open("rb") as src, candidate.open("rb") as dst:
        original = mmap.mmap(src.fileno(), 0, access=mmap.ACCESS_READ)
        patched = mmap.mmap(dst.fileno(), 0, access=mmap.ACCESS_READ)
        for begin, width, sign in selected:
            raw = original[begin:begin + width]
            actual = patched[begin:begin + width]
            if actual != replace(raw, width, sign, delta):
                raise ValueError(f"Wrong patch at offset {begin}")
            changed_bytes += sum(x != y for x, y in zip(raw, actual))
            changed_values += (sum(raw[i:i + 2] != actual[i:i + 2] for i in range(0, width, 2)))
        block_size = 8 * 1024 * 1024
        index = 0
        for start in range(0, len(original), block_size):
            end = min(start + block_size, len(original))
            expected = bytearray(original[start:end])
            actual = bytearray(patched[start:end])
            while index < len(selected) and selected[index][0] < end:
                begin, width, _ = selected[index]
                lo, hi = max(begin, start), min(begin + width, end)
                if lo < hi:
                    actual[lo - start:hi - start] = expected[lo - start:hi - start]
                if begin + width <= end:
                    index += 1
                else:
                    break
            if actual != expected:
                raise ValueError(f"Unintended byte change in block at {start}")
        original.close()
        patched.close()
    return changed_values, changed_bytes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=("alpha", "beta", "mlp"), required=True)
    ap.add_argument("--delta", type=float, required=True)
    ap.add_argument("--coverage", type=float, default=1.0)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    if not 0 < args.delta < 1 or not 0 < args.coverage <= 1 or (args.arm != "mlp" and args.coverage != 1):
        ap.error("delta and coverage must be valid; state coverage is all groups")
    if sha256(MODEL) != SOURCE_SHA:
        raise ValueError("Original model hash changed")
    version, metadata, rows, data_start = layout(MODEL)
    selected, details = specs(rows, data_start, args.arm, args.coverage, args.seed)
    name = variant_name(args.arm, args.delta, args.coverage, args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    candidate = OUT / f"{name}.gguf"
    manifest = OUT / f"{name}.json"
    if args.verify_only:
        if not candidate.exists() or not manifest.exists():
            raise FileNotFoundError(candidate)
    else:
        if candidate.exists() or manifest.exists():
            raise FileExistsError(candidate)
        subprocess.run(["cp", "--reflink=always", "--", str(MODEL), str(candidate)], check=True)
        try:
            with MODEL.open("rb") as src, candidate.open("r+b") as dst:
                original = mmap.mmap(src.fileno(), 0, access=mmap.ACCESS_READ)
                patched = mmap.mmap(dst.fileno(), 0, access=mmap.ACCESS_WRITE)
                for begin, width, sign in selected:
                    patched[begin:begin + width] = replace(original[begin:begin + width], width, sign, args.delta)
                patched.flush()
                patched.close()
                original.close()
            changed_values, changed_bytes = verify(MODEL, candidate, selected, args.delta)
            record = {"status": "uncalibrated_candidate", "arm": args.arm, "delta": args.delta,
                      "coverage_requested": args.coverage, "seed": args.seed,
                      "source_path": str(MODEL.relative_to(ROOT)), "source_sha256": SOURCE_SHA,
                      "candidate_path": str(candidate.relative_to(ROOT)),
                      "candidate_sha256": sha256(candidate), "candidate_bytes": candidate.stat().st_size,
                      "group_size_weights": 128, "selected_groups": len(selected),
                      "selected_values": len(selected) * (1 if args.arm == "mlp" else 128),
                      "changed_values_after_rounding": changed_values,
                      "changed_bytes": changed_bytes, "selected_tensors": details,
                      "selection": "all groups for separate gate; evenly spaced groups for MLP; balanced seeded signs per tensor",
                      "source_format": f"GGUF v{version} {metadata['general.architecture']}",
                      "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
            manifest.write_text(json.dumps(record, indent=2) + "\n")
        except BaseException:
            candidate.unlink(missing_ok=True)
            manifest.unlink(missing_ok=True)
            raise
    if args.verify_only:
        record = json.loads(manifest.read_text())
        if record["candidate_sha256"] != sha256(candidate):
            raise ValueError("Candidate hash changed")
        changed_values, changed_bytes = verify(MODEL, candidate, selected, args.delta)
        if (changed_values, changed_bytes) != (record["changed_values_after_rounding"], record["changed_bytes"]):
            raise ValueError("Realized patch stats changed")
    print(json.dumps({"name": name, "candidate": str(candidate), "selected_groups": len(selected),
                      "changed_values_after_rounding": changed_values, "changed_bytes": changed_bytes}))


if __name__ == "__main__":
    main()

"""Create and verify controlled Bonsai 2 PQ2_0 perturbation candidates.

Each candidate is a Btrfs reflink of the original. Only BF16 values in the
state arm or FP16 PQ2_0 scales in the MLP arm are written. The original is
opened read-only. Run this before calibration; candidate deltas are not final.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import struct
import subprocess
from pathlib import Path

from inspect_gguf import inspect


DEFAULT_MODEL = Path("models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf")
DEFAULT_OUT = Path(__file__).resolve().parent / "variants"
SEED = 20260929


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bf16_scaled(raw: bytes, multiplier: float) -> bytes:
    result = bytearray(len(raw))
    for pos in range(0, len(raw), 2):
        word = int.from_bytes(raw[pos:pos + 2], "little")
        value = struct.unpack("<f", struct.pack("<I", word << 16))[0]
        if not math.isfinite(value):
            raise ValueError("Non-finite BF16 weight")
        bits = struct.unpack("<I", struct.pack("<f", value * multiplier))[0]
        # Round to nearest even when reducing F32 back to BF16.
        rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
        result[pos:pos + 2] = rounded.to_bytes(2, "little")
    return bytes(result)


def selections(rows, arm: str, seed: int):
    """Yield (tensor row, group indices, signs), balanced within each block."""
    by_name = {row["tensor"]: row for row in rows}
    for block in range(64):
        if block % 4 == 3:
            continue
        state = [by_name[f"blk.{block}.ssm_{kind}.weight"] for kind in ("alpha", "beta")]
        target_groups = sum(row["elements"] for row in state) // 128
        selected = state if arm == "state" else [by_name[f"blk.{block}.ffn_up.weight"]]
        for row in selected:
            if arm == "state":
                if row["storage"] != "BF16":
                    raise ValueError(f"Unexpected storage: {row['tensor']}")
                indices = list(range(row["elements"] // 128))
            else:
                if row["storage"] != "PQ2_0":
                    raise ValueError(f"Unexpected storage: {row['tensor']}")
                available = row["elements"] // 128
                indices = [(i * available + available // 2) // target_groups
                           for i in range(target_groups)]
                if len(set(indices)) != len(indices):
                    raise ValueError("MLP sampling produced duplicate groups")
            signs = [1] * (len(indices) // 2) + [-1] * (len(indices) // 2)
            if len(signs) != len(indices):
                raise ValueError("Expected an even number of groups")
            random.Random(f"{seed}:{arm}:{row['tensor']}").shuffle(signs)
            yield row, indices, signs


def layout(path: Path):
    version, metadata, rows, header_bytes = inspect(path)
    if version != 3 or metadata.get("general.architecture") != "qwen35" or metadata.get("qwen35.block_count") != 64:
        raise ValueError("Expected Bonsai 2 64-block qwen35 GGUF v3")
    alignment = metadata.get("general.alignment", 32)
    data_start = (header_bytes + alignment - 1) // alignment * alignment
    if max(data_start + r["offset"] + r["bytes"] for r in rows) > path.stat().st_size:
        raise ValueError("Tensor range exceeds file")
    return version, metadata, rows, data_start


def create(model: Path, out: Path, arm: str, delta: float, seed: int, source_sha: str):
    _, metadata, rows, data_start = layout(model)
    destination = out / f"bonsai2-{arm}-d{int(round(delta * 100)):02d}.gguf"
    manifest_path = destination.with_suffix(".json")
    if destination.exists() or manifest_path.exists():
        raise FileExistsError(f"Candidate exists: {destination}; remove it explicitly to rebuild")
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "--reflink=always", "--", str(model), str(destination)], check=True)
    patch_rows = []
    changed_bytes = 0
    try:
        with model.open("rb") as original, destination.open("r+b", buffering=0) as patched:
            for row, indices, signs in selections(rows, arm, seed):
                stride = 256 if arm == "state" else 34
                item = {"tensor": row["tensor"], "storage": row["storage"],
                        "tensor_offset": row["offset"], "group_indices": indices,
                        "signs": "seeded_shuffle_equal_plus_minus_per_tensor",
                        "plus_groups": signs.count(1), "minus_groups": signs.count(-1)}
                for index, sign in zip(indices, signs):
                    offset = data_start + row["offset"] + stride * index
                    original.seek(offset)
                    raw = original.read(stride if arm == "state" else 2)
                    if len(raw) != (stride if arm == "state" else 2):
                        raise ValueError(f"Truncated tensor: {row['tensor']}")
                    multiplier = 1 + sign * delta
                    if arm == "state":
                        replacement = bf16_scaled(raw, multiplier)
                    else:
                        scale = struct.unpack("<e", raw)[0]
                        if not math.isfinite(scale):
                            raise ValueError("Non-finite PQ2_0 scale")
                        replacement = struct.pack("<e", scale * multiplier)
                    changed_bytes += sum(a != b for a, b in zip(raw, replacement))
                    patched.seek(offset)
                    patched.write(replacement)
                patch_rows.append(item)
        manifest = {
            "status": "uncalibrated_candidate", "source": str(model.resolve()),
            "source_sha256": source_sha, "source_bytes": model.stat().st_size,
            "candidate": str(destination.resolve()), "seed": seed, "arm": arm,
            "delta": delta, "group_size_weights": 128, "data_start": data_start,
            "tensor_count": len(rows), "selected_tensors": patch_rows,
            "selected_groups": sum(len(row["group_indices"]) for row in patch_rows),
            "affected_weights": 128 * sum(len(row["group_indices"]) for row in patch_rows),
            "changed_payload_bytes": changed_bytes,
            "candidate_bytes": destination.stat().st_size,
            "metadata": metadata,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        return destination, manifest_path
    except BaseException:
        destination.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise


def verify(model: Path, candidate: Path, manifest_path: Path):
    manifest = json.loads(manifest_path.read_text())
    if manifest["source_sha256"] != sha256(model):
        raise ValueError("Original SHA-256 differs from manifest")
    source_layout = layout(model)
    candidate_layout = layout(candidate)
    if source_layout != candidate_layout or model.stat().st_size != candidate.stat().st_size:
        raise ValueError("GGUF metadata, tensor dimensions, offsets, or file size changed")
    data_start = source_layout[3]
    allowed = []
    expected = []
    for row in manifest["selected_tensors"]:
        stride = 256 if row["storage"] == "BF16" else 34
        width = 256 if row["storage"] == "BF16" else 2
        signs = [1] * row["plus_groups"] + [-1] * row["minus_groups"]
        random.Random(f"{manifest['seed']}:{manifest['arm']}:{row['tensor']}").shuffle(signs)
        for index, sign in zip(row["group_indices"], signs):
            begin = data_start + row["tensor_offset"] + stride * index
            allowed.append((begin, begin + width))
            expected.append((begin, width, sign))
    allowed.sort()
    if any(a[1] > b[0] for a, b in zip(allowed, allowed[1:])):
        raise ValueError("Overlapping intended patch ranges")
    actual_changed = 0
    with model.open("rb") as original, candidate.open("rb") as patched:
        for begin, width, sign in expected:
            original.seek(begin)
            patched.seek(begin)
            raw, actual = original.read(width), patched.read(width)
            if width == 256:
                wanted = bf16_scaled(raw, 1 + sign * manifest["delta"])
            else:
                scale = struct.unpack("<e", raw)[0]
                wanted = struct.pack("<e", scale * (1 + sign * manifest["delta"]))
            if actual != wanted:
                raise ValueError(f"Wrong perturbation at byte {begin}")
            actual_changed += sum(a != b for a, b in zip(raw, actual))
    if actual_changed != manifest["changed_payload_bytes"]:
        raise ValueError("Changed byte count differs from manifest")
    # Compare every unselected byte, including header, padding, and ternary codes.
    with model.open("rb") as original, candidate.open("rb") as patched:
        cursor = 0
        for begin, end in allowed + [(model.stat().st_size, model.stat().st_size)]:
            while cursor < begin:
                count = min(8 * 1024 * 1024, begin - cursor)
                if original.read(count) != patched.read(count):
                    raise ValueError(f"Unexpected change before byte {begin}")
                cursor += count
            original.seek(end)
            patched.seek(end)
            cursor = end
    return {"candidate": str(candidate), "verified": True,
            "selected_groups": len(allowed), "file_bytes": candidate.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--deltas", type=float, nargs="+", default=[0.01, 0.03, 0.05])
    parser.add_argument("--arms", choices=["state", "mlp"], nargs="+", default=["state", "mlp"])
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if any(not 0 < delta < 1 for delta in args.deltas):
        parser.error("deltas must be between zero and one")
    model = args.model.resolve()
    if args.out.resolve() == model.parent:
        parser.error("variant output must not be the source directory")
    source_sha = sha256(model)
    print(f"Source SHA-256: {source_sha}", flush=True)
    for delta in args.deltas:
        for arm in args.arms:
            path = args.out / f"bonsai2-{arm}-d{int(round(delta * 100)):02d}.gguf"
            manifest = path.with_suffix(".json")
            if not args.verify_only:
                path, manifest = create(model, args.out, arm, delta, args.seed, source_sha)
            result = verify(model, path, manifest)
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

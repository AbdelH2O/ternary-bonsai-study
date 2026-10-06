"""Independent numeric fixtures for the pinned Hadamard and GDN layout contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from phase0 import OUT, reorder_v

BLOCK = 1024


def fwht_last(x: np.ndarray) -> np.ndarray:
    """Natural-order Sylvester WHT along the final axis."""
    y = np.array(x, dtype=np.float64, copy=True)
    assert y.shape[-1] % BLOCK == 0
    blocks = y.reshape(*y.shape[:-1], -1, BLOCK)
    for step in (1 << k for k in range(10)):
        z = blocks.reshape(*blocks.shape[:-1], BLOCK // (2 * step), 2, step)
        a, b = z[..., 0, :].copy(), z[..., 1, :].copy()
        z[..., 0, :], z[..., 1, :] = a + b, a - b
    return y / np.sqrt(BLOCK)


def sylvester_column(index: int) -> np.ndarray:
    # Independent bit-parity construction, mirroring llama-model.cpp's H.
    return np.array([1 - 2 * ((i & index).bit_count() & 1) for i in range(BLOCK)], dtype=np.float64) / np.sqrt(BLOCK)


def fold_rows(w: np.ndarray, signs: np.ndarray) -> np.ndarray:
    return fwht_last(w * signs)


def rotate_activation(x: np.ndarray, signs: np.ndarray) -> np.ndarray:
    return fwht_last(x * signs)


def restore_embedding(z: np.ndarray, signs: np.ndarray) -> np.ndarray:
    return fwht_last(z) * signs


def main():
    rng = np.random.default_rng(0xB0A5A1)
    record = {}
    for width in (5120, 6144, 17408):
        signs = rng.choice([-1.0, 1.0], width)
        # Non-symmetric signs, including across the 1024/128 boundaries.
        signs[0] = -1
        signs[127:130] = [1, -1, 1]
        signs[1023:1026] = [-1, 1, -1]
        x = rng.standard_normal(width)
        w = rng.standard_normal((3, width))
        folded = fold_rows(w, signs)
        rotated = rotate_activation(x, signs)
        err = float(np.max(np.abs(folded @ rotated - w @ x)))
        z = fold_rows(w, signs)
        embedding_err = float(np.max(np.abs(restore_embedding(z, signs) - w)))
        wrong = fwht_last(w) * signs
        negative_control_err = float(np.max(np.abs(wrong @ rotated - w @ x)))
        assert err < 1e-10 and embedding_err < 1e-12
        assert negative_control_err > 1e-3
        for col in (0, 1, 127, 128, 1023):
            basis = np.zeros(width)
            basis[col] = 1
            got = rotate_activation(basis, signs)[col // BLOCK * BLOCK:(col // BLOCK + 1) * BLOCK]
            expected = signs[col] * sylvester_column(col % BLOCK)
            assert np.array_equal(got, expected)
        record[str(width)] = {"projection_max_abs_error": err, "embedding_max_abs_error": embedding_err,
                              "wrong_order_max_abs_error": negative_control_err}

    grouped = np.arange(16 * 3 * 128).reshape(16, 3, 128)
    tiled = reorder_v(grouped.reshape(-1), 0, 128).reshape(3, 16, 128)
    assert tiled[2, 7, 91] == grouped[7, 2, 91]
    back = tiled.transpose(1, 0, 2).reshape(-1)
    assert np.array_equal(back, grouped.reshape(-1))
    x_grouped = rng.standard_normal(6144)
    x_tiled = reorder_v(x_grouped, 0, 128)
    signs = rng.choice([-1.0, 1.0], 6144)
    w = rng.standard_normal((3, 6144))
    folded = fold_rows(w, signs)
    restored_grouped = x_tiled.reshape(3, 16, 128).transpose(1, 0, 2).reshape(-1)
    gdn_err = float(np.max(np.abs(folded @ rotate_activation(restored_grouped, signs) - w @ x_grouped)))
    wrong_gdn_err = float(np.max(np.abs(folded @ rotate_activation(x_tiled, signs) - w @ x_grouped)))
    assert gdn_err < 1e-10 and wrong_gdn_err > 1e-3
    record["gdn"] = {"grouped_projection_max_abs_error": gdn_err,
                     "unpermuted_tiled_control_max_abs_error": wrong_gdn_err}
    OUT.mkdir(exist_ok=True)
    (OUT / "contract_fixtures.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

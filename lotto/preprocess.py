from __future__ import annotations

import numpy as np


def rounds_to_multihot(
    rounds: list[list[int]], size: int = 45
) -> np.ndarray:
    """Encode each round as 45-dim multi-hot (1 if number drawn)."""
    x = np.zeros((len(rounds), size), dtype=np.float32)
    for i, nums in enumerate(rounds):
        for n in nums:
            x[i, int(n) - 1] = 1.0
    return x


def build_sequences(
    multihot: np.ndarray, seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """Input (seq_len, 45) -> next round target (45,)."""
    if len(multihot) <= seq_len:
        raise ValueError("Not enough rounds for sequence length.")
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for t in range(seq_len, len(multihot)):
        xs.append(multihot[t - seq_len : t])
        ys.append(multihot[t])
    return np.stack(xs, axis=0), np.stack(ys, axis=0)

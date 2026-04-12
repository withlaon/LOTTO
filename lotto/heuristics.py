from __future__ import annotations

import numpy as np


def cold_hot_weights(
    history_rounds: list[list[int]], window: int = 10, size: int = 45
) -> np.ndarray:
    """Recent-window frequency: dampen hot, boost cold numbers."""
    counts = np.zeros(size, dtype=np.float64)
    recent = history_rounds[-window:] if len(history_rounds) >= window else history_rounds
    for nums in recent:
        for n in nums:
            counts[int(n) - 1] += 1.0
    mean = counts.mean() + 1e-6
    w = np.exp(-0.35 * (counts - mean) / (mean + 1e-6))
    return np.clip(w, 0.25, 3.0)


def pattern_match_boost(
    history_rounds: list[list[int]],
    seq_len: int,
    size: int = 45,
) -> np.ndarray:
    """Boost numbers that followed windows similar to the latest window (cosine)."""
    if len(history_rounds) <= seq_len + 1:
        return np.ones(size, dtype=np.float64)
    mh = []
    for nums in history_rounds:
        v = np.zeros(size, dtype=np.float64)
        for n in nums:
            v[int(n) - 1] = 1.0
        mh.append(v)
    mh_arr = np.stack(mh, axis=0)
    target = mh_arr[-seq_len:].mean(axis=0)
    target_norm = np.linalg.norm(target) + 1e-9
    boost = np.zeros(size, dtype=np.float64)
    for t in range(seq_len, len(mh_arr) - 1):
        window = mh_arr[t - seq_len : t].mean(axis=0)
        sim = float(np.dot(window, target) / (np.linalg.norm(window) * target_norm))
        if sim > 0.92:
            nxt = mh_arr[t + 1]
            boost += nxt * (sim - 0.92) * 10.0
    if boost.max() > 0:
        boost = boost / (boost.max() + 1e-9)
    return 1.0 + 0.6 * boost


def moving_average_sum_bias(
    history_rounds: list[list[int]], lookback: int = 20
) -> tuple[int, int]:
    """Estimate acceptable sum range from recent draw sums (mean +/- k*std)."""
    sums = [sum(r) for r in history_rounds[-lookback:]]
    if len(sums) < 3:
        return 100, 175
    mu = float(np.mean(sums))
    sigma = float(np.std(sums)) + 1e-6
    low = max(75, int(mu - 1.2 * sigma))
    high = min(220, int(mu + 1.2 * sigma))
    return low, high


def odd_even_ok(nums: list[int]) -> bool:
    odd = sum(1 for n in nums if n % 2 == 1)
    even = 6 - odd
    return (odd, even) in {(2, 4), (3, 3), (4, 2)}


def passes_filters(
    nums: list[int],
    sum_low: int,
    sum_high: int,
    strict_odd_even: bool = True,
) -> bool:
    s = sum(nums)
    if not (sum_low <= s <= sum_high):
        return False
    if strict_odd_even and not odd_even_ok(nums):
        return False
    return True


def sample_six_from_scores(
    scores: np.ndarray,
    rng: np.random.Generator,
    sum_low: int,
    sum_high: int,
    max_tries: int = 8000,
) -> list[int] | None:
    p = np.maximum(scores.astype(np.float64), 1e-12)
    for _ in range(max_tries):
        idx = rng.choice(45, size=6, replace=False, p=p / p.sum())
        nums = sorted(int(i) + 1 for i in idx)
        if passes_filters(nums, sum_low, sum_high, strict_odd_even=True):
            return nums
    for _ in range(max_tries):
        idx = rng.choice(45, size=6, replace=False, p=p / p.sum())
        nums = sorted(int(i) + 1 for i in idx)
        if passes_filters(nums, sum_low, sum_high, strict_odd_even=False):
            return nums
    top = np.argsort(-p)[:12]
    for _ in range(max_tries):
        idx = rng.choice(top, size=6, replace=False)
        nums = sorted(int(i) + 1 for i in idx)
        if passes_filters(nums, sum_low, sum_high, strict_odd_even=False):
            return nums
    return None


def generate_sets(
    base_scores: np.ndarray,
    num_sets: int,
    seed: int,
    sum_low: int,
    sum_high: int,
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    sets: list[list[int]] = []
    used: set[tuple[int, ...]] = set()
    attempt = 0
    while len(sets) < num_sets and attempt < num_sets * 400:
        attempt += 1
        noise = rng.normal(0, 0.07, size=base_scores.shape)
        scores = np.maximum(base_scores * np.exp(noise), 1e-12)
        got = sample_six_from_scores(scores, rng, sum_low, sum_high)
        if got is None:
            continue
        key = tuple(got)
        if key in used:
            continue
        used.add(key)
        sets.append(got)
    while len(sets) < num_sets:
        nums = sorted(rng.choice(np.arange(1, 46), size=6, replace=False).tolist())
        key = tuple(nums)
        if key not in used:
            used.add(key)
            sets.append(nums)
    return sets

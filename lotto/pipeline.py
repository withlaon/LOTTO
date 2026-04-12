from __future__ import annotations

from typing import Any

import numpy as np

from lotto import heuristics
from lotto.db import Client
from lotto.db import fetch_history_ordered
from lotto.lstm_model import predict_next_probs, train_lstm
from lotto.preprocess import build_sequences, rounds_to_multihot


def history_numbers_only(
    history: list[tuple[int, list[int]]],
) -> list[list[int]]:
    return [nums for _, nums in history]


def next_target_round(history: list[tuple[int, list[int]]]) -> int:
    if not history:
        return 1
    return max(r for r, _ in history) + 1


def run_prediction(
    client: Client,
    target_round: int | None = None,
    seq_len: int = 12,
    lstm_epochs: int = 120,
    seed: int = 42,
) -> tuple[int, list[list[int]], dict[str, Any]]:
    history = fetch_history_ordered(client)
    if not history:
        raise RuntimeError("EMPTY_HISTORY")

    nums_only = history_numbers_only(history)
    if target_round is None:
        target_round = next_target_round(history)

    max_r = max(r for r, _ in history)
    if target_round != max_r + 1:
        raise RuntimeError(
            f"BAD_TARGET_ROUND:{max_r}:{target_round}"
        )

    mh = rounds_to_multihot(nums_only)
    if len(mh) < 2:
        raise RuntimeError("NEED_MORE_HISTORY")
    seq_len = min(seq_len, len(mh) - 1)
    seq_len = max(1, seq_len)

    X, y = build_sequences(mh, seq_len)
    model = train_lstm(X, y, epochs=lstm_epochs, seed=seed)
    last_seq = mh[-seq_len:].astype(np.float32)
    lstm_probs = predict_next_probs(model, last_seq)

    cold_hot = heuristics.cold_hot_weights(nums_only, window=10)
    pat = heuristics.pattern_match_boost(nums_only, seq_len=seq_len)
    sum_low, sum_high = heuristics.moving_average_sum_bias(nums_only, lookback=20)
    base = lstm_probs * cold_hot * pat
    base = base / (base.sum() + 1e-12)

    sets = heuristics.generate_sets(
        base, num_sets=5, seed=seed + target_round, sum_low=sum_low, sum_high=sum_high
    )

    meta: dict[str, Any] = {
        "seq_len": seq_len,
        "sum_range": [sum_low, sum_high],
        "lstm_probs_top5": [
            int(i) + 1 for i in np.argsort(-lstm_probs)[:5].tolist()
        ],
    }
    return target_round, sets, meta


def compute_matches(
    predicted_sets: list[list[int]], actual: list[int]
) -> list[dict[str, Any]]:
    actual_set = set(actual)
    out: list[dict[str, Any]] = []
    for i, s in enumerate(predicted_sets):
        matched = sorted(set(s) & actual_set)
        out.append(
            {
                "set_index": i,
                "predicted": s,
                "matched_numbers": matched,
                "match_count": len(matched),
            }
        )
    return out

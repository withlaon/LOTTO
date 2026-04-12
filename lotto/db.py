from __future__ import annotations

from typing import Any

from supabase import Client, create_client

from lotto.config import get_supabase_key, get_supabase_url


def get_client() -> Client:
    return create_client(get_supabase_url(), get_supabase_key())


def fetch_history_ordered(client: Client) -> list[tuple[int, list[int]]]:
    res = (
        client.table("lotto_history")
        .select("round_no, numbers")
        .order("round_no")
        .execute()
    )
    rows = res.data or []
    out: list[tuple[int, list[int]]] = []
    for r in rows:
        rn = int(r["round_no"])
        nums = [int(x) for x in r["numbers"]]
        out.append((rn, sorted(nums)))
    return out


def upsert_round(client: Client, round_no: int, numbers: list[int]) -> None:
    numbers = sorted(int(x) for x in numbers)
    if len(numbers) != 6 or len(set(numbers)) != 6:
        raise ValueError("Need exactly 6 distinct numbers.")
    if any(n < 1 or n > 45 for n in numbers):
        raise ValueError("Numbers must be in 1..45.")
    client.table("lotto_history").upsert(
        {"round_no": round_no, "numbers": numbers},
        on_conflict="round_no",
    ).execute()


def count_history(client: Client) -> int:
    res = client.table("lotto_history").select("round_no", count="exact").execute()
    return res.count or 0


def max_round_in_history(client: Client) -> int | None:
    res = (
        client.table("lotto_history")
        .select("round_no")
        .order("round_no", desc=True)
        .limit(1)
        .execute()
    )
    if not res.data:
        return None
    return int(res.data[0]["round_no"])


def insert_prediction(
    client: Client,
    target_round: int,
    predicted_sets: list[list[int]],
) -> int:
    payload = {
        "target_round": target_round,
        "predicted_sets": predicted_sets,
        "actual_numbers": None,
        "matches": None,
    }
    res = client.table("lotto_predictions").insert(payload).execute()
    if not res.data:
        raise RuntimeError("Failed to save prediction row.")
    return int(res.data[0]["id"])


def latest_prediction_for_round(
    client: Client, target_round: int
) -> dict[str, Any] | None:
    res = (
        client.table("lotto_predictions")
        .select("*")
        .eq("target_round", target_round)
        .order("id", desc=True)
        .limit(1)
        .execute()
    )
    if not res.data:
        return None
    return res.data[0]


def update_prediction_actual(
    client: Client,
    prediction_id: int,
    actual_numbers: list[int],
    matches: list[dict[str, Any]],
) -> None:
    actual_numbers = sorted(int(x) for x in actual_numbers)
    client.table("lotto_predictions").update(
        {"actual_numbers": actual_numbers, "matches": matches}
    ).eq("id", prediction_id).execute()

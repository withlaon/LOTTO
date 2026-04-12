# -*- coding: utf-8 -*-
"""Korean Lotto predictor CLI: Supabase, PyTorch LSTM, heuristic filters."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from lotto import db as dbmod
from lotto import pipeline
from lotto.db import upsert_round

console = Console()


def _parse_six(nums: list[str]) -> list[int]:
    if len(nums) != 6:
        raise ValueError("Exactly 6 numbers required.")
    return [int(x) for x in nums]


def cmd_status() -> None:
    client = dbmod.get_client()
    n = dbmod.count_history(client)
    mx = dbmod.max_round_in_history(client)
    console.print(f"[bold]Rounds in DB[/bold]: {n}")
    if mx is not None:
        console.print(f"[bold]Latest round[/bold]: {mx} -> next predict: {mx + 1}")
    else:
        console.print("[bold]Next predict[/bold]: 1 (empty DB)")


def cmd_input_round(round_no: int, numbers: list[int] | None) -> None:
    client = dbmod.get_client()
    if numbers is None:
        console.print(
            f"Enter 6 winning numbers for round {round_no} (1-45), space-separated:"
        )
        line = input().strip()
        parts = line.split()
        numbers = _parse_six(parts)
    upsert_round(client, round_no, numbers)
    console.print(f"[green]Saved round {round_no}:[/green] {sorted(numbers)}")


def cmd_predict(
    target: int | None,
    min_rounds_before_first_101: int,
    epochs: int,
) -> None:
    client = dbmod.get_client()
    history = dbmod.fetch_history_ordered(client)
    if not history:
        console.print("[red]No data. Enter history rounds first.[/red]")
        return
    max_r = max(r for r, _ in history)
    next_r = max_r + 1
    if target is None:
        target = next_r
    if target != next_r:
        console.print(
            f"[red]Target must be latest+1. Only round {next_r} is valid now.[/red]"
        )
        return
    if target == 101 and max_r < min_rounds_before_first_101:
        console.print(
            f"[red]Need rounds up to {min_rounds_before_first_101} before predicting "
            f"101. Latest is {max_r}.[/red]"
        )
        return

    console.print("[cyan]Training LSTM and generating 5 sets…[/cyan]")
    try:
        tr, sets, meta = pipeline.run_prediction(
            client, target_round=target, lstm_epochs=epochs
        )
    except RuntimeError as e:
        msg = str(e)
        if msg == "EMPTY_HISTORY":
            console.print("[red]No history.[/red]")
        elif msg.startswith("BAD_TARGET_ROUND"):
            console.print("[red]Invalid prediction target.[/red]")
        elif msg == "NEED_MORE_HISTORY":
            console.print("[red]Need at least 2 rounds in history to train.[/red]")
        else:
            raise
        return

    pid = dbmod.insert_prediction(client, tr, sets)
    console.print(f"[bold green]Saved prediction for round {tr}[/bold green] (id={pid})")
    console.print(f"Sequence length: {meta['seq_len']}, sum range: {meta['sum_range']}")
    console.print(f"LSTM top-5 score balls: {meta['lstm_probs_top5']}")
    tbl = Table(title=f"Round {tr} — 5 recommended sets")
    tbl.add_column("Set", justify="center")
    tbl.add_column("Numbers", justify="left")
    for i, s in enumerate(sets, start=1):
        tbl.add_row(str(i), " ".join(f"{n:02d}" for n in s))
    console.print(tbl)


def cmd_result(round_no: int, actual: list[int]) -> None:
    client = dbmod.get_client()
    pred_row = dbmod.latest_prediction_for_round(client, round_no)
    if not pred_row:
        console.print(f"[red]No prediction row for round {round_no}.[/red]")
        return
    predicted_sets = pred_row["predicted_sets"]
    if isinstance(predicted_sets, str):
        predicted_sets = json.loads(predicted_sets)
    predicted_sets = [[int(x) for x in row] for row in predicted_sets]
    actual = sorted(int(x) for x in actual)
    matches = pipeline.compute_matches(predicted_sets, actual)
    dbmod.update_prediction_actual(client, int(pred_row["id"]), actual, matches)
    upsert_round(client, round_no, actual)
    console.print(
        f"[bold]Round {round_no} actual[/bold]: "
        f"{' '.join(f'{n:02d}' for n in actual)}"
    )
    console.print("[bold]Matches per set[/bold]:")
    for m in matches:
        i = m["set_index"] + 1
        pred = m["predicted"]
        hit = m["matched_numbers"]
        cnt = m["match_count"]
        hit_style = (
            "[yellow]" + " ".join(f"{n:02d}" for n in hit) + "[/yellow]"
            if hit
            else "-"
        )
        console.print(
            f"  Set {i}: {' '.join(f'{n:02d}' for n in pred)} -> "
            f"[bold]{cnt}[/bold] hit(s): {hit_style}"
        )
    console.print(
        "[dim]Actual numbers upserted to lotto_history; "
        "next predict will retrain including this draw.[/dim]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Korean Lotto + Supabase predictor")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("status", help="Show DB round count")
    p0.set_defaults(func=lambda _: cmd_status())

    p1 = sub.add_parser("input", help="Upsert one round (6 numbers)")
    p1.add_argument("round", type=int, help="Round number")
    p1.add_argument(
        "numbers", type=int, nargs="*", help="Six numbers, or omit for interactive"
    )
    p1.set_defaults(
        func=lambda a: cmd_input_round(
            a.round, list(a.numbers) if len(a.numbers) == 6 else None
        )
    )

    p2 = sub.add_parser("predict", help="Predict next round (5 sets)")
    p2.add_argument(
        "--round",
        type=int,
        default=None,
        help="Target round (default: latest+1)",
    )
    p2.add_argument(
        "--min-for-101",
        type=int,
        default=100,
        help="Min latest round before allowing predict for 101 (default 100)",
    )
    p2.add_argument("--epochs", type=int, default=120, help="LSTM training epochs")
    p2.set_defaults(func=lambda a: cmd_predict(a.round, a.min_for_101, a.epochs))

    p3 = sub.add_parser("result", help="Submit actual draw; compute matches")
    p3.add_argument("round", type=int)
    p3.add_argument("numbers", type=int, nargs=6)
    p3.set_defaults(func=lambda a: cmd_result(a.round, list(a.numbers)))

    args = parser.parse_args()
    try:
        args.func(args)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()

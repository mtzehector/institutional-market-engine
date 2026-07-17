from __future__ import annotations

import pandas as pd

from market_engine.evaluation.universe_tournament import (
    STRATEGIES,
    add_strategy_scores,
    build_tournament_selections,
    run_universe_tournament,
)


def _frame(days: int = 45, tickers: int = 12) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2026-01-02", periods=days, freq="B")
    for day_index, day in enumerate(dates):
        for ticker_index in range(tickers):
            strength = ticker_index / max(tickers - 1, 1)
            if (day_index + ticker_index) % 13 == 0:
                actual = "GAP_DOWN"
            elif (day_index + ticker_index) % 7 == 0:
                actual = "GAP_UP"
            else:
                actual = "SIN_GAP"
            predicted = actual if ticker_index >= tickers // 2 else "SIN_GAP"
            rows.append(
                {
                    "ticker": f"T{ticker_index:02d}",
                    "origin_date": day,
                    "target_date": day + pd.offsets.BDay(1),
                    "return_1d_pct": strength,
                    "return_3d_pct": strength * 2,
                    "return_5d_pct": strength * 3,
                    "close_position_pct": 40 + 50 * strength,
                    "close_vs_vwap_pct": -1 + 2 * strength,
                    "relative_volume": 0.8 + strength,
                    "smart_money_pct": 40 + 20 * strength,
                    "smart_money_slope_5d": -2 + 4 * strength,
                    "atr_pct": 1.0 + strength,
                    "actual_gap_pct": 1.4 if actual == "GAP_UP" else (-1.3 if actual == "GAP_DOWN" else 0.2),
                    "actual_gap_up": int(actual == "GAP_UP"),
                    "actual_gap_down": int(actual == "GAP_DOWN"),
                    "actual_direction": actual,
                    "probability_up": 0.72 if predicted == "GAP_UP" else 0.10,
                    "probability_down": 0.72 if predicted == "GAP_DOWN" else 0.10,
                    "probability_no_gap": 0.80 if predicted == "SIN_GAP" else 0.18,
                    "predicted_direction": predicted,
                    "correct_direction": predicted == actual,
                }
            )
    return pd.DataFrame(rows)


def test_strategy_scores_rank_expected_leaders() -> None:
    scored = add_strategy_scores(_frame(days=1, tickers=10))
    assert scored.sort_values("rank_MOMENTUM").iloc[0]["ticker"] == "T09"
    assert scored.sort_values("rank_VOLUME").iloc[0]["ticker"] == "T09"
    assert scored.sort_values("rank_SMART_MONEY").iloc[0]["ticker"] == "T09"


def test_tournament_builds_every_strategy_and_size() -> None:
    scored = add_strategy_scores(_frame(days=2, tickers=12))
    selections = build_tournament_selections(scored, (5, 10))
    expected = {f"{strategy}_TOP_{size}" for strategy in STRATEGIES for size in (5, 10)}
    expected.add("UNIVERSE")
    assert set(selections["tournament_entry"]) == expected
    counts = selections.groupby(["tournament_entry", "origin_date"])["ticker"].nunique()
    assert (counts.loc["MOMENTUM_TOP_5"] == 5).all()


def test_tournament_returns_auditable_leaderboard() -> None:
    result = run_universe_tournament(_frame(), (5, 10))
    assert not result.leaderboard.empty
    assert "UNIVERSE" in set(result.leaderboard["tournament_entry"])
    assert "HYBRID_TOP_5" in set(result.leaderboard["tournament_entry"])
    assert result.leaderboard["tournament_score"].between(0, 100).all()
    assert not result.daily_diagnostics.empty

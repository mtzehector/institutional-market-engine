from __future__ import annotations

import numpy as np
import pandas as pd

from market_engine.evaluation.adaptive_universe import (
    add_bullish_score,
    build_daily_cohorts,
    run_adaptive_universe,
)


def _frame(days: int = 50, tickers: int = 12) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2026-01-02", periods=days, freq="B")
    for day_index, day in enumerate(dates):
        for ticker_index in range(tickers):
            strength = ticker_index / max(tickers - 1, 1)
            actual = "GAP_UP" if (day_index + ticker_index) % 9 == 0 else "SIN_GAP"
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
                    "qqq_return_1d_pct": 0.4,
                    "atr_pct": 1.0 + strength,
                    "actual_gap_pct": 1.4 if actual == "GAP_UP" else 0.2,
                    "actual_gap_up": int(actual == "GAP_UP"),
                    "actual_gap_down": 0,
                    "actual_direction": actual,
                    "probability_up": 0.72 if predicted == "GAP_UP" else 0.12,
                    "probability_down": 0.08,
                    "probability_no_gap": 0.20 if predicted == "GAP_UP" else 0.80,
                    "predicted_direction": predicted,
                    "correct_direction": predicted == actual,
                }
            )
    return pd.DataFrame(rows)


def test_bullish_score_orders_stronger_ticker_first() -> None:
    scored = add_bullish_score(_frame(days=1, tickers=10))
    leader = scored.sort_values("bullish_rank").iloc[0]
    assert leader["ticker"] == "T09"
    assert leader["bullish_rank"] == 1


def test_daily_cohorts_have_requested_sizes() -> None:
    scored = add_bullish_score(_frame(days=3, tickers=12))
    cohorts = build_daily_cohorts(scored, (5, 10))
    counts = cohorts.groupby(["cohort", "origin_date"])["ticker"].nunique()
    assert (counts.loc["TOP_5"] == 5).all()
    assert (counts.loc["TOP_10"] == 10).all()
    assert (counts.loc["UNIVERSE"] == 12).all()


def test_adaptive_universe_compares_cohorts() -> None:
    result = run_adaptive_universe(_frame(), (5, 10))
    assert set(result.cohort_metrics["cohort"]) == {"TOP_5", "TOP_10", "UNIVERSE"}
    assert not result.selections.empty
    assert not result.daily_diagnostics.empty
    top_five = result.cohort_metrics.loc[result.cohort_metrics["cohort"] == "TOP_5"].iloc[0]
    assert np.isfinite(top_five["balanced_accuracy"])

from __future__ import annotations

import pandas as pd

from market_engine.evaluation.champion_validation import (
    CHAMPIONS,
    build_champion_selections,
    build_stability_ranking,
    build_window_definitions,
    add_champion_scores,
)


def _frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2026-01-02", periods=80, freq="B")
    for day_index, day in enumerate(dates):
        for ticker_index in range(30):
            strength = ticker_index / 29
            actual = "GAP_UP" if (day_index + ticker_index) % 11 == 0 else "SIN_GAP"
            predicted = actual if ticker_index > 15 else "SIN_GAP"
            rows.append(
                {
                    "ticker": f"T{ticker_index:02d}",
                    "origin_date": day,
                    "target_date": day + pd.offsets.BDay(1),
                    "actual_gap_pct": 1.4 if actual == "GAP_UP" else 0.1,
                    "actual_gap_up": int(actual == "GAP_UP"),
                    "actual_gap_down": 0,
                    "actual_direction": actual,
                    "probability_up": 0.72 if predicted == "GAP_UP" else 0.12,
                    "probability_down": 0.08,
                    "probability_no_gap": 0.20 if predicted == "GAP_UP" else 0.80,
                    "predicted_direction": predicted,
                    "correct_direction": predicted == actual,
                    "return_1d_pct": strength,
                    "return_3d_pct": strength * 2,
                    "return_5d_pct": strength * 3,
                    "relative_volume": 0.8 + strength,
                    "close_position_pct": 40 + 50 * strength,
                    "close_vs_vwap_pct": -1 + 2 * strength,
                    "smart_money_pct": 40 + 20 * strength,
                    "smart_money_slope_5d": -2 + 4 * strength,
                    "atr_pct": 2.0 - strength,
                }
            )
    return pd.DataFrame(rows)


def test_champion_scores_and_selections() -> None:
    scored = add_champion_scores(_frame())
    selections = build_champion_selections(scored)
    expected = set(CHAMPIONS) | {"UNIVERSE"}
    assert set(selections["champion"]) == expected
    counts = selections.loc[selections["champion"] == "PARSIMONIOUS_MOMENTUM_VOLUME"].groupby("origin_date")["ticker"].nunique()
    assert (counts == 20).all()


def test_monthly_windows_are_chronological() -> None:
    windows = build_window_definitions(_frame(), "M")
    assert windows["window_start"].is_monotonic_increasing
    assert windows["window"].nunique() >= 3


def test_stability_ranking_orders_higher_quality_first() -> None:
    metrics = pd.DataFrame(
        [
            {"champion": "A", "window": "2026-01", "rare_event_f1": 0.5, "balanced_accuracy": 0.6, "macro_f1": 0.55, "mean_brier_skill": 0.1, "mean_calibration_error": 0.1},
            {"champion": "A", "window": "2026-02", "rare_event_f1": 0.48, "balanced_accuracy": 0.59, "macro_f1": 0.54, "mean_brier_skill": 0.08, "mean_calibration_error": 0.11},
            {"champion": "B", "window": "2026-01", "rare_event_f1": 0.2, "balanced_accuracy": 0.4, "macro_f1": 0.35, "mean_brier_skill": -0.1, "mean_calibration_error": 0.2},
            {"champion": "B", "window": "2026-02", "rare_event_f1": 0.1, "balanced_accuracy": 0.35, "macro_f1": 0.3, "mean_brier_skill": -0.2, "mean_calibration_error": 0.25},
        ]
    )
    ranking = build_stability_ranking(metrics)
    assert ranking.iloc[0]["champion"] == "A"

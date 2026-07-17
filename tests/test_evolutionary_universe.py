from __future__ import annotations

import numpy as np
import pandas as pd

from market_engine.evaluation.evolutionary_universe import (
    BASE_FACTORS,
    add_evolutionary_factor_scores,
    generate_candidate_definitions,
    run_evolutionary_universe,
    score_candidates,
)


def _frame(days: int = 40, tickers: int = 12) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2026-01-02", periods=days, freq="B")
    for day_index, day in enumerate(dates):
        for ticker_index in range(tickers):
            strength = ticker_index / max(tickers - 1, 1)
            rare = (day_index + ticker_index) % 8 == 0
            actual = "GAP_UP" if rare else "SIN_GAP"
            predicted = actual if ticker_index >= tickers // 2 else "SIN_GAP"
            rows.append(
                {
                    "ticker": f"T{ticker_index:02d}",
                    "origin_date": day,
                    "target_date": day + pd.offsets.BDay(1),
                    "return_1d_pct": strength,
                    "return_3d_pct": strength * 2,
                    "return_5d_pct": strength * 3,
                    "relative_volume": 0.8 + strength,
                    "close_position_pct": 40 + 50 * strength,
                    "close_vs_vwap_pct": -1 + 2 * strength,
                    "smart_money_pct": 40 + 20 * strength,
                    "smart_money_slope_5d": -2 + 4 * strength,
                    "atr_pct": 2.0 - strength,
                    "actual_gap_pct": 1.5 if rare else 0.2,
                    "actual_gap_up": int(rare),
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


def test_candidate_generation_is_bounded() -> None:
    definitions = generate_candidate_definitions((5, 10), max_factors=2)
    expected_combinations = len(BASE_FACTORS) + (len(BASE_FACTORS) * (len(BASE_FACTORS) - 1)) // 2
    assert len(definitions) == expected_combinations * 2
    assert definitions["factor_count"].max() == 2


def test_candidate_selection_respects_daily_cohort_size() -> None:
    scored = add_evolutionary_factor_scores(_frame(days=3))
    definitions = generate_candidate_definitions((5,), max_factors=1).head(1)
    selected = score_candidates(scored, definitions)
    counts = selected.groupby("origin_date")["ticker"].nunique()
    assert (counts == 5).all()


def test_evolutionary_laboratory_builds_holdout_leaderboard() -> None:
    result = run_evolutionary_universe(
        _frame(),
        cohort_sizes=(5, 10),
        max_factors=2,
        validation_fraction=0.30,
        hall_of_fame_size=5,
    )
    assert not result.leaderboard.empty
    assert not result.hall_of_fame.empty
    assert len(result.hall_of_fame) <= 5
    assert {"discovery_rare_event_f1", "validation_rare_event_f1"}.issubset(
        result.leaderboard.columns
    )
    assert np.isfinite(result.leaderboard.iloc[0]["evolution_score"])

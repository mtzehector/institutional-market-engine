from __future__ import annotations

import numpy as np
import pandas as pd

from market_engine.evaluation.model_evaluation import (
    evaluate_predictions,
    rank_ticker_evaluations,
)


def _prediction_frame(correct_ratio: float = 0.75, rows: int = 80) -> pd.DataFrame:
    dates = pd.date_range("2025-01-02", periods=rows, freq="B")
    actual_direction = np.where(np.arange(rows) % 5 == 0, "GAP_UP", "SIN_GAP")
    actual_direction = np.where(np.arange(rows) % 7 == 0, "GAP_DOWN", actual_direction)

    predicted_direction = actual_direction.copy()
    incorrect_count = int(rows * (1.0 - correct_ratio))
    predicted_direction[:incorrect_count] = "SIN_GAP"

    probability_up = np.where(predicted_direction == "GAP_UP", 0.72, 0.12)
    probability_down = np.where(predicted_direction == "GAP_DOWN", 0.70, 0.10)
    probability_no_gap = 1.0 - probability_up - probability_down

    return pd.DataFrame(
        {
            "target_date": dates,
            "actual_gap_pct": np.where(
                actual_direction == "GAP_UP",
                1.5,
                np.where(actual_direction == "GAP_DOWN", -1.4, 0.2),
            ),
            "actual_gap_up": (actual_direction == "GAP_UP").astype(int),
            "actual_gap_down": (actual_direction == "GAP_DOWN").astype(int),
            "actual_direction": actual_direction,
            "probability_up": probability_up,
            "probability_down": probability_down,
            "probability_no_gap": probability_no_gap,
            "predicted_direction": predicted_direction,
            "correct_direction": predicted_direction == actual_direction,
        }
    )


def test_evaluate_predictions_returns_score_and_grade() -> None:
    result = evaluate_predictions(_prediction_frame(), ticker="TEAM")
    summary = result.summary.iloc[0]

    assert summary["ticker"] == "TEAM"
    assert 0 <= summary["predictability_score"] <= 100
    assert summary["observations"] == 80
    assert isinstance(summary["predictability_grade"], str)
    assert not result.confidence_bands.empty
    assert not result.recent_performance.empty


def test_better_predictions_receive_higher_score() -> None:
    strong = evaluate_predictions(
        _prediction_frame(correct_ratio=0.90), ticker="STRONG"
    ).summary.iloc[0]["predictability_score"]
    weak = evaluate_predictions(
        _prediction_frame(correct_ratio=0.45), ticker="WEAK"
    ).summary.iloc[0]["predictability_score"]

    assert strong > weak


def test_rank_ticker_evaluations_orders_score_descending() -> None:
    summaries = pd.DataFrame(
        [
            {"ticker": "BBB", "predictability_score": 52.0, "observations": 100, "directional_accuracy": 0.55},
            {"ticker": "AAA", "predictability_score": 81.0, "observations": 80, "directional_accuracy": 0.70},
        ]
    )
    ranking = rank_ticker_evaluations(summaries)

    assert ranking.iloc[0]["ticker"] == "AAA"
    assert ranking.iloc[0]["rank"] == 1

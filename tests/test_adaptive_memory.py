from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from market_engine.backtesting.walk_forward import WalkForwardResult, walk_forward_gap
from market_engine.evaluation import adaptive_memory
from market_engine.evaluation.adaptive_memory import (
    evaluate_memory_windows,
    rank_best_memories,
)


def test_walk_forward_rejects_memory_shorter_than_minimum() -> None:
    with pytest.raises(ValueError, match="max_history_rows"):
        walk_forward_gap(
            pd.DataFrame({"date": pd.date_range("2025-01-01", periods=10)}),
            from_date=date(2025, 1, 1),
            to_date=date(2025, 1, 10),
            min_history_rows=180,
            max_history_rows=90,
        )


def test_rank_best_memories_orders_score_descending() -> None:
    frame = pd.DataFrame(
        [
            {
                "ticker": "TEAM",
                "memory_label": "252_ROWS",
                "predictability_score": 72.0,
                "observations": 80,
                "directional_accuracy": 0.68,
            },
            {
                "ticker": "VIRT",
                "memory_label": "504_ROWS",
                "predictability_score": 81.0,
                "observations": 75,
                "directional_accuracy": 0.74,
            },
        ]
    )
    ranking = rank_best_memories(frame)
    assert ranking.iloc[0]["ticker"] == "VIRT"
    assert ranking.iloc[0]["adaptive_rank"] == 1


def test_memory_window_keeps_one_extra_origin_row(monkeypatch) -> None:
    retained: list[int | None] = []

    predictions = pd.DataFrame(
        {
            "target_date": pd.date_range("2026-01-05", periods=40, freq="B"),
            "actual_gap_pct": [0.0] * 40,
            "actual_gap_up": [0] * 40,
            "actual_gap_down": [0] * 40,
            "actual_direction": ["SIN_GAP"] * 40,
            "probability_up": [0.10] * 40,
            "probability_down": [0.10] * 40,
            "probability_no_gap": [0.80] * 40,
            "predicted_direction": ["SIN_GAP"] * 40,
            "correct_direction": [True] * 40,
        }
    )

    def fake_walk_forward(*args, **kwargs):
        retained.append(kwargs["max_history_rows"])
        return WalkForwardResult(
            predictions=predictions,
            metrics=pd.DataFrame(),
            calibration=pd.DataFrame(),
        )

    monkeypatch.setattr(adaptive_memory, "walk_forward_gap", fake_walk_forward)

    result = evaluate_memory_windows(
        pd.DataFrame({"date": pd.date_range("2025-01-01", periods=300)}),
        ticker="SNDK",
        from_date=date(2026, 1, 2),
        to_date=date(2026, 7, 15),
        memory_windows=[180],
    )

    assert retained == [181]
    assert result.best_by_ticker.iloc[0]["memory_label"] == "180_ROWS"


def test_one_failed_window_does_not_cancel_other_windows(monkeypatch) -> None:
    calls: list[int | None] = []

    predictions = pd.DataFrame(
        {
            "target_date": pd.date_range("2026-01-05", periods=40, freq="B"),
            "actual_gap_pct": [0.0] * 40,
            "actual_gap_up": [0] * 40,
            "actual_gap_down": [0] * 40,
            "actual_direction": ["SIN_GAP"] * 40,
            "probability_up": [0.10] * 40,
            "probability_down": [0.10] * 40,
            "probability_no_gap": [0.80] * 40,
            "predicted_direction": ["SIN_GAP"] * 40,
            "correct_direction": [True] * 40,
        }
    )

    def fake_walk_forward(*args, **kwargs):
        value = kwargs["max_history_rows"]
        calls.append(value)
        if value == 181:
            raise ValueError("histórico insuficiente")
        return WalkForwardResult(
            predictions=predictions,
            metrics=pd.DataFrame(),
            calibration=pd.DataFrame(),
        )

    monkeypatch.setattr(adaptive_memory, "walk_forward_gap", fake_walk_forward)

    result = evaluate_memory_windows(
        pd.DataFrame({"date": pd.date_range("2025-01-01", periods=400)}),
        ticker="SNDK",
        from_date=date(2026, 1, 2),
        to_date=date(2026, 7, 15),
        memory_windows=[180, 252],
    )

    assert calls == [181, 253]
    assert result.best_by_ticker.iloc[0]["memory_label"] == "252_ROWS"
    assert result.errors.iloc[0]["memory_label"] == "180_ROWS"

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from market_engine.backtesting.walk_forward import walk_forward_gap
from market_engine.evaluation.adaptive_memory import rank_best_memories


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

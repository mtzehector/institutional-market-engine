from __future__ import annotations

import numpy as np
import pandas as pd

from market_engine.gaps.predictor import _comparable_probability


COMPARABLE_COLUMNS = [
    "smart_money_pct",
    "smart_money_delta_3d",
    "relative_volume",
    "atr_pct",
    "return_1d_pct",
    "return_5d_pct",
    "close_position_pct",
    "gap_today_pct",
    "qqq_return_1d_pct",
    "volatility_10d_pct",
]


def test_comparable_probability_accepts_object_numeric_columns() -> None:
    rows = 60
    frame = pd.DataFrame(
        {
            column: pd.Series(
                np.linspace(1.0, 2.0, rows),
                dtype="object",
            )
            for column in COMPARABLE_COLUMNS
        }
    )
    frame["target_gap_up_next"] = [0, 1] * (rows // 2)

    latest = frame.iloc[-1].copy()
    probability = _comparable_probability(
        frame,
        latest,
        "target_gap_up_next",
    )

    assert 0.0 < probability < 1.0

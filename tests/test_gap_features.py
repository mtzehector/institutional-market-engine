from __future__ import annotations

import numpy as np
import pandas as pd

from market_engine.gaps.features import build_gap_features


def sample_frame(rows: int = 220) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-01", periods=rows)
    close = pd.Series(np.linspace(100, 130, rows))
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": close.shift(1).fillna(close.iloc[0]) * 1.001,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.linspace(1_000_000, 2_000_000, rows),
            "vwap": close * 0.999,
        }
    )
    return frame


def test_last_row_has_no_future_target() -> None:
    frame = sample_frame()
    features = build_gap_features(frame, frame.copy())
    assert pd.isna(features.iloc[-1]["target_gap_up_next"])
    assert pd.isna(features.iloc[-1]["target_gap_down_next"])


def test_gap_threshold_never_below_minimum() -> None:
    frame = sample_frame()
    features = build_gap_features(frame, frame.copy(), gap_min_pct=1.0)
    mature = features["gap_threshold_pct"].dropna()
    assert (mature >= 1.0).all()

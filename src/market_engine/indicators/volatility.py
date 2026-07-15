from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(frame: pd.DataFrame) -> pd.Series:
    previous_close = frame["close"].shift(1)
    return pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr_percent(frame: pd.DataFrame, length: int = 14) -> pd.Series:
    atr = true_range(frame).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    return atr / frame["close"] * 100


def efficiency_ratio(close: pd.Series, window: int = 10) -> pd.Series:
    net_change = (close - close.shift(window)).abs()
    path = close.diff().abs().rolling(window).sum()
    return net_change / path.replace(0, np.nan)

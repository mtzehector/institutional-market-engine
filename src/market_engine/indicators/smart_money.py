from __future__ import annotations

import numpy as np
import pandas as pd


def rsi_wilder(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    average_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    average_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(average_loss != 0, 100.0)
    rsi = rsi.where(average_gain != 0, 0.0)
    return rsi.clip(0, 100)


def add_smart_money(
    frame: pd.DataFrame,
    length: int = 14,
    volume_length: int = 20,
    ema_length: int = 5,
) -> pd.DataFrame:
    data = frame.copy()
    volume_average = data["volume"].rolling(volume_length, min_periods=volume_length).mean()
    raw_flow = data["close"].diff() * data["volume"] / volume_average.replace(0, np.nan)
    smart_flow = raw_flow.ewm(span=ema_length, adjust=False, min_periods=ema_length).mean()
    data["smart_money_pct"] = rsi_wilder(smart_flow, length)
    data["retail_pct"] = 100 - data["smart_money_pct"]
    data["smart_money_delta_1d"] = data["smart_money_pct"].diff()
    data["smart_money_delta_3d"] = data["smart_money_pct"].diff(3)
    data["smart_money_slope_5d"] = data["smart_money_pct"].diff(5)
    return data

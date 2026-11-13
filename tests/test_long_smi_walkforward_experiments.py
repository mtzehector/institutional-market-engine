"""Controles temporales para los experimentos 36 y 37."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / file)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


E36 = module("experiment36", "36_long_smi_filter_walkforward_v1.py")
E37 = module("experiment37", "37_long_smi_ticker_selection_walkforward_v1.py")


def synthetic_daily():
    dates = pd.bdate_range("2025-01-06", periods=130)
    close = 100 + np.arange(len(dates)) * 0.25 + np.sin(np.arange(len(dates)) / 3)
    return pd.DataFrame({
        "open": close - 0.1, "high": close + 1, "low": close - 1,
        "close": close, "volume": 1000,
    }, index=dates)


def test_weekly_value_cannot_see_later_bars_in_same_week():
    original = synthetic_daily()
    changed = original.copy()
    target = pd.Timestamp("2025-06-27")
    changed.loc[target, ["open", "high", "low", "close"]] *= 5
    p1 = E36.daily_weekly_panel(original, 1)
    p2 = E36.daily_weekly_panel(changed, 1)
    monday = pd.Timestamp("2025-06-23")
    friday = pd.Timestamp("2025-06-27")
    next_monday = pd.Timestamp("2025-06-30")
    assert p1.loc[monday, "Weekly_SMI"] == p2.loc[monday, "Weekly_SMI"]
    assert p1.loc[friday, "Weekly_SMI"] == p2.loc[friday, "Weekly_SMI"]
    assert p1.loc[next_monday, "Weekly_SMI"] != p2.loc[next_monday, "Weekly_SMI"]


def test_forward_return_starts_at_next_open():
    daily = synthetic_daily().iloc[:15]
    panel = E36.forward_outcome(daily, 3)
    expected = (daily["close"].iloc[3] / daily["open"].iloc[1] - 1) * 100
    assert panel["Entry_Date"].iloc[0] == daily.index[1]
    assert panel["Exit_Date"].iloc[0] == daily.index[3]
    assert abs(panel["Forward_Return_Pct"].iloc[0] - expected) < 1e-12


def test_training_selection_uses_only_passed_frame():
    train = pd.DataFrame({
        "Ticker": ["A"] * 6 + ["B"] * 6,
        "Eligible": [True] * 3 + [False] * 3 + [True] * 3 + [False] * 3,
        "Forward_Return_Pct": [2, 2, 2, 0, 0, 0, -2, -2, -2, 0, 0, 0],
    })
    choice = E37.select_tickers(train, 2, 2, 0.5)
    assert choice.loc[choice["Selected"], "Ticker"].tolist() == ["A"]

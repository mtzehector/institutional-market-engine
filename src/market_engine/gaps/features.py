from __future__ import annotations

import numpy as np
import pandas as pd

from market_engine.indicators.smart_money import add_smart_money
from market_engine.indicators.volatility import atr_percent, efficiency_ratio


FEATURE_COLUMNS = [
    "return_1d_pct",
    "return_3d_pct",
    "return_5d_pct",
    "intraday_return_pct",
    "range_pct",
    "close_position_pct",
    "relative_volume",
    "atr_pct",
    "volatility_10d_pct",
    "gap_today_pct",
    "gap_abs_today_pct",
    "gap_up_frequency_20d",
    "gap_down_frequency_20d",
    "gap_significant_frequency_60d",
    "close_vs_vwap_pct",
    "efficiency_ratio_10d",
    "positive_persistence_10d",
    "smart_money_pct",
    "smart_money_delta_1d",
    "smart_money_delta_3d",
    "smart_money_slope_5d",
    "qqq_return_1d_pct",
    "qqq_return_5d_pct",
    "qqq_volatility_10d_pct",
    "qqq_gap_today_pct",
    "weekday_sin",
    "weekday_cos",
]


def build_gap_features(
    frame: pd.DataFrame,
    qqq: pd.DataFrame,
    *,
    gap_min_pct: float = 1.0,
    gap_atr_multiplier: float = 0.5,
    atr_length: int = 14,
    smart_money_length: int = 14,
    smart_money_volume_length: int = 20,
    smart_money_ema: int = 5,
) -> pd.DataFrame:
    data = frame.sort_values("date").reset_index(drop=True).copy()
    previous_close = data["close"].shift(1)

    data["atr_pct"] = atr_percent(data, atr_length)
    data["gap_threshold_pct"] = np.maximum(
        gap_min_pct, gap_atr_multiplier * data["atr_pct"]
    )
    data["gap_today_pct"] = (data["open"] - previous_close) / previous_close * 100
    data["gap_abs_today_pct"] = data["gap_today_pct"].abs()
    prior_threshold = data["gap_threshold_pct"].shift(1)
    data["gap_up_today"] = (data["gap_today_pct"] >= prior_threshold).astype(float)
    data["gap_down_today"] = (data["gap_today_pct"] <= -prior_threshold).astype(float)
    data["gap_significant_today"] = (
        (data["gap_up_today"] == 1) | (data["gap_down_today"] == 1)
    ).astype(float)

    # Etiqueta t+1: usa el umbral conocido al cierre de t.
    data["gap_next_pct"] = (
        data["open"].shift(-1) - data["close"]
    ) / data["close"] * 100
    data["target_gap_up_next"] = np.where(
        data["gap_next_pct"].notna(),
        (data["gap_next_pct"] >= data["gap_threshold_pct"]).astype(int),
        np.nan,
    )
    data["target_gap_down_next"] = np.where(
        data["gap_next_pct"].notna(),
        (data["gap_next_pct"] <= -data["gap_threshold_pct"]).astype(int),
        np.nan,
    )

    data["return_1d_pct"] = data["close"].pct_change() * 100
    data["return_3d_pct"] = data["close"].pct_change(3) * 100
    data["return_5d_pct"] = data["close"].pct_change(5) * 100
    data["intraday_return_pct"] = (data["close"] - data["open"]) / data["open"] * 100
    data["range_pct"] = (data["high"] - data["low"]) / data["open"] * 100
    daily_range = data["high"] - data["low"]
    data["close_position_pct"] = np.where(
        daily_range > 0, (data["close"] - data["low"]) / daily_range * 100, 50.0
    )
    volume_average = data["volume"].rolling(20).mean()
    data["relative_volume"] = data["volume"] / volume_average
    data["volatility_10d_pct"] = data["return_1d_pct"].rolling(10).std()
    data["close_vs_vwap_pct"] = np.where(
        data["vwap"].notna() & (data["vwap"] != 0),
        (data["close"] - data["vwap"]) / data["vwap"] * 100,
        np.nan,
    )
    data["gap_up_frequency_20d"] = data["gap_up_today"].rolling(20).mean()
    data["gap_down_frequency_20d"] = data["gap_down_today"].rolling(20).mean()
    data["gap_significant_frequency_60d"] = data["gap_significant_today"].rolling(60).mean()
    data["efficiency_ratio_10d"] = efficiency_ratio(data["close"], 10)
    data["positive_persistence_10d"] = (
        (data["return_1d_pct"] > 0).astype(float).rolling(10).mean()
    )
    data = add_smart_money(
        data,
        smart_money_length,
        smart_money_volume_length,
        smart_money_ema,
    )

    weekday = data["date"].dt.weekday
    data["weekday_sin"] = np.sin(2 * np.pi * weekday / 5)
    data["weekday_cos"] = np.cos(2 * np.pi * weekday / 5)

    market = qqq.sort_values("date").reset_index(drop=True).copy()
    market["qqq_return_1d_pct"] = market["close"].pct_change() * 100
    market["qqq_return_5d_pct"] = market["close"].pct_change(5) * 100
    market["qqq_volatility_10d_pct"] = market["qqq_return_1d_pct"].rolling(10).std()
    market["qqq_gap_today_pct"] = (
        market["open"] - market["close"].shift(1)
    ) / market["close"].shift(1) * 100

    return data.merge(
        market[
            [
                "date",
                "qqq_return_1d_pct",
                "qqq_return_5d_pct",
                "qqq_volatility_10d_pct",
                "qqq_gap_today_pct",
            ]
        ],
        on="date",
        how="left",
    )

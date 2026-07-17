from __future__ import annotations

import numpy as np
import pandas as pd

from market_engine.evaluation.smart_money_regimes import (
    analyze_smart_money_regimes,
    correlation_report,
)


def _frame() -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=24, freq="B")
    smart_money = [
        47, 47.5, 48, 49.5, 52, 53, 54, 53, 52,
        50, 48, 47, 46, 47, 48, 50, 52, 53, 54, 55,
        54, 53, 52, 51.5,
    ]
    relative_volume = [
        0.9, 1.0, 1.1, 1.2, 1.8, 1.6, 1.4, 1.3, 1.2,
        1.1, 1.7, 1.5, 1.3, 1.2, 1.1, 1.0, 1.9, 1.6, 1.5, 1.4,
        1.3, 1.2, 1.1, 1.0,
    ]
    return pd.DataFrame(
        {
            "date": dates,
            "smart_money_pct": smart_money,
            "volume": np.array(relative_volume) * 1_000_000,
            "relative_volume": relative_volume,
        }
    )


def test_analyze_smart_money_regimes_detects_confirmed_crossings() -> None:
    result = analyze_smart_money_regimes(
        _frame(),
        ticker="TEST",
        memory_rows=24,
        minimum_persistence=2,
    )

    summary = result.summary.iloc[0]
    assert summary["ticker"] == "TEST"
    assert summary["equilibrium_cross_count"] >= 2
    assert summary["median_regime_duration"] > 0
    assert summary["median_institutional_regime_strength"] > 0
    assert not result.regimes.empty
    assert not result.crossings.empty
    assert "cross_relative_volume" in result.crossings.columns


def test_high_volume_regime_is_stronger_than_low_volume_regime() -> None:
    high = _frame()
    low = high.copy()
    low["relative_volume"] = low["relative_volume"] * 0.4

    high_strength = analyze_smart_money_regimes(
        high, ticker="HIGH", memory_rows=24
    ).summary.iloc[0]["median_institutional_regime_strength"]
    low_strength = analyze_smart_money_regimes(
        low, ticker="LOW", memory_rows=24
    ).summary.iloc[0]["median_institutional_regime_strength"]

    assert high_strength > low_strength


def test_correlation_report_relates_regime_metrics_to_model_metrics() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D"],
            "crossings_per_100_sessions": [1.0, 2.0, 4.0, 8.0],
            "median_regime_duration": [80.0, 50.0, 25.0, 10.0],
            "median_relative_volume": [1.4, 1.3, 1.2, 1.1],
            "smart_money_regime_coherence": [0.8, 0.6, 0.4, 0.2],
            "predictability_score": [80.0, 70.0, 55.0, 35.0],
            "rare_event_f1": [0.65, 0.55, 0.40, 0.20],
        }
    )
    report = correlation_report(frame)
    assert not report.empty
    assert set(report["target_metric"]) == {"predictability_score", "rare_event_f1"}
    assert report["spearman_correlation"].notna().any()

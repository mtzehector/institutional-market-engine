from __future__ import annotations

import pandas as pd

from market_engine.evaluation.market_memory_confidence import (
    run_market_memory_confidence_laboratory,
)


def _states(days: int = 20) -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=days, freq="B")
    return pd.DataFrame([
        {
            "origin_date": date,
            "regime": "RISK_ON_BROAD" if i % 4 < 2 else "ROTATION_VOLATILE",
            "qqq_return_1d_pct": 0.8 if i % 4 < 2 else -0.6,
            "breadth_positive_1d": 0.70 if i % 4 < 2 else 0.38,
            "breadth_positive_5d": 0.68 if i % 4 < 2 else 0.42,
            "median_relative_volume": 1.15 if i % 4 < 2 else 1.45,
            "median_atr_pct": 1.7 if i % 4 < 2 else 3.8,
            "institutional_breadth": 0.62 if i % 4 < 2 else 0.35,
        }
        for i, date in enumerate(dates)
    ])


def _selections(days: int = 20, tickers: int = 12) -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=days, freq="B")
    rows = []
    for i, date in enumerate(dates):
        risk_on = i % 4 < 2
        for champion in ("MOMENTUM", "INSTITUTIONAL", "UNIVERSE"):
            for ticker_index in range(tickers):
                rare = ticker_index % 5 == 0
                actual = "GAP_UP" if rare else "SIN_GAP"
                good = (champion == "MOMENTUM" and risk_on) or (champion == "INSTITUTIONAL" and not risk_on)
                predicted = actual if good else "SIN_GAP"
                rows.append({
                    "origin_date": date,
                    "target_date": date + pd.offsets.BDay(1),
                    "champion": champion,
                    "ticker": f"T{ticker_index:02d}",
                    "actual_gap_pct": 1.5 if rare else 0.1,
                    "actual_gap_up": int(rare),
                    "actual_gap_down": 0,
                    "actual_direction": actual,
                    "predicted_direction": predicted,
                    "correct_direction": predicted == actual,
                    "probability_up": 0.75 if predicted == "GAP_UP" else 0.08,
                    "probability_down": 0.05,
                    "probability_no_gap": 0.20 if predicted == "GAP_UP" else 0.87,
                })
    return pd.DataFrame(rows)


def test_confidence_laboratory_builds_decisions_and_validation() -> None:
    result = run_market_memory_confidence_laboratory(
        _states(), _selections(), ks=(3, 5, 7), baseline_k=5, minimum_history=8
    )
    assert not result.recommendations.empty
    assert result.recommendations["memory_confidence_score"].between(0, 100).all()
    assert set(result.recommendations["decision_level"]).issubset(
        {"RECOMMEND", "CAUTIOUS_RECOMMEND", "WATCH", "ABSTAIN"}
    )
    assert not result.sensitivity_k.empty
    assert set(result.sensitivity_k["k"]) == {3, 5, 7}
    assert not result.confidence_validation.empty
    assert not result.abstention_counterfactual.empty

from __future__ import annotations

import numpy as np
import pandas as pd

from market_engine.evaluation.market_memory import (
    build_date_champion_quality,
    find_causal_neighbors,
    run_market_memory_laboratory,
)


def _states(days: int = 16) -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=days, freq="B")
    rows: list[dict[str, object]] = []
    for index, date in enumerate(dates):
        risk_on = index % 4 < 2
        rows.append(
            {
                "origin_date": date,
                "regime": "RISK_ON_BROAD" if risk_on else "ROTATION_VOLATILE",
                "qqq_return_1d_pct": 0.8 if risk_on else -0.6,
                "breadth_positive_1d": 0.70 if risk_on else 0.38,
                "breadth_positive_5d": 0.68 if risk_on else 0.42,
                "median_relative_volume": 1.15 if risk_on else 1.45,
                "median_atr_pct": 1.7 if risk_on else 3.8,
                "institutional_breadth": 0.62 if risk_on else 0.35,
            }
        )
    return pd.DataFrame(rows)


def _selections(days: int = 16, tickers: int = 12) -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=days, freq="B")
    champions = ["MOMENTUM", "INSTITUTIONAL", "UNIVERSE"]
    rows: list[dict[str, object]] = []
    for index, date in enumerate(dates):
        risk_on = index % 4 < 2
        for champion in champions:
            for ticker_index in range(tickers):
                rare = ticker_index % 5 == 0
                actual = "GAP_UP" if rare else "SIN_GAP"
                good = (
                    champion == "MOMENTUM" and risk_on
                ) or (
                    champion == "INSTITUTIONAL" and not risk_on
                )
                predicted = actual if good else "SIN_GAP"
                rows.append(
                    {
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
                    }
                )
    return pd.DataFrame(rows)


def test_neighbors_are_strictly_causal() -> None:
    neighbors = find_causal_neighbors(_states(), neighbors=3, minimum_history=6)
    assert not neighbors.empty
    assert (neighbors["neighbor_date"] < neighbors["target_date"]).all()
    assert set(neighbors.groupby("target_date").size()) == {3}
    assert np.isfinite(neighbors["distance"]).all()


def test_date_quality_is_built_for_every_champion() -> None:
    quality = build_date_champion_quality(_selections())
    assert not quality.empty
    assert set(quality["champion"]) == {"MOMENTUM", "INSTITUTIONAL", "UNIVERSE"}
    assert quality["quality_score"].notna().all()


def test_market_memory_generates_auditable_recommendations() -> None:
    result = run_market_memory_laboratory(
        _states(),
        _selections(),
        neighbors=3,
        minimum_history=6,
    )
    assert not result.recommendations.empty
    assert not result.neighbor_states.empty
    assert not result.champion_scores.empty
    assert not result.summary.empty
    assert result.recommendations["recommended_champion"].notna().all()
    assert (result.recommendations["neighbor_dates_used"] == 3).all()
    assert result.summary.iloc[0]["recommendation_dates"] == len(result.recommendations)

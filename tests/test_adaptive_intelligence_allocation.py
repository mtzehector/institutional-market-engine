from __future__ import annotations

import numpy as np
import pandas as pd

from market_engine.evaluation.adaptive_intelligence_allocation import (
    ALLOCATION_POLICIES,
    build_current_intelligence_states,
    run_adaptive_intelligence_allocation_laboratory,
)


def _states(days: int = 24) -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=days, freq="B")
    rows: list[dict[str, object]] = []
    for index, date in enumerate(dates):
        risk_on = index % 6 < 3
        rows.append(
            {
                "origin_date": date,
                "regime": "RISK_ON_BROAD" if risk_on else "ROTATION_VOLATILE",
                "qqq_return_1d_pct": 0.9 if risk_on else -0.7,
                "breadth_positive_1d": 0.72 if risk_on else 0.36,
                "breadth_positive_5d": 0.68 if risk_on else 0.40,
                "median_relative_volume": 1.15 if risk_on else 1.45,
                "median_atr_pct": 1.8 if risk_on else 4.0,
                "institutional_breadth": 0.60 if risk_on else 0.38,
            }
        )
    return pd.DataFrame(rows)


def _selections(days: int = 24, tickers: int = 14) -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=days, freq="B")
    champions = [
        "CHAMPION_VOLUME_CONFIDENCE_STRUCTURE",
        "PARSIMONIOUS_MOMENTUM_VOLUME",
        "INSTITUTIONAL_STABLE",
        "INSTITUTIONAL_CONFIRMED",
        "CONTROL_VOLUME",
        "CONTROL_MOMENTUM",
        "CONTROL_SMART_MONEY",
        "UNIVERSE",
    ]
    rows: list[dict[str, object]] = []
    for index, date in enumerate(dates):
        risk_on = index % 6 < 3
        for champion in champions:
            for ticker_index in range(tickers):
                rare = ticker_index % 4 == 0
                actual = "GAP_UP" if rare else "SIN_GAP"
                good = (
                    risk_on
                    and champion in {
                        "CONTROL_MOMENTUM",
                        "PARSIMONIOUS_MOMENTUM_VOLUME",
                    }
                ) or (
                    not risk_on
                    and champion in {
                        "INSTITUTIONAL_STABLE",
                        "CONTROL_SMART_MONEY",
                    }
                )
                predicted = actual if good else "SIN_GAP"
                rows.append(
                    {
                        "origin_date": date,
                        "target_date": date + pd.offsets.BDay(1),
                        "champion": champion,
                        "ticker": f"T{ticker_index:02d}",
                        "actual_gap_pct": 1.6 if rare else 0.1,
                        "actual_gap_up": int(rare),
                        "actual_gap_down": 0,
                        "actual_direction": actual,
                        "predicted_direction": predicted,
                        "correct_direction": predicted == actual,
                        "probability_up": 0.76 if predicted == "GAP_UP" else 0.08,
                        "probability_down": 0.04,
                        "probability_no_gap": 0.20 if predicted == "GAP_UP" else 0.88,
                    }
                )
    return pd.DataFrame(rows)


def test_current_intelligence_states_are_causal_and_bounded() -> None:
    states = build_current_intelligence_states(_states(), minimum_history=8)
    assert not states.empty
    score_columns = [column for column in states.columns if column.startswith("score_")]
    assert score_columns
    assert np.isfinite(states[score_columns].to_numpy()).all()
    assert ((states[score_columns] >= 0) & (states[score_columns] <= 100)).all().all()
    assert states["target_date"].min() > _states()["origin_date"].min()


def test_allocation_laboratory_compares_all_policies() -> None:
    result = run_adaptive_intelligence_allocation_laboratory(
        _states(),
        _selections(),
        ks=(3, 5, 7),
        baseline_k=5,
        minimum_history=8,
        calibration_history=4,
    )
    assert not result.recommendations.empty
    assert not result.allocation_by_date.empty
    assert not result.policy_comparison.empty
    assert set(result.policy_comparison["policy"]) == set(ALLOCATION_POLICIES)
    assert result.recommendations["memory_weight"].between(0.0, 1.0).all()
    assert result.recommendations["recommended_champion"].notna().all()
    assert result.policy_comparison["allocation_policy_score"].notna().all()

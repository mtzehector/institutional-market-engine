from __future__ import annotations

import pandas as pd

from market_engine.evaluation.survival_utility import (
    STATE_UTILITY,
    build_survival_utility_history,
    run_survival_utility_laboratory,
)


def _selections(days: int = 24, tickers: int = 16) -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=days, freq="B")
    champions = ["CHAMPION_A", "CHAMPION_B", "UNIVERSE"]
    rows: list[dict[str, object]] = []
    for day_index, date in enumerate(dates):
        for champion in champions:
            for ticker_index in range(tickers):
                rare = ticker_index % 4 == 0
                actual = "GAP_UP" if rare else "SIN_GAP"
                if champion == "CHAMPION_A":
                    good = day_index < 8 or day_index >= 17
                elif champion == "CHAMPION_B":
                    good = 6 <= day_index < 18
                else:
                    good = ticker_index % 2 == 0
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
                        "probability_up": 0.78 if predicted == "GAP_UP" else 0.08,
                        "probability_down": 0.04,
                        "probability_no_gap": 0.18 if predicted == "GAP_UP" else 0.88,
                    }
                )
    return pd.DataFrame(rows)


def _states(days: int = 24) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "origin_date": pd.date_range("2026-01-02", periods=days, freq="B"),
            "regime": ["RISK_ON_BROAD"] * 12 + ["ROTATION_VOLATILE"] * 12,
        }
    )


def test_utility_calibration_produces_bounded_authorities() -> None:
    result = run_survival_utility_laboratory(
        _selections(),
        _states(),
        horizons=(1, 3, 5),
        minimum_completed_spells=1,
    )
    assert not result.current_utility.empty
    for column in [
        "short_term_authority",
        "medium_term_authority",
        "utility_adjusted_authority",
    ]:
        assert result.current_utility[column].between(0.0, 100.0).all()
    assert result.current_utility["evidence_strength"].isin(["LOW", "MEDIUM", "HIGH"]).all()
    assert not result.summary.empty


def test_adverse_persistence_does_not_become_high_utility() -> None:
    history = pd.DataFrame(
        [
            {
                "origin_date": pd.Timestamp("2026-07-10"),
                "champion": "BAD",
                "lifecycle_state": "DETERIORATING",
                "state_age": 1,
                "completed_spells_available": 10,
                "survival_reference_scope": "CHAMPION",
                "lifecycle_health_score": 20.0,
                "performance_level": "WEAK",
                "performance_direction": "DECLINING",
                "short_advantage": -5.0,
                "advantage_slope": -2.0,
                "survival_probability_1": 0.90,
                "survival_probability_3": 0.90,
                "survival_probability_5": 0.90,
                "at_risk_support_1": 10,
                "at_risk_support_3": 10,
                "at_risk_support_5": 10,
                "survival_confidence_score": 80.0,
            }
        ]
    )
    calibrated = build_survival_utility_history(history, horizons=(1, 3, 5))
    row = calibrated.iloc[0]
    assert row["state_utility"] == STATE_UTILITY["DETERIORATING"]
    assert row["adverse_state_persistence"] > 0.70
    assert row["utility_adjusted_authority"] < 30.0

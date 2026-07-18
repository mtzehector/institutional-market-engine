from __future__ import annotations

import pandas as pd

from market_engine.evaluation.champion_stress import run_champion_stress_laboratory


def _selections(days: int = 28, tickers: int = 16) -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=days, freq="B")
    champions = ["CHAMPION_A", "CHAMPION_B", "UNIVERSE"]
    rows: list[dict[str, object]] = []
    for day_index, date in enumerate(dates):
        for champion in champions:
            for ticker_index in range(tickers):
                rare = ticker_index % 4 == 0
                actual = "GAP_UP" if rare else "SIN_GAP"
                if champion == "CHAMPION_A":
                    good = day_index < 8 or day_index >= 19
                elif champion == "CHAMPION_B":
                    good = 7 <= day_index < 22
                else:
                    good = ticker_index % 2 == 0
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
                    "probability_up": 0.78 if predicted == "GAP_UP" else 0.08,
                    "probability_down": 0.04,
                    "probability_no_gap": 0.18 if predicted == "GAP_UP" else 0.88,
                })
    return pd.DataFrame(rows)


def _states(days: int = 28) -> pd.DataFrame:
    return pd.DataFrame({
        "origin_date": pd.date_range("2026-01-02", periods=days, freq="B"),
        "regime": ["RISK_ON_BROAD"] * 14 + ["ROTATION_VOLATILE"] * 14,
    })


def test_stress_engine_builds_episode_metrics() -> None:
    result = run_champion_stress_laboratory(
        _selections(),
        _states(),
        minimum_own_episodes=1,
        minimum_family_episodes=1,
        trigger_lookback=2,
        relapse_horizon=3,
    )
    assert not result.stress_episodes.empty
    required = {
        "deterioration_velocity",
        "maximum_damage",
        "time_to_bottom",
        "recovery_efficiency",
        "relapse_within_horizon",
        "stress_score",
    }
    assert required.issubset(result.stress_episodes.columns)
    assert result.stress_episodes["stress_score"].between(0, 100).all()


def test_stress_engine_builds_current_and_group_reports() -> None:
    result = run_champion_stress_laboratory(
        _selections(),
        _states(),
        minimum_own_episodes=1,
        minimum_family_episodes=1,
    )
    assert not result.current_stress.empty
    assert result.current_stress["champion"].is_unique
    assert result.current_stress["stress_risk_score"].between(0, 100).all()
    assert not result.champion_summary.empty
    assert not result.family_summary.empty
    assert not result.summary.empty

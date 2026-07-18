from __future__ import annotations

import pandas as pd

from market_engine.evaluation.champion_lifecycle import (
    LIFECYCLE_STATES,
    build_champion_lifecycle_history,
    run_champion_lifecycle_laboratory,
)


def _selections(days: int = 12, tickers: int = 16) -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=days, freq="B")
    champions = ["CHAMPION_A", "CHAMPION_B", "UNIVERSE"]
    rows: list[dict[str, object]] = []
    for day_index, date in enumerate(dates):
        for champion in champions:
            for ticker_index in range(tickers):
                rare = ticker_index % 4 == 0
                actual = "GAP_UP" if rare else "SIN_GAP"
                if champion == "CHAMPION_A":
                    good = day_index < 7
                elif champion == "CHAMPION_B":
                    good = day_index >= 5
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


def _states(days: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "origin_date": pd.date_range("2026-01-02", periods=days, freq="B"),
            "regime": ["RISK_ON_BROAD"] * 6 + ["ROTATION_VOLATILE"] * 6,
        }
    )


def test_lifecycle_history_is_causal_and_classified() -> None:
    history = build_champion_lifecycle_history(
        _selections(), short_window=3, long_window=5, minimum_history=4
    )
    assert not history.empty
    assert set(history["lifecycle_state"]).issubset(set(LIFECYCLE_STATES))
    first = history.sort_values("origin_date").groupby("champion").head(1)
    assert (first["lifecycle_state"] == "DISCOVERY").all()
    assert history["lifecycle_health_score"].between(0, 100).all()


def test_lifecycle_laboratory_builds_current_status_and_regime_report() -> None:
    result = run_champion_lifecycle_laboratory(
        _selections(), _states(), short_window=3, long_window=5, minimum_history=4
    )
    assert not result.current_status.empty
    assert result.current_status["champion"].is_unique
    assert result.current_status["deployment_score"].between(0, 100).all()
    assert result.current_status["recommended_action"].notna().all()
    assert not result.regime_performance.empty
    assert not result.summary.empty

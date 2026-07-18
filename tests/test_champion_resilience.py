from __future__ import annotations

import pandas as pd

from market_engine.evaluation.champion_resilience import run_champion_resilience_laboratory


def _selections(days: int = 30, tickers: int = 16) -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=days, freq="B")
    champions = ["CONTROL_MOMENTUM", "INSTITUTIONAL_STABLE", "UNIVERSE"]
    rows: list[dict[str, object]] = []
    for day_index, date in enumerate(dates):
        for champion in champions:
            for ticker_index in range(tickers):
                rare = ticker_index % 4 == 0
                actual = "GAP_UP" if rare else "SIN_GAP"
                if champion == "CONTROL_MOMENTUM":
                    good = day_index < 7 or 14 <= day_index < 21 or day_index >= 26
                elif champion == "INSTITUTIONAL_STABLE":
                    good = day_index < 10 or day_index >= 17
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


def _states(days: int = 30) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "origin_date": pd.date_range("2026-01-02", periods=days, freq="B"),
            "regime": ["RISK_ON_BROAD"] * 15 + ["ROTATION_VOLATILE"] * 15,
        }
    )


def test_resilience_engine_builds_recovery_episodes_and_scores() -> None:
    result = run_champion_resilience_laboratory(
        _selections(),
        _states(),
        short_window=3,
        long_window=6,
        minimum_history=4,
        minimum_state_persistence=2,
        minimum_own_episodes=1,
        minimum_family_episodes=1,
    )
    assert not result.current_resilience.empty
    assert not result.resilience_history.empty
    assert result.current_resilience["champion"].is_unique
    assert result.current_resilience["resilience_score"].between(0.0, 100.0).all()
    assert result.current_resilience["recovery_probability"].between(0.0, 1.0).all()
    assert result.current_resilience["full_recovery_probability"].between(0.0, 1.0).all()
    assert not result.summary.empty


def test_resilience_history_is_causal_and_evidence_is_explicit() -> None:
    result = run_champion_resilience_laboratory(
        _selections(),
        _states(),
        minimum_own_episodes=2,
        minimum_family_episodes=2,
    )
    history = result.resilience_history.sort_values("origin_date")
    first = history.groupby("champion").head(1)
    assert (first["completed_recovery_episodes"] == 0).all()
    assert set(result.current_resilience["evidence_strength"]).issubset({"LOW", "MEDIUM", "HIGH"})
    assert set(result.current_resilience["resilience_reference_scope"]).issubset(
        {"CHAMPION", "FAMILY", "POOLED"}
    )

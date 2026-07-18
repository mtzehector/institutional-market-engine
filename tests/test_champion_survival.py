from __future__ import annotations

import pandas as pd

from market_engine.evaluation.champion_survival import run_champion_survival_laboratory


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


def test_survival_engine_builds_bounded_probabilities() -> None:
    result = run_champion_survival_laboratory(
        _selections(),
        _states(),
        short_window=3,
        long_window=6,
        minimum_history=4,
        minimum_state_persistence=2,
        horizons=(1, 3, 5),
        minimum_completed_spells=1,
    )
    assert not result.survival_history.empty
    assert not result.current_survival.empty
    for column in ["survival_probability_1", "survival_probability_3", "survival_probability_5"]:
        assert result.survival_history[column].between(0.0, 1.0).all()
    assert result.current_survival["champion"].is_unique
    assert result.current_survival["survival_adjusted_authority"].between(0.0, 100.0).all()


def test_survival_probabilities_do_not_increase_with_horizon() -> None:
    result = run_champion_survival_laboratory(
        _selections(),
        _states(),
        horizons=(1, 3, 5),
        minimum_completed_spells=1,
    )
    history = result.survival_history
    assert (history["survival_probability_1"] >= history["survival_probability_3"]).all()
    assert (history["survival_probability_3"] >= history["survival_probability_5"]).all()
    assert not result.state_duration_spells.empty
    assert not result.summary.empty

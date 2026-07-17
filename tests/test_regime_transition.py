from __future__ import annotations

import numpy as np
import pandas as pd

from market_engine.evaluation.regime_transition import (
    build_daily_market_states,
    build_transitions,
    run_regime_transition_laboratory,
)


def _frame(days: int = 30, tickers: int = 24) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2026-01-02", periods=days, freq="B")
    for day_index, day in enumerate(dates):
        stressed = day_index >= days // 2
        for ticker_index in range(tickers):
            strength = ticker_index / max(tickers - 1, 1)
            rare = (day_index + ticker_index) % 7 == 0
            actual = "GAP_DOWN" if stressed and rare else ("GAP_UP" if rare else "SIN_GAP")
            predicted = actual if ticker_index >= tickers // 3 else "SIN_GAP"
            rows.append(
                {
                    "ticker": f"T{ticker_index:02d}",
                    "origin_date": day,
                    "target_date": day + pd.offsets.BDay(1),
                    "return_1d_pct": (-1.2 + strength) if stressed else (0.4 + strength),
                    "return_3d_pct": (-2.0 + strength) if stressed else (0.8 + 2 * strength),
                    "return_5d_pct": (-3.0 + strength) if stressed else (1.2 + 3 * strength),
                    "close_position_pct": 30 + 50 * strength,
                    "close_vs_vwap_pct": -1 + 2 * strength,
                    "relative_volume": (1.4 if stressed else 1.0) + 0.2 * strength,
                    "smart_money_pct": 45 + 12 * strength,
                    "smart_money_slope_5d": -1 + 2 * strength,
                    "qqq_return_1d_pct": -1.0 if stressed else 0.8,
                    "atr_pct": (3.0 if stressed else 1.0) + strength,
                    "actual_gap_pct": -1.5 if actual == "GAP_DOWN" else (1.5 if actual == "GAP_UP" else 0.1),
                    "actual_gap_up": int(actual == "GAP_UP"),
                    "actual_gap_down": int(actual == "GAP_DOWN"),
                    "actual_direction": actual,
                    "probability_up": 0.72 if predicted == "GAP_UP" else 0.10,
                    "probability_down": 0.72 if predicted == "GAP_DOWN" else 0.10,
                    "probability_no_gap": 0.18 if predicted != "SIN_GAP" else 0.80,
                    "predicted_direction": predicted,
                    "correct_direction": predicted == actual,
                }
            )
    return pd.DataFrame(rows)


def test_daily_states_detect_more_than_one_regime() -> None:
    states = build_daily_market_states(_frame(), lookback=8)
    assert states["regime"].nunique() >= 2
    assert states["is_transition"].any()
    assert np.isfinite(states["transition_strength"]).all()


def test_transitions_have_from_and_to_regime() -> None:
    states = build_daily_market_states(_frame(), lookback=8)
    transitions = build_transitions(states)
    assert not transitions.empty
    assert transitions["from_regime"].notna().all()
    assert transitions["to_regime"].notna().all()


def test_regime_laboratory_builds_rankings_and_impact() -> None:
    result = run_regime_transition_laboratory(
        _frame(),
        lookback=8,
        minimum_observations=8,
        transition_radius=2,
    )
    assert not result.daily_states.empty
    assert not result.transitions.empty
    assert not result.regime_ranking.empty
    assert set(result.regime_ranking["regime_rank"].dropna().astype(int))
    assert not result.transition_impact.empty

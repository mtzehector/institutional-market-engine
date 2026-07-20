from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_engine.evaluation.historical_outcome_distributions import (
    _opportunity_index,
    _stability_score,
    _tail_risk_index,
    _validate_thresholds,
    run_historical_outcome_distribution_laboratory,
)


def _frame(values: list[float], ticker: str) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=len(values), freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ticker,
            "market_cap": values,
        }
    )


def test_stability_score_rewards_lower_dispersion() -> None:
    stable = pd.Series([0.02, 0.021, 0.019, 0.022, 0.020])
    volatile = pd.Series([0.20, -0.18, 0.15, -0.12, 0.10])
    assert _stability_score(stable) > _stability_score(volatile)


def test_tail_risk_increases_with_negative_tail() -> None:
    mild_returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.00])
    mild_drawdowns = pd.Series([-0.01, -0.02, -0.01, -0.03, 0.00])
    severe_returns = pd.Series([0.01, 0.02, -0.25, 0.03, -0.15])
    severe_drawdowns = pd.Series([-0.01, -0.02, -0.30, -0.03, -0.20])
    assert _tail_risk_index(severe_returns, severe_drawdowns) > _tail_risk_index(
        mild_returns, mild_drawdowns
    )


def test_opportunity_index_rewards_large_positive_outcomes() -> None:
    modest = pd.Series([0.01, 0.02, -0.01, 0.03, 0.00])
    strong = pd.Series([0.06, 0.08, 0.12, 0.15, 0.04])
    assert _opportunity_index(strong) > _opportunity_index(modest)


def test_distribution_percentiles_are_ordered_and_probabilities_bounded() -> None:
    sample = _frame([100, 102, 105, 103, 108, 111, 109, 115, 118, 120, 123, 125], "S")
    reference = _frame([100, 101, 103, 102, 105, 107, 106, 109, 111, 112, 114, 116], "R")
    result = run_historical_outcome_distribution_laboratory(
        sample,
        reference,
        horizons=(1, 3),
        smoothing_window=2,
        spread_window=3,
        minimum_episode_observations=1,
    )
    assert not result.distributions.empty
    assert not result.probabilities.empty
    assert not result.ranking.empty
    assert (result.distributions["p05"] <= result.distributions["p10"]).all()
    assert (result.distributions["p10"] <= result.distributions["p25"]).all()
    assert (result.distributions["p25"] <= result.distributions["p50"]).all()
    assert (result.distributions["p50"] <= result.distributions["p75"]).all()
    probability_columns = [
        column for column in result.probabilities.columns if column.startswith("probability_")
    ]
    assert result.probabilities[probability_columns].apply(
        lambda column: column.between(0.0, 1.0).all()
    ).all()


def test_scores_are_finite_and_bounded() -> None:
    sample = _frame([100, 101, 102, 103, 105, 104, 107, 109, 110, 112], "S")
    reference = _frame([100, 100.5, 101, 102, 103, 102.5, 104, 105, 106, 107], "R")
    result = run_historical_outcome_distribution_laboratory(
        sample,
        reference,
        horizons=(1, 2),
        smoothing_window=2,
        spread_window=3,
        minimum_episode_observations=1,
    )
    for column in (
        "outcome_stability_score",
        "tail_risk_index",
        "opportunity_index",
        "balanced_outcome_score",
    ):
        assert np.isfinite(result.ranking[column]).all()
        assert result.ranking[column].between(0.0, 100.0).all()


def test_invalid_thresholds_are_rejected() -> None:
    with pytest.raises(ValueError):
        _validate_thresholds([], name="thresholds")
    with pytest.raises(ValueError):
        _validate_thresholds([0.0, 0.05], name="thresholds")

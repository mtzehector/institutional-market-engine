from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_engine.evaluation.distribution_reliability_calibration import (
    _calibrated_stability,
    _effective_sample_size,
    _non_overlapping_count,
    _wilson_interval,
    run_distribution_reliability_calibration_laboratory,
)


def _market(values: list[float], ticker: str) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=len(values), freq="D")
    return pd.DataFrame(
        {"date": dates, "ticker": ticker, "market_cap": values}
    )


def test_drawdowns_are_never_positive_after_calibration() -> None:
    sample = _market([100 + i * 2 for i in range(50)], "S")
    reference = _market([100 + i for i in range(50)], "R")
    result = run_distribution_reliability_calibration_laboratory(
        sample,
        reference,
        horizons=(1, 5),
        spread_window=5,
    )
    assert (result.calibrated_drawdowns["mean_future_drawdown"] <= 0).all()
    assert (result.calibrated_drawdowns["worst_future_drawdown"] <= 0).all()


def test_effective_sample_size_drops_for_autocorrelated_values() -> None:
    independent = pd.Series([1.0, -1.0] * 25)
    autocorrelated = pd.Series(np.linspace(0.0, 1.0, 50))
    assert _effective_sample_size(autocorrelated, 10) < _effective_sample_size(independent, 10)


def test_non_overlapping_count_respects_horizon() -> None:
    positions = np.array([0, 1, 2, 5, 6, 10])
    assert _non_overlapping_count(positions, 5) == 3


def test_wilson_interval_is_bounded_and_contains_rate() -> None:
    lower, upper = _wilson_interval(7, 10)
    assert 0.0 <= lower <= 0.7 <= upper <= 1.0


def test_calibrated_stability_distinguishes_dispersion() -> None:
    stable = pd.Series([0.01, 0.011, 0.009, 0.0105, 0.0102])
    volatile = pd.Series([0.20, -0.15, 0.18, -0.12, 0.10])
    assert _calibrated_stability(stable) > _calibrated_stability(volatile)


def test_small_samples_are_shrunk_toward_neutral() -> None:
    sample = _market([100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122], "S")
    reference = _market([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111], "R")
    result = run_distribution_reliability_calibration_laboratory(
        sample,
        reference,
        horizons=(5,),
        spread_window=3,
        confidence_prior=30.0,
    )
    ranking = result.calibrated_ranking
    assert not ranking.empty
    assert ranking["confidence_weight"].between(0.0, 1.0).all()
    distance_calibrated = (ranking["calibrated_balanced_score"] - 50.0).abs()
    distance_raw = (ranking["raw_balanced_score"] - 50.0).abs()
    assert (distance_calibrated <= distance_raw + 1e-12).all()


def test_invalid_confidence_prior_is_rejected() -> None:
    sample = _market([100, 101, 102, 103, 104], "S")
    reference = _market([100, 101, 102, 103, 104], "R")
    with pytest.raises(ValueError, match="confidence_prior"):
        run_distribution_reliability_calibration_laboratory(
            sample,
            reference,
            horizons=(1,),
            confidence_prior=0,
        )

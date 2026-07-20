from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_engine.evaluation.divergence_outcomes import run_divergence_outcome_laboratory


def _market_caps(sample_totals: list[float], reference_totals: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2025-01-02", periods=len(sample_totals), freq="B")
    sample_rows: list[dict[str, object]] = []
    reference_rows: list[dict[str, object]] = []
    for date, sample_total, reference_total in zip(dates, sample_totals, reference_totals):
        sample_rows.extend(
            [
                {"date": date, "ticker": "S1", "market_cap": sample_total * 0.6},
                {"date": date, "ticker": "S2", "market_cap": sample_total * 0.4},
            ]
        )
        reference_rows.extend(
            [
                {"date": date, "ticker": "R1", "market_cap": reference_total * 0.5},
                {"date": date, "ticker": "R2", "market_cap": reference_total * 0.3},
                {"date": date, "ticker": "R3", "market_cap": reference_total * 0.2},
            ]
        )
    return pd.DataFrame(sample_rows), pd.DataFrame(reference_rows)


def test_forward_returns_use_only_future_observations() -> None:
    sample, reference = _market_caps([100, 110, 121, 133.1, 146.41], [100, 105, 110.25, 115.7625, 121.550625])
    result = run_divergence_outcome_laboratory(
        sample,
        reference,
        horizons=(1,),
        smoothing_window=2,
        spread_window=3,
        minimum_episode_observations=1,
    )
    daily = result.daily_outcomes
    assert daily.loc[0, "sample_return_forward_1"] == pytest.approx(0.10)
    assert daily.loc[0, "reference_return_forward_1"] == pytest.approx(0.05)
    assert pd.isna(daily.iloc[-1]["sample_return_forward_1"])
    assert not bool(daily.iloc[-1]["outcome_completed_1"])


def test_future_drawdown_captures_worst_path_inside_horizon() -> None:
    sample, reference = _market_caps([100, 90, 95, 110, 115], [100, 98, 99, 101, 103])
    result = run_divergence_outcome_laboratory(
        sample,
        reference,
        horizons=(3,),
        smoothing_window=2,
        spread_window=3,
        minimum_episode_observations=1,
    )
    first = result.daily_outcomes.iloc[0]
    assert first["sample_future_drawdown_3"] == pytest.approx(-0.10)
    assert first["reference_future_drawdown_3"] == pytest.approx(-0.02)


def test_state_summary_excludes_censored_rows() -> None:
    sample, reference = _market_caps(
        [100, 102, 105, 109, 112, 115, 117, 116, 118, 120],
        [100, 101, 102, 104, 106, 108, 110, 111, 112, 113],
    )
    result = run_divergence_outcome_laboratory(
        sample,
        reference,
        horizons=(5,),
        smoothing_window=2,
        spread_window=3,
        minimum_episode_observations=1,
    )
    assert int(result.state_outcomes["observations"].sum()) == 5


def test_transition_probabilities_sum_to_one_by_origin_state() -> None:
    sample, reference = _market_caps(
        [100, 103, 106, 104, 101, 99, 102, 108, 112, 110, 115, 118],
        [100, 101, 103, 105, 104, 102, 103, 105, 107, 109, 110, 112],
    )
    result = run_divergence_outcome_laboratory(
        sample,
        reference,
        horizons=(1, 3),
        smoothing_window=2,
        spread_window=3,
        minimum_episode_observations=1,
    )
    grouped = result.transition_matrix.groupby(["horizon", "from_state"])[
        "transition_probability"
    ].sum()
    assert np.allclose(grouped.to_numpy(), 1.0)


def test_episode_outcomes_preserve_censoring() -> None:
    sample, reference = _market_caps(
        [100, 105, 110, 100, 95, 90, 92, 94, 93, 91],
        [100, 102, 104, 103, 102, 101, 102, 103, 104, 105],
    )
    result = run_divergence_outcome_laboratory(
        sample,
        reference,
        horizons=(1, 5),
        smoothing_window=2,
        spread_window=3,
        minimum_episode_observations=1,
    )
    if not result.episode_outcomes.empty:
        assert "outcome_completed_5" in result.episode_outcomes.columns
        assert result.episode_outcomes["outcome_completed_5"].isin([True, False]).all()


def test_invalid_horizons_are_rejected() -> None:
    sample, reference = _market_caps([100, 101, 102], [100, 101, 102])
    with pytest.raises(ValueError, match="horizons"):
        run_divergence_outcome_laboratory(sample, reference, horizons=(0, 5))

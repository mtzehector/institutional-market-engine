from __future__ import annotations

import pandas as pd
import pytest

from market_engine.evaluation.sample_morphology_fidelity import compare_sample_to_reference


def _frame(values: list[float], tickers: tuple[str, ...] = ("AAA", "BBB")) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=len(values), freq="D")
    rows: list[dict[str, object]] = []
    weights = [0.6, 0.4] if len(tickers) == 2 else [1.0 / len(tickers)] * len(tickers)
    for date, total in zip(dates, values, strict=True):
        for ticker, weight in zip(tickers, weights, strict=True):
            rows.append({"date": date, "ticker": ticker, "market_cap": total * weight})
    return pd.DataFrame(rows)


def test_identical_sample_has_perfect_fidelity() -> None:
    frame = _frame([100, 105, 110, 108, 115, 120, 118, 125])
    result = compare_sample_to_reference(frame, frame, smoothing_window=2, rolling_window=3)
    summary = result.summary.iloc[0]

    assert summary["index_correlation"] == pytest.approx(1.0)
    assert summary["returns_correlation"] == pytest.approx(1.0)
    assert summary["drawdown_correlation"] == pytest.approx(1.0)
    assert summary["mean_absolute_index_error"] == pytest.approx(0.0)
    assert summary["morphology_fidelity_score"] == pytest.approx(100.0)


def test_scaled_sample_preserves_geometry() -> None:
    reference = _frame([100, 110, 125, 100, 90, 115, 130])
    sample = reference.copy()
    sample["market_cap"] *= 0.15

    result = compare_sample_to_reference(sample, reference, smoothing_window=2, rolling_window=3)
    summary = result.summary.iloc[0]

    assert summary["index_correlation"] == pytest.approx(1.0)
    assert summary["mean_absolute_index_error"] == pytest.approx(0.0)
    assert summary["morphology_fidelity_score"] == pytest.approx(100.0)


def test_distorted_sample_loses_fidelity() -> None:
    reference = _frame([100, 110, 125, 100, 90, 115, 130, 140])
    sample = _frame([100, 98, 95, 110, 120, 105, 90, 85])

    result = compare_sample_to_reference(sample, reference, smoothing_window=2, rolling_window=3)
    summary = result.summary.iloc[0]

    assert summary["morphology_fidelity_score"] < 60
    assert summary["directional_agreement"] < 0.6
    assert summary["mean_absolute_index_error"] > 10


def test_comparison_requires_common_dates() -> None:
    sample = _frame([100, 101, 102])
    reference = _frame([100, 101, 102])
    reference["date"] = pd.to_datetime(reference["date"]) + pd.Timedelta(days=30)

    with pytest.raises(ValueError, match="fechas comunes"):
        compare_sample_to_reference(sample, reference)


def test_rolling_window_must_be_valid() -> None:
    frame = _frame([100, 101, 102])
    with pytest.raises(ValueError, match="rolling_window"):
        compare_sample_to_reference(frame, frame, rolling_window=2)

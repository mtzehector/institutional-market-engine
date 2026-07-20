from __future__ import annotations

import pandas as pd

from market_engine.evaluation.sentinel_divergence import run_sentinel_divergence_laboratory


def _frame(values: list[float], ticker: str = "AAA") -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=len(values), freq="D")
    return pd.DataFrame({"date": dates, "ticker": ticker, "market_cap": values})


def test_identical_series_keep_zero_spread() -> None:
    frame = _frame([100, 101, 102, 103, 104, 105])
    result = run_sentinel_divergence_laboratory(frame, frame, spread_window=3)
    assert result.daily["leadership_spread"].abs().max() == 0
    assert set(result.daily["divergence_state"]) == {"BROAD_ALIGNMENT"}


def test_faster_sample_creates_positive_leadership() -> None:
    sample = _frame([100, 103, 107, 112, 118, 125])
    reference = _frame([100, 101, 103, 105, 108, 111])
    result = run_sentinel_divergence_laboratory(sample, reference, spread_window=3)
    assert result.daily.iloc[-1]["leadership_spread"] > 0
    assert (result.daily["divergence_state"] == "SENTINEL_LEADERSHIP").any()


def test_opposite_directions_create_directional_break() -> None:
    sample = _frame([100, 102, 104, 102, 101, 100])
    reference = _frame([100, 101, 102, 103, 104, 105])
    result = run_sentinel_divergence_laboratory(sample, reference, spread_window=3)
    assert (result.daily["divergence_state"] == "DIRECTIONAL_BREAK").any()


def test_stress_amplification_is_detected() -> None:
    sample = _frame([100, 105, 100, 85, 80, 82])
    reference = _frame([100, 104, 102, 95, 92, 94])
    result = run_sentinel_divergence_laboratory(sample, reference, spread_window=3)
    assert (result.daily["divergence_state"] == "STRESS_AMPLIFICATION").any()


def test_invalid_parameters_are_rejected() -> None:
    frame = _frame([100, 101, 102])
    try:
        run_sentinel_divergence_laboratory(frame, frame, spread_window=2)
    except ValueError as exc:
        assert "spread_window" in str(exc)
    else:
        raise AssertionError("Se esperaba ValueError")

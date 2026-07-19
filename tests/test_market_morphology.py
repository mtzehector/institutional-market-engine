from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_engine.evaluation.market_morphology import (
    build_market_geometry,
    detect_drawdown_episodes,
    run_market_morphology_laboratory,
)


def _sample_frame() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=8, freq="D")
    totals = [100.0, 110.0, 120.0, 105.0, 90.0, 100.0, 115.0, 121.0]
    rows: list[dict[str, object]] = []
    for date, total in zip(dates, totals, strict=True):
        rows.append({"date": date, "ticker": "AAA", "market_cap": total * 0.6})
        rows.append({"date": date, "ticker": "BBB", "market_cap": total * 0.4})
    return pd.DataFrame(rows)


def test_market_geometry_aggregates_and_normalizes() -> None:
    geometry = build_market_geometry(_sample_frame(), smoothing_window=2)

    assert len(geometry) == 8
    assert geometry.iloc[0]["aggregate_market_cap"] == pytest.approx(100.0)
    assert geometry.iloc[0]["market_cap_index"] == pytest.approx(100.0)
    assert geometry.iloc[2]["aggregate_market_cap"] == pytest.approx(120.0)
    assert geometry.iloc[4]["drawdown_pct"] == pytest.approx(-25.0)
    assert geometry["observed_companies"].eq(2).all()


def test_detect_drawdown_episode_recovers_at_new_peak() -> None:
    geometry = build_market_geometry(_sample_frame(), smoothing_window=2)
    episodes = detect_drawdown_episodes(geometry, minimum_drawdown_pct=10.0)

    assert len(episodes) == 1
    episode = episodes.iloc[0]
    assert episode["episode_id"] == "EPISODE_001"
    assert episode["maximum_drawdown_pct"] == pytest.approx(-25.0)
    assert bool(episode["episode_completed"])
    assert episode["recovery_date"] == pd.Timestamp("2026-01-08")


def test_active_episode_is_not_misclassified_as_recovered() -> None:
    frame = _sample_frame()
    frame = frame.loc[frame["date"] <= pd.Timestamp("2026-01-05")]
    result = run_market_morphology_laboratory(
        frame,
        smoothing_window=2,
        minimum_drawdown_pct=10.0,
    )

    assert len(result.episodes) == 1
    assert not bool(result.episodes.iloc[0]["episode_completed"])
    assert bool(result.summary.iloc[0]["active_episode"])
    assert pd.isna(result.episodes.iloc[0]["recovery_date"])


def test_geometry_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="Faltan columnas requeridas"):
        build_market_geometry(pd.DataFrame({"date": ["2026-01-01"]}))


def test_geometry_rejects_invalid_window() -> None:
    with pytest.raises(ValueError, match="smoothing_window"):
        build_market_geometry(_sample_frame(), smoothing_window=0)


def test_acceleration_is_derived_from_smoothed_slope() -> None:
    geometry = build_market_geometry(_sample_frame(), smoothing_window=2)
    expected = geometry["smoothed_slope"].diff()
    np.testing.assert_allclose(
        geometry["acceleration"].fillna(0.0),
        expected.fillna(0.0),
    )

from __future__ import annotations

from datetime import date
import json

import pandas as pd

from market_engine.evaluation.research_snapshot import (
    compare_snapshots,
    create_snapshot,
    export_snapshot,
)


def _ranking(score: float = 70.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "TEAM",
                "predictability_score": score,
                "predictability_grade": "B_CONFIABLE",
                "observations": 100,
                "directional_accuracy": 0.68,
                "confidence_weighted_accuracy": 0.70,
                "high_confidence_accuracy": 0.72,
                "high_confidence_observations": 40,
                "mean_brier_skill": 0.12,
                "mean_calibration_error": 0.08,
                "memory_rows": 252,
                "memory_label": "252_ROWS",
            }
        ]
    )


def test_snapshot_exports_json_excel_and_markdown(tmp_path) -> None:
    source = tmp_path / "ranking.csv"
    _ranking().to_csv(source, index=False)

    snapshot = create_snapshot(
        [source],
        label="BASELINE_TEST",
        as_of_date=date(2026, 7, 16),
        engine_version="0.7.0",
        regime_label="REGIME_2026",
    )
    paths = export_snapshot(snapshot, tmp_path / "snapshot")

    assert all(path.exists() for path in paths.values())
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["metadata"]["snapshot_label"] == "BASELINE_TEST"
    assert payload["ticker_metrics"][0]["ticker"] == "TEAM"


def test_snapshot_comparison_reports_change_and_new_metric(tmp_path) -> None:
    baseline_source = tmp_path / "baseline.csv"
    current_source = tmp_path / "current.csv"
    baseline = _ranking(60.0)
    current = _ranking(75.0)
    current["average_absolute_gap_pct"] = 1.8
    baseline.to_csv(baseline_source, index=False)
    current.to_csv(current_source, index=False)

    baseline_snapshot = create_snapshot(
        [baseline_source], "BASE", date(2026, 7, 16), "0.7.0"
    )
    current_snapshot = create_snapshot(
        [current_source], "CURRENT", date(2026, 8, 16), "0.8.0"
    )

    from market_engine.evaluation.research_snapshot import snapshot_to_dict

    comparison = compare_snapshots(
        snapshot_to_dict(baseline_snapshot), snapshot_to_dict(current_snapshot)
    )

    score_row = comparison[comparison["metric"] == "predictability_score.mean"].iloc[0]
    assert score_row["absolute_change"] == 15.0
    assert "catalog.new.average_absolute_gap_pct" in set(comparison["metric"])

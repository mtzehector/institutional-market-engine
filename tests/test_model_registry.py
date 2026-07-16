from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from market_engine.evaluation.model_registry import (
    build_leaderboard,
    excel_safe_dataframe,
    export_registry_workbook,
    record_evaluation,
    teacher_level,
)


def _ranking() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "TEAM",
                "predictability_score": 82.0,
                "predictability_grade": "A_EXCELENTE",
                "observations": 120,
                "directional_accuracy": 0.74,
                "confidence_weighted_accuracy": 0.76,
                "mean_brier_skill": 0.18,
                "mean_calibration_error": 0.07,
            },
            {
                "ticker": "VIRT",
                "predictability_score": 68.0,
                "predictability_grade": "B_CONFIABLE",
                "observations": 110,
                "directional_accuracy": 0.66,
                "confidence_weighted_accuracy": 0.67,
                "mean_brier_skill": 0.11,
                "mean_calibration_error": 0.10,
            },
        ]
    )


def test_teacher_level_respects_sample_size() -> None:
    assert teacher_level(90.0, 20) == "MUESTRA_INSUFICIENTE"
    assert teacher_level(90.0, 100) == "LEGENDARIO"
    assert teacher_level(68.0, 100) == "PROFESOR_A"


def test_registry_persists_and_builds_leaderboard(tmp_path) -> None:
    path = tmp_path / "registry.csv"
    result = record_evaluation(
        _ranking(),
        registry_path=path,
        model_version="0.5.0",
        run_label="pilot",
        recorded_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
    )

    assert path.exists()
    assert result.leaderboard.iloc[0]["ticker"] == "TEAM"
    assert result.leaderboard.iloc[0]["hall_of_fame_rank"] == 1


def test_leaderboard_uses_latest_record_per_ticker() -> None:
    history = pd.DataFrame(
        [
            {
                **_ranking().iloc[0].to_dict(),
                "recorded_at": "2026-07-01T00:00:00Z",
                "model_version": "0.4.0",
                "run_label": "base",
                "teacher_level": "ELITE",
            },
            {
                **_ranking().iloc[0].to_dict(),
                "predictability_score": 88.0,
                "recorded_at": "2026-07-15T00:00:00Z",
                "model_version": "0.5.0",
                "run_label": "base",
                "teacher_level": "LEGENDARIO",
            },
        ]
    )
    leaderboard = build_leaderboard(history)
    assert len(leaderboard) == 1
    assert leaderboard.iloc[0]["predictability_score"] == 88.0


def test_excel_safe_dataframe_removes_timezone() -> None:
    frame = pd.DataFrame(
        {
            "recorded_at": pd.to_datetime(
                ["2026-07-15T12:30:00Z", "2026-07-16T12:30:00Z"],
                utc=True,
            ),
            "object_datetime": [
                datetime(2026, 7, 15, 12, 30, tzinfo=timezone.utc),
                datetime(2026, 7, 16, 12, 30, tzinfo=timezone.utc),
            ],
        }
    )

    safe = excel_safe_dataframe(frame)

    assert safe["recorded_at"].dt.tz is None
    assert all(value.tzinfo is None for value in safe["object_datetime"])


def test_registry_workbook_exports_timezone_aware_dates(tmp_path) -> None:
    registry_path = tmp_path / "registry.csv"
    output = tmp_path / "hall_of_fame.xlsx"
    result = record_evaluation(
        _ranking(),
        registry_path=registry_path,
        model_version="0.5.0",
        run_label="timezone-test",
        recorded_at=datetime(2026, 7, 15, 12, 30, tzinfo=timezone.utc),
    )

    export_registry_workbook(result, output)

    assert output.exists()
    assert output.stat().st_size > 0

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "ticker",
    "predictability_score",
    "predictability_grade",
    "observations",
    "directional_accuracy",
    "confidence_weighted_accuracy",
    "mean_brier_skill",
    "mean_calibration_error",
}


@dataclass(frozen=True)
class RegistryResult:
    history: pd.DataFrame
    leaderboard: pd.DataFrame
    changes: pd.DataFrame


def teacher_level(score: float, observations: int) -> str:
    if observations < 40:
        return "MUESTRA_INSUFICIENTE"
    if score >= 85:
        return "LEGENDARIO"
    if score >= 75:
        return "ELITE"
    if score >= 65:
        return "PROFESOR_A"
    if score >= 50:
        return "PROFESOR_B"
    if score >= 35:
        return "DIFICIL"
    return "NO_MODELABLE_ACTUALMENTE"


def _validate(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el ranking: {sorted(missing)}")
    if frame.empty:
        raise ValueError("El ranking está vacío")


def load_registry(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, parse_dates=["recorded_at"], encoding="utf-8")
    return frame


def save_registry(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["recorded_at", "ticker"]).to_csv(path, index=False, encoding="utf-8")


def record_evaluation(
    ranking: pd.DataFrame,
    registry_path: Path,
    model_version: str,
    run_label: str = "default",
    recorded_at: datetime | None = None,
) -> RegistryResult:
    _validate(ranking)
    timestamp = recorded_at or datetime.now(timezone.utc)
    current = ranking.copy()
    current["ticker"] = current["ticker"].astype(str).str.upper()
    current["model_version"] = model_version
    current["run_label"] = run_label
    current["recorded_at"] = pd.Timestamp(timestamp)
    current["teacher_level"] = current.apply(
        lambda row: teacher_level(
            float(row["predictability_score"]), int(row["observations"])
        ),
        axis=1,
    )

    history = load_registry(registry_path)
    if not history.empty:
        history["recorded_at"] = pd.to_datetime(history["recorded_at"], utc=True)
    history = pd.concat([history, current], ignore_index=True, sort=False)
    history = history.drop_duplicates(
        subset=["ticker", "model_version", "run_label", "recorded_at"], keep="last"
    )
    save_registry(history, registry_path)

    leaderboard = build_leaderboard(history)
    changes = build_score_changes(history)
    return RegistryResult(history=history, leaderboard=leaderboard, changes=changes)


def build_leaderboard(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    ordered = history.copy()
    ordered["recorded_at"] = pd.to_datetime(ordered["recorded_at"], utc=True)
    latest = (
        ordered.sort_values("recorded_at")
        .groupby("ticker", as_index=False, sort=False)
        .tail(1)
        .copy()
    )
    latest["teacher_level"] = latest.apply(
        lambda row: teacher_level(
            float(row["predictability_score"]), int(row["observations"])
        ),
        axis=1,
    )
    latest = latest.sort_values(
        ["predictability_score", "observations", "directional_accuracy"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    latest.insert(0, "hall_of_fame_rank", np.arange(1, len(latest) + 1))
    return latest


def build_score_changes(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    ordered = history.copy()
    ordered["recorded_at"] = pd.to_datetime(ordered["recorded_at"], utc=True)
    ordered = ordered.sort_values(["ticker", "recorded_at"])
    ordered["previous_score"] = ordered.groupby("ticker")["predictability_score"].shift(1)
    ordered["score_change"] = ordered["predictability_score"] - ordered["previous_score"]
    ordered["trend"] = np.select(
        [ordered["score_change"] > 1.0, ordered["score_change"] < -1.0],
        ["MEJORA", "DETERIORO"],
        default="ESTABLE",
    )
    return ordered


def _excel_safe_value(value: Any) -> Any:
    """Convert timezone-aware scalar datetimes into Excel-compatible values."""
    if value is None or value is pd.NaT:
        return value

    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None:
            return value.tz_convert("UTC").tz_localize(None)
        return value

    if isinstance(value, datetime) and value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    return value


def excel_safe_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy without timezone-aware datetimes in any column.

    Pandas may store timezone-aware values either as ``datetime64[ns, tz]`` or
    inside ``object`` columns. Excel/openpyxl accepts neither, so both cases are
    normalized here before exporting.
    """
    safe = frame.copy()

    for column in safe.columns:
        series = safe[column]

        if isinstance(series.dtype, pd.DatetimeTZDtype):
            safe[column] = series.dt.tz_convert("UTC").dt.tz_localize(None)
            continue

        if series.dtype == "object":
            safe[column] = series.map(_excel_safe_value)

    return safe


def export_registry_workbook(result: RegistryResult, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        excel_safe_dataframe(result.leaderboard).to_excel(
            writer, sheet_name="Hall_of_Fame", index=False
        )
        excel_safe_dataframe(result.changes).to_excel(
            writer, sheet_name="Historial_Scores", index=False
        )
        excel_safe_dataframe(result.history).to_excel(
            writer, sheet_name="Registro_Completo", index=False
        )

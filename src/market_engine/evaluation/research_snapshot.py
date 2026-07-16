from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import json

import numpy as np
import pandas as pd


PREFERRED_SHEETS = (
    "Mejor_Memoria",
    "Ranking",
    "Hall_of_Fame",
    "Prediccion",
)

KEY_METRICS = (
    "predictability_score",
    "observations",
    "directional_accuracy",
    "confidence_weighted_accuracy",
    "high_confidence_accuracy",
    "high_confidence_observations",
    "mean_brier_skill",
    "mean_calibration_error",
    "average_absolute_gap_pct",
    "memory_rows",
)


@dataclass(frozen=True)
class ResearchSnapshot:
    metadata: pd.DataFrame
    ticker_metrics: pd.DataFrame
    aggregate_metrics: pd.DataFrame
    metric_catalog: pd.DataFrame
    source_files: pd.DataFrame


def _excel_safe(frame: pd.DataFrame) -> pd.DataFrame:
    safe = frame.copy()
    for column in safe.columns:
        series = safe[column]
        if isinstance(series.dtype, pd.DatetimeTZDtype):
            safe[column] = series.dt.tz_localize(None)
        elif series.dtype == "object":
            safe[column] = series.map(
                lambda value: value.tz_localize(None)
                if isinstance(value, pd.Timestamp) and value.tzinfo is not None
                else value.replace(tzinfo=None)
                if isinstance(value, datetime) and value.tzinfo is not None
                else value
            )
    return safe


def _read_best_sheet(path: Path, preferred_sheet: str | None = None) -> tuple[pd.DataFrame, str]:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        workbook = pd.ExcelFile(path)
        candidates = [preferred_sheet] if preferred_sheet else []
        candidates.extend(PREFERRED_SHEETS)
        for sheet in candidates:
            if sheet and sheet in workbook.sheet_names:
                return pd.read_excel(path, sheet_name=sheet), sheet
        return pd.read_excel(path, sheet_name=workbook.sheet_names[0]), workbook.sheet_names[0]
    return pd.read_csv(path), "CSV"


def _normalize_ticker_frame(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    current = frame.copy()
    current = current.dropna(how="all")
    if current.empty:
        return current

    if "ticker" not in current.columns:
        current["ticker"] = source.stem.upper()

    current["ticker"] = current["ticker"].astype(str).str.upper()
    current["source_file"] = source.name

    for column in current.columns:
        if column in {"ticker", "source_file", "predictability_grade", "memory_label", "regime_label", "selection"}:
            continue
        converted = pd.to_numeric(current[column], errors="coerce")
        if converted.notna().sum() >= max(1, int(0.7 * len(current))):
            current[column] = converted

    return current


def _aggregate(ticker_metrics: pd.DataFrame) -> pd.DataFrame:
    if ticker_metrics.empty:
        return pd.DataFrame(columns=["metric", "value"])

    rows: list[dict[str, Any]] = [
        {"metric": "ticker_count", "value": int(ticker_metrics["ticker"].nunique())},
        {"metric": "row_count", "value": int(len(ticker_metrics))},
    ]

    for metric in KEY_METRICS:
        if metric not in ticker_metrics.columns:
            continue
        numeric = pd.to_numeric(ticker_metrics[metric], errors="coerce").dropna()
        if numeric.empty:
            continue
        rows.extend(
            [
                {"metric": f"{metric}.mean", "value": float(numeric.mean())},
                {"metric": f"{metric}.median", "value": float(numeric.median())},
                {"metric": f"{metric}.min", "value": float(numeric.min())},
                {"metric": f"{metric}.max", "value": float(numeric.max())},
            ]
        )

    if "predictability_grade" in ticker_metrics.columns:
        counts = ticker_metrics["predictability_grade"].fillna("SIN_CLASIFICAR").value_counts()
        rows.extend(
            {"metric": f"grade_count.{grade}", "value": int(count)}
            for grade, count in counts.items()
        )

    if "memory_label" in ticker_metrics.columns:
        counts = ticker_metrics["memory_label"].fillna("SIN_MEMORIA").value_counts()
        rows.extend(
            {"metric": f"memory_count.{label}", "value": int(count)}
            for label, count in counts.items()
        )

    return pd.DataFrame(rows)


def _metric_catalog(ticker_metrics: pd.DataFrame) -> pd.DataFrame:
    descriptions = {
        "predictability_score": "Score compuesto de predictibilidad fuera de muestra.",
        "observations": "Cantidad de predicciones walk-forward evaluadas.",
        "directional_accuracy": "Proporción de aciertos entre GAP_UP, GAP_DOWN y SIN_GAP.",
        "confidence_weighted_accuracy": "Exactitud ponderada por la confianza emitida.",
        "high_confidence_accuracy": "Exactitud de predicciones por encima del umbral de alta confianza.",
        "high_confidence_observations": "Número de predicciones de alta confianza.",
        "mean_brier_skill": "Mejora media del Brier Score frente a la referencia base; mayor es mejor.",
        "mean_calibration_error": "Error medio entre probabilidad emitida y frecuencia observada; menor es mejor.",
        "average_absolute_gap_pct": "Magnitud absoluta media del gap observado.",
        "memory_rows": "Sesiones efectivas de entrenamiento de la memoria seleccionada.",
    }
    rows = []
    for metric in KEY_METRICS:
        rows.append(
            {
                "metric": metric,
                "present": metric in ticker_metrics.columns,
                "description": descriptions.get(metric, "Métrica registrada por el motor."),
                "direction": "LOWER_IS_BETTER" if metric == "mean_calibration_error" else "HIGHER_IS_BETTER",
            }
        )
    return pd.DataFrame(rows)


def create_snapshot(
    input_files: Iterable[Path],
    label: str,
    as_of_date: date,
    engine_version: str,
    regime_label: str = "UNSPECIFIED",
    notes: str = "",
    preferred_sheet: str | None = None,
) -> ResearchSnapshot:
    frames: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []

    for raw_path in input_files:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"No existe el archivo de entrada: {path}")
        frame, sheet = _read_best_sheet(path, preferred_sheet)
        normalized = _normalize_ticker_frame(frame, path)
        if not normalized.empty:
            frames.append(normalized)
        sources.append(
            {
                "source_file": path.name,
                "source_path": str(path),
                "sheet": sheet,
                "rows_loaded": int(len(normalized)),
            }
        )

    if not frames:
        raise ValueError("Ningún archivo produjo métricas válidas")

    ticker_metrics = pd.concat(frames, ignore_index=True, sort=False)
    if "predictability_score" in ticker_metrics.columns:
        ticker_metrics = ticker_metrics.sort_values(
            ["predictability_score", "ticker"], ascending=[False, True]
        ).reset_index(drop=True)

    now = datetime.now(timezone.utc)
    metadata = pd.DataFrame(
        [
            {"field": "snapshot_label", "value": label},
            {"field": "as_of_date", "value": as_of_date.isoformat()},
            {"field": "created_at_utc", "value": now.isoformat()},
            {"field": "engine_version", "value": engine_version},
            {"field": "regime_label", "value": regime_label},
            {"field": "scope", "value": "PARTIAL" if ticker_metrics["ticker"].nunique() < 10 else "MULTI_TICKER"},
            {"field": "notes", "value": notes},
        ]
    )

    return ResearchSnapshot(
        metadata=metadata,
        ticker_metrics=ticker_metrics,
        aggregate_metrics=_aggregate(ticker_metrics),
        metric_catalog=_metric_catalog(ticker_metrics),
        source_files=pd.DataFrame(sources),
    )


def snapshot_to_dict(snapshot: ResearchSnapshot) -> dict[str, Any]:
    metadata = dict(zip(snapshot.metadata["field"], snapshot.metadata["value"]))
    return {
        "metadata": metadata,
        "aggregate_metrics": snapshot.aggregate_metrics.to_dict(orient="records"),
        "metric_catalog": snapshot.metric_catalog.to_dict(orient="records"),
        "source_files": snapshot.source_files.to_dict(orient="records"),
        "ticker_metrics": snapshot.ticker_metrics.replace({np.nan: None}).to_dict(orient="records"),
    }


def export_snapshot(snapshot: ResearchSnapshot, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(zip(snapshot.metadata["field"], snapshot.metadata["value"]))
    safe_label = str(metadata["snapshot_label"]).replace(" ", "_")

    json_path = output_dir / f"{safe_label}.json"
    excel_path = output_dir / f"{safe_label}.xlsx"
    markdown_path = output_dir / f"{safe_label}.md"

    json_path.write_text(
        json.dumps(snapshot_to_dict(snapshot), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        _excel_safe(snapshot.metadata).to_excel(writer, sheet_name="Metadata", index=False)
        _excel_safe(snapshot.aggregate_metrics).to_excel(writer, sheet_name="Resumen", index=False)
        _excel_safe(snapshot.ticker_metrics).to_excel(writer, sheet_name="Metricas_Ticker", index=False)
        _excel_safe(snapshot.metric_catalog).to_excel(writer, sheet_name="Catalogo_Metricas", index=False)
        _excel_safe(snapshot.source_files).to_excel(writer, sheet_name="Fuentes", index=False)

    lines = [
        f"# Snapshot {metadata['snapshot_label']}",
        "",
        f"- Fecha de referencia: {metadata['as_of_date']}",
        f"- Versión del motor: {metadata['engine_version']}",
        f"- Régimen: {metadata['regime_label']}",
        f"- Alcance: {metadata['scope']}",
        "",
        "## Resumen agregado",
        "",
        "| Métrica | Valor |",
        "|---|---:|",
    ]
    for row in snapshot.aggregate_metrics.to_dict(orient="records"):
        lines.append(f"| {row['metric']} | {row['value']} |")
    lines.extend(["", "## Métricas por ticker", ""])
    visible = [
        column
        for column in [
            "ticker",
            "predictability_score",
            "predictability_grade",
            "observations",
            "directional_accuracy",
            "mean_brier_skill",
            "mean_calibration_error",
            "memory_label",
            "regime_label",
        ]
        if column in snapshot.ticker_metrics.columns
    ]
    if visible:
        lines.append(snapshot.ticker_metrics[visible].to_markdown(index=False))
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"json": json_path, "excel": excel_path, "markdown": markdown_path}


def load_snapshot_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_snapshots(baseline: dict[str, Any], current: dict[str, Any]) -> pd.DataFrame:
    baseline_metrics = {row["metric"]: row["value"] for row in baseline["aggregate_metrics"]}
    current_metrics = {row["metric"]: row["value"] for row in current["aggregate_metrics"]}
    rows: list[dict[str, Any]] = []

    for metric in sorted(set(baseline_metrics) | set(current_metrics)):
        old = baseline_metrics.get(metric)
        new = current_metrics.get(metric)
        comparable = isinstance(old, (int, float)) and isinstance(new, (int, float))
        change = float(new - old) if comparable else np.nan
        change_pct = float(change / abs(old)) if comparable and old not in (0, None) else np.nan
        rows.append(
            {
                "metric": metric,
                "baseline_value": old,
                "current_value": new,
                "absolute_change": change,
                "relative_change": change_pct,
                "status": "NEW" if old is None else "REMOVED" if new is None else "COMPARABLE" if comparable else "DEFINITION_REVIEW",
            }
        )

    baseline_catalog = {row["metric"] for row in baseline.get("metric_catalog", []) if row.get("present")}
    current_catalog = {row["metric"] for row in current.get("metric_catalog", []) if row.get("present")}
    for metric in sorted(current_catalog - baseline_catalog):
        rows.append(
            {
                "metric": f"catalog.new.{metric}",
                "baseline_value": None,
                "current_value": "PRESENT",
                "absolute_change": np.nan,
                "relative_change": np.nan,
                "status": "NEW_METRIC",
            }
        )
    for metric in sorted(baseline_catalog - current_catalog):
        rows.append(
            {
                "metric": f"catalog.removed.{metric}",
                "baseline_value": "PRESENT",
                "current_value": None,
                "absolute_change": np.nan,
                "relative_change": np.nan,
                "status": "REMOVED_METRIC",
            }
        )

    return pd.DataFrame(rows)

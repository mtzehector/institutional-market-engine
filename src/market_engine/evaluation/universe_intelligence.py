from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd


NUMERIC_METRICS = (
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
class UniverseIntelligenceResult:
    executive_summary: pd.DataFrame
    metric_statistics: pd.DataFrame
    grade_distribution: pd.DataFrame
    memory_distribution: pd.DataFrame
    top_tickers: pd.DataFrame
    bottom_tickers: pd.DataFrame
    coverage: pd.DataFrame
    manifest: pd.DataFrame
    ticker_metrics: pd.DataFrame


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_input(path: Path, preferred_sheet: str | None = None) -> tuple[pd.DataFrame, str]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("ticker_metrics", [])
        return pd.DataFrame(rows), "ticker_metrics"
    if suffix in {".xlsx", ".xls"}:
        workbook = pd.ExcelFile(path)
        candidates = [preferred_sheet, "Mejor_Memoria", "Ranking", "Hall_of_Fame", "Metricas_Ticker"]
        for sheet in candidates:
            if sheet and sheet in workbook.sheet_names:
                return pd.read_excel(path, sheet_name=sheet), sheet
        return pd.read_excel(path, sheet_name=workbook.sheet_names[0]), workbook.sheet_names[0]
    return pd.read_csv(path), "CSV"


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.dropna(how="all").copy()
    if data.empty:
        raise ValueError("El archivo no contiene filas de métricas")
    if "ticker" not in data.columns:
        raise ValueError("La entrada debe contener la columna ticker")

    data["ticker"] = data["ticker"].astype(str).str.strip().str.upper()
    data = data[data["ticker"].ne("")].drop_duplicates("ticker", keep="first")

    for column in NUMERIC_METRICS:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    if "predictability_score" not in data.columns:
        raise ValueError("La entrada debe contener predictability_score")
    return data.reset_index(drop=True)


def _metric_statistics(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in NUMERIC_METRICS:
        if metric not in data.columns:
            continue
        values = pd.to_numeric(data[metric], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "metric": metric,
                "count": int(values.size),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "min": float(values.min()),
                "p10": float(values.quantile(0.10)),
                "p25": float(values.quantile(0.25)),
                "p75": float(values.quantile(0.75)),
                "p90": float(values.quantile(0.90)),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def _distribution(data: pd.DataFrame, column: str, empty_label: str) -> pd.DataFrame:
    if column not in data.columns:
        return pd.DataFrame(columns=[column, "count", "percentage"])
    values = data[column].fillna(empty_label).astype(str)
    counts = values.value_counts(dropna=False)
    result = counts.rename_axis(column).reset_index(name="count")
    result["percentage"] = result["count"] / len(data)
    return result


def _coverage(data: pd.DataFrame, expected_tickers: int | None) -> pd.DataFrame:
    evaluated = int(data["ticker"].nunique())
    expected = int(expected_tickers) if expected_tickers else evaluated
    sufficient = int((pd.to_numeric(data.get("observations"), errors="coerce") >= 40).sum()) if "observations" in data.columns else 0
    return pd.DataFrame(
        [
            {"metric": "expected_tickers", "value": expected},
            {"metric": "evaluated_tickers", "value": evaluated},
            {"metric": "missing_tickers", "value": max(expected - evaluated, 0)},
            {"metric": "coverage_ratio", "value": evaluated / expected if expected else np.nan},
            {"metric": "sufficient_sample_tickers", "value": sufficient},
            {"metric": "sufficient_sample_ratio", "value": sufficient / evaluated if evaluated else np.nan},
        ]
    )


def build_universe_intelligence(
    input_path: Path,
    preferred_sheet: str | None = None,
    expected_tickers: int | None = None,
    engine_version: str = "unknown",
    regime_label: str = "UNSPECIFIED",
    top_n: int = 20,
) -> UniverseIntelligenceResult:
    frame, sheet = _read_input(input_path, preferred_sheet)
    data = _normalize(frame)
    statistics = _metric_statistics(data)
    grades = _distribution(data, "predictability_grade", "SIN_CLASIFICAR")
    memories = _distribution(data, "memory_label", "SIN_MEMORIA")
    coverage = _coverage(data, expected_tickers)

    ranked = data.sort_values(
        ["predictability_score", "ticker"], ascending=[False, True]
    ).reset_index(drop=True)
    top = ranked.head(top_n).copy()
    bottom = ranked.tail(top_n).sort_values("predictability_score").copy()

    score = pd.to_numeric(data["predictability_score"], errors="coerce")
    directional = pd.to_numeric(data.get("directional_accuracy"), errors="coerce")
    brier = pd.to_numeric(data.get("mean_brier_skill"), errors="coerce")
    calibration = pd.to_numeric(data.get("mean_calibration_error"), errors="coerce")

    executive = pd.DataFrame(
        [
            {"metric": "ticker_count", "value": int(data["ticker"].nunique())},
            {"metric": "predictability_mean", "value": float(score.mean())},
            {"metric": "predictability_median", "value": float(score.median())},
            {"metric": "predictability_std", "value": float(score.std(ddof=1))},
            {"metric": "directional_accuracy_mean", "value": float(directional.mean())},
            {"metric": "mean_brier_skill", "value": float(brier.mean())},
            {"metric": "mean_calibration_error", "value": float(calibration.mean())},
            {"metric": "best_ticker", "value": str(ranked.iloc[0]["ticker"])},
            {"metric": "best_score", "value": float(ranked.iloc[0]["predictability_score"])},
            {"metric": "worst_ticker", "value": str(ranked.iloc[-1]["ticker"])},
            {"metric": "worst_score", "value": float(ranked.iloc[-1]["predictability_score"])},
        ]
    )

    manifest = pd.DataFrame(
        [
            {"field": "created_at_utc", "value": datetime.now(timezone.utc).isoformat()},
            {"field": "engine_version", "value": engine_version},
            {"field": "regime_label", "value": regime_label},
            {"field": "source_file", "value": input_path.name},
            {"field": "source_sheet", "value": sheet},
            {"field": "source_sha256", "value": _file_sha256(input_path)},
            {"field": "ticker_count", "value": int(data["ticker"].nunique())},
            {"field": "top_n", "value": top_n},
        ]
    )

    return UniverseIntelligenceResult(
        executive_summary=executive,
        metric_statistics=statistics,
        grade_distribution=grades,
        memory_distribution=memories,
        top_tickers=top,
        bottom_tickers=bottom,
        coverage=coverage,
        manifest=manifest,
        ticker_metrics=ranked,
    )


def _excel_safe(frame: pd.DataFrame) -> pd.DataFrame:
    safe = frame.copy()
    for column in safe.columns:
        series = safe[column]
        if isinstance(series.dtype, pd.DatetimeTZDtype):
            safe[column] = series.dt.tz_localize(None)
    return safe


def export_universe_intelligence(
    result: UniverseIntelligenceResult,
    output_dir: Path,
    label: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_label = label.replace(" ", "_")
    excel_path = output_dir / f"{safe_label}.xlsx"
    json_path = output_dir / f"{safe_label}.json"
    markdown_path = output_dir / f"{safe_label}.md"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for sheet, frame in [
            ("Resumen_Ejecutivo", result.executive_summary),
            ("Estadisticas", result.metric_statistics),
            ("Distribucion_Grados", result.grade_distribution),
            ("Distribucion_Memoria", result.memory_distribution),
            ("Cobertura", result.coverage),
            ("Top", result.top_tickers),
            ("Bottom", result.bottom_tickers),
            ("Manifest", result.manifest),
            ("Metricas_Ticker", result.ticker_metrics),
        ]:
            _excel_safe(frame).to_excel(writer, sheet_name=sheet, index=False)

    payload = {
        "manifest": dict(zip(result.manifest["field"], result.manifest["value"])),
        "executive_summary": result.executive_summary.to_dict(orient="records"),
        "metric_statistics": result.metric_statistics.to_dict(orient="records"),
        "grade_distribution": result.grade_distribution.to_dict(orient="records"),
        "memory_distribution": result.memory_distribution.to_dict(orient="records"),
        "coverage": result.coverage.to_dict(orient="records"),
        "top_tickers": result.top_tickers.to_dict(orient="records"),
        "bottom_tickers": result.bottom_tickers.to_dict(orient="records"),
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    summary = dict(zip(result.executive_summary["metric"], result.executive_summary["value"]))
    lines = [
        f"# Universe Intelligence Report — {label}",
        "",
        f"- Tickers evaluados: {summary.get('ticker_count')}",
        f"- Predictibilidad promedio: {summary.get('predictability_mean')}",
        f"- Mediana: {summary.get('predictability_median')}",
        f"- Exactitud direccional promedio: {summary.get('directional_accuracy_mean')}",
        f"- Brier Skill promedio: {summary.get('mean_brier_skill')}",
        f"- Error de calibración promedio: {summary.get('mean_calibration_error')}",
        f"- Mejor ticker: {summary.get('best_ticker')} ({summary.get('best_score')})",
        f"- Ticker con menor score: {summary.get('worst_ticker')} ({summary.get('worst_score')})",
        "",
        "## Distribución de memorias",
        "",
        result.memory_distribution.to_markdown(index=False),
        "",
        "## Distribución de grados",
        "",
        result.grade_distribution.to_markdown(index=False),
        "",
        "## Top",
        "",
        result.top_tickers.head(20).to_markdown(index=False),
    ]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"excel": excel_path, "json": json_path, "markdown": markdown_path}

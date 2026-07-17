from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from market_engine.backtesting.walk_forward import walk_forward_gap
from market_engine.cli import _build_features_for_ticker, _excel_safe, _load_tickers
from market_engine.config import PROJECT_ROOT
from market_engine.evaluation.adaptive_universe import run_adaptive_universe


SCREEN_COLUMNS = [
    "date",
    "return_1d_pct",
    "return_3d_pct",
    "return_5d_pct",
    "close_position_pct",
    "close_vs_vwap_pct",
    "relative_volume",
    "smart_money_pct",
    "smart_money_slope_5d",
    "qqq_return_1d_pct",
    "atr_pct",
]


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _parse_sizes(text: str) -> tuple[int, ...]:
    sizes = tuple(sorted({int(token.strip()) for token in text.split(",") if token.strip()}))
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("--cohort-sizes debe contener enteros positivos")
    return sizes


def _load_memories(path_text: str | None, sheet: str) -> dict[str, int | None]:
    if not path_text:
        return {}
    path = _resolve(path_text)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de memorias: {path}")
    frame = pd.read_excel(path, sheet_name=sheet)
    if "ticker" not in frame.columns:
        raise ValueError("La hoja de memorias debe contener ticker")

    memories: dict[str, int | None] = {}
    for _, row in frame.iterrows():
        ticker = str(row["ticker"]).upper()
        value = row.get("memory_rows")
        if pd.notna(value):
            memories[ticker] = int(value)
        else:
            label = str(row.get("memory_label", "")).upper()
            memories[ticker] = None if label == "EXPANDING" else None
    return memories


def _evaluate(args: argparse.Namespace) -> int:
    evaluation_start = date.fromisoformat(args.from_date)
    evaluation_end = date.fromisoformat(args.to_date)
    if evaluation_end < evaluation_start:
        raise ValueError("--to-date no puede ser anterior a --from-date")

    tickers = _load_tickers(_resolve(args.tickers))
    cohort_sizes = _parse_sizes(args.cohort_sizes)
    memories = _load_memories(args.memory_input, args.memory_sheet)
    maximum_memory = max([value for value in memories.values() if value is not None] or [args.memory_rows])
    download_start = evaluation_start - timedelta(days=max(365 * 6, maximum_memory * 2))

    all_rows: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []

    for position, ticker in enumerate(tickers, 1):
        print(f"[{position}/{len(tickers)}] Adaptive Universe para {ticker}")
        try:
            features, _ = _build_features_for_ticker(
                ticker, download_start, evaluation_end, args.force_refresh
            )
            memory_rows = memories.get(ticker, args.memory_rows)
            result = walk_forward_gap(
                features,
                from_date=evaluation_start,
                to_date=evaluation_end,
                min_history_rows=args.min_history_rows,
                step=args.step,
                decision_threshold=args.decision_threshold,
                max_history_rows=memory_rows,
            )
            predictions = result.predictions.copy()
            screen = features[[column for column in SCREEN_COLUMNS if column in features.columns]].copy()
            screen = screen.rename(columns={"date": "origin_date"})
            predictions["origin_date"] = pd.to_datetime(predictions["origin_date"])
            screen["origin_date"] = pd.to_datetime(screen["origin_date"])
            merged = predictions.merge(screen, on="origin_date", how="left")
            merged["ticker"] = ticker
            merged["memory_rows_used"] = memory_rows
            all_rows.append(merged)
            print(f"{ticker}: predicciones={len(merged)}, memoria={memory_rows or 'EXPANDING'}")
        except Exception as exc:
            errors.append({"ticker": ticker, "error": str(exc)})
            print(f"{ticker}: ERROR - {exc}")

    if not all_rows:
        raise ValueError(f"No se generaron predicciones. Errores: {errors[:5]}")

    prediction_features = pd.concat(all_rows, ignore_index=True)
    result = run_adaptive_universe(prediction_features, cohort_sizes)
    errors_frame = pd.DataFrame(errors)

    print("\nCOMPARACIÓN DE COHORTES")
    display = [
        "cohort",
        "observations",
        "unique_tickers",
        "unique_origin_dates",
        "predictability_score",
        "balanced_accuracy",
        "macro_f1",
        "rare_event_f1",
        "mean_brier_skill",
        "mean_calibration_error",
        "bullish_score_mean",
        "bullish_score_std",
    ]
    print(result.cohort_metrics[[c for c in display if c in result.cohort_metrics.columns]].to_string(index=False))

    if args.export:
        output = _resolve(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            _excel_safe(result.cohort_metrics).to_excel(writer, sheet_name="Comparacion_Cohortes", index=False)
            _excel_safe(result.selections).to_excel(writer, sheet_name="Selecciones_Diarias", index=False)
            _excel_safe(result.cohort_predictions).to_excel(writer, sheet_name="Predicciones_Cohorte", index=False)
            _excel_safe(result.daily_diagnostics).to_excel(writer, sheet_name="Diagnostico_Diario", index=False)
            _excel_safe(errors_frame).to_excel(writer, sheet_name="Errores", index=False)
        print(f"Excel generado: {output}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="market-adaptive-universe")
    parser.add_argument("--tickers", required=True)
    parser.add_argument("--from-date", required=True)
    parser.add_argument("--to-date", required=True)
    parser.add_argument("--cohort-sizes", default="5,10,20")
    parser.add_argument("--memory-rows", type=int, default=180)
    parser.add_argument("--memory-input")
    parser.add_argument("--memory-sheet", default="Mejor_Memoria")
    parser.add_argument("--min-history-rows", type=int, default=180)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--decision-threshold", type=float, default=0.50)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--output", default="adaptive_universe_v083.xlsx")
    parser.add_argument("--force-refresh", action="store_true")
    parser.set_defaults(handler=_evaluate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

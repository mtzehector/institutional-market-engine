from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from market_engine.backtesting.walk_forward import walk_forward_gap
from market_engine.config import PROJECT_ROOT, Settings
from market_engine.evaluation.model_evaluation import (
    evaluate_predictions,
    rank_ticker_evaluations,
)
from market_engine.gaps.features import build_gap_features
from market_engine.gaps.predictor import predict_next_gap
from market_engine.providers.fmp import FMPProvider


def _build_features_for_ticker(
    ticker: str,
    start: date,
    end: date,
    force_refresh: bool,
) -> tuple[pd.DataFrame, Settings]:
    settings = Settings.from_env()
    provider = FMPProvider(settings)

    print(f"Descargando {ticker} y QQQ: {start} -> {end}")
    ticker_data = provider.historical_eod(ticker.upper(), start, end, force_refresh)
    qqq_data = provider.historical_eod("QQQ", start, end, force_refresh)

    features = build_gap_features(
        ticker_data,
        qqq_data,
        gap_min_pct=settings.gap_min_pct,
        gap_atr_multiplier=settings.gap_atr_multiplier,
        atr_length=settings.atr_length,
        smart_money_length=settings.smart_money_length,
        smart_money_volume_length=settings.smart_money_volume_length,
        smart_money_ema=settings.smart_money_ema,
    )
    return features, settings


def _excel_safe(frame: pd.DataFrame) -> pd.DataFrame:
    safe = frame.copy()
    for column in safe.columns:
        if isinstance(safe[column].dtype, pd.DatetimeTZDtype):
            safe[column] = safe[column].dt.tz_localize(None)
    return safe


def _load_tickers(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de tickers: {path}")
    tickers: list[str] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        ticker = raw.strip().upper()
        if ticker and not ticker.startswith("#") and ticker not in seen:
            seen.add(ticker)
            tickers.append(ticker)
    if not tickers:
        raise ValueError("El archivo no contiene tickers válidos")
    return tickers


def _predict_gap(args: argparse.Namespace) -> int:
    end = date.fromisoformat(args.to_date)
    start = end - timedelta(days=365 * args.years + 180)
    features, _ = _build_features_for_ticker(
        args.ticker, start, end, args.force_refresh
    )

    result = predict_next_gap(features)
    latest = features.dropna(subset=["smart_money_pct", "atr_pct"]).iloc[-1]

    row = {
        "ticker": args.ticker.upper(),
        "as_of_date": latest["date"],
        "gap_threshold_pct": latest["gap_threshold_pct"],
        **asdict(result),
        "smart_money_pct": latest["smart_money_pct"],
        "relative_volume": latest["relative_volume"],
        "atr_pct": latest["atr_pct"],
        "return_1d_pct": latest["return_1d_pct"],
        "return_5d_pct": latest["return_5d_pct"],
        "close_position_pct": latest["close_position_pct"],
        "close_vs_vwap_pct": latest["close_vs_vwap_pct"],
        "qqq_return_1d_pct": latest["qqq_return_1d_pct"],
    }

    print("\nPREDICCIÓN SIGUIENTE SESIÓN")
    print(pd.DataFrame([row]).to_string(index=False))

    if args.export:
        output = Path(args.output)
        if not output.is_absolute():
            output = PROJECT_ROOT / output
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            _excel_safe(pd.DataFrame([row])).to_excel(
                writer, sheet_name="Prediccion", index=False
            )
            _excel_safe(features).to_excel(writer, sheet_name="Auditoria", index=False)
        print(f"Excel generado: {output}")

    return 0


def _walk_forward(args: argparse.Namespace) -> int:
    evaluation_start = date.fromisoformat(args.from_date)
    evaluation_end = date.fromisoformat(args.to_date)
    if evaluation_end < evaluation_start:
        raise ValueError("--to-date no puede ser anterior a --from-date")

    download_start = evaluation_start - timedelta(days=365 * args.training_years + 180)
    features, _ = _build_features_for_ticker(
        args.ticker, download_start, evaluation_end, args.force_refresh
    )

    result = walk_forward_gap(
        features,
        from_date=evaluation_start,
        to_date=evaluation_end,
        min_history_rows=args.min_history_rows,
        step=args.step,
        decision_threshold=args.decision_threshold,
    )

    print("\nRESULTADOS WALK-FORWARD")
    display_columns = [
        "target_date",
        "actual_gap_pct",
        "gap_threshold_pct",
        "probability_up",
        "probability_down",
        "probability_no_gap",
        "predicted_direction",
        "actual_direction",
        "correct_direction",
        "training_rows",
    ]
    print(result.predictions[display_columns].to_string(index=False))

    print("\nMÉTRICAS")
    print(result.metrics.to_string(index=False))

    if args.export:
        output = Path(args.output)
        if not output.is_absolute():
            output = PROJECT_ROOT / output
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            _excel_safe(result.predictions).to_excel(
                writer, sheet_name="Predicciones", index=False
            )
            _excel_safe(result.metrics).to_excel(writer, sheet_name="Metricas", index=False)
            _excel_safe(result.calibration).to_excel(
                writer, sheet_name="Calibracion", index=False
            )
        print(f"Excel generado: {output}")

    return 0


def _evaluate_models(args: argparse.Namespace) -> int:
    evaluation_start = date.fromisoformat(args.from_date)
    evaluation_end = date.fromisoformat(args.to_date)
    if evaluation_end < evaluation_start:
        raise ValueError("--to-date no puede ser anterior a --from-date")

    if args.ticker:
        tickers = [args.ticker.upper()]
    else:
        ticker_file = Path(args.tickers)
        if not ticker_file.is_absolute():
            ticker_file = PROJECT_ROOT / ticker_file
        tickers = _load_tickers(ticker_file)

    summaries: list[pd.DataFrame] = []
    confidence_tables: list[pd.DataFrame] = []
    recent_tables: list[pd.DataFrame] = []
    prediction_tables: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []

    download_start = evaluation_start - timedelta(days=365 * args.training_years + 180)

    for position, ticker in enumerate(tickers, 1):
        print(f"\n[{position}/{len(tickers)}] Evaluando {ticker}")
        try:
            features, _ = _build_features_for_ticker(
                ticker, download_start, evaluation_end, args.force_refresh
            )
            walk_result = walk_forward_gap(
                features,
                from_date=evaluation_start,
                to_date=evaluation_end,
                min_history_rows=args.min_history_rows,
                step=args.step,
                decision_threshold=args.decision_threshold,
            )
            evaluation = evaluate_predictions(
                walk_result.predictions,
                ticker=ticker,
                high_confidence_threshold=args.high_confidence_threshold,
            )

            summaries.append(evaluation.summary)

            confidence = evaluation.confidence_bands.copy()
            confidence.insert(0, "ticker", ticker)
            confidence_tables.append(confidence)

            recent = evaluation.recent_performance.copy()
            recent.insert(0, "ticker", ticker)
            recent_tables.append(recent)

            if args.include_predictions or len(tickers) == 1:
                predictions = walk_result.predictions.copy()
                predictions.insert(0, "ticker", ticker)
                prediction_tables.append(predictions)

            score = float(evaluation.summary.iloc[0]["predictability_score"])
            grade = str(evaluation.summary.iloc[0]["predictability_grade"])
            print(f"{ticker}: score={score:.2f}, grado={grade}")

        except Exception as exc:
            errors.append({"ticker": ticker, "error": str(exc)})
            print(f"{ticker}: ERROR - {exc}")

    if not summaries:
        raise ValueError(f"No se pudo evaluar ningún ticker. Errores: {errors[:5]}")

    summary_frame = pd.concat(summaries, ignore_index=True)
    ranking = rank_ticker_evaluations(summary_frame)
    confidence_frame = (
        pd.concat(confidence_tables, ignore_index=True)
        if confidence_tables
        else pd.DataFrame()
    )
    recent_frame = (
        pd.concat(recent_tables, ignore_index=True)
        if recent_tables
        else pd.DataFrame()
    )
    predictions_frame = (
        pd.concat(prediction_tables, ignore_index=True)
        if prediction_tables
        else pd.DataFrame()
    )
    errors_frame = pd.DataFrame(errors)

    print("\nRANKING DE PREDICTIBILIDAD")
    display_columns = [
        "rank",
        "ticker",
        "predictability_score",
        "predictability_grade",
        "observations",
        "directional_accuracy",
        "confidence_weighted_accuracy",
        "mean_brier_skill",
        "mean_calibration_error",
        "high_confidence_accuracy",
        "high_confidence_observations",
    ]
    print(ranking[display_columns].to_string(index=False))

    if args.export:
        output = Path(args.output)
        if not output.is_absolute():
            output = PROJECT_ROOT / output
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            _excel_safe(ranking).to_excel(writer, sheet_name="Ranking", index=False)
            _excel_safe(confidence_frame).to_excel(
                writer, sheet_name="Bandas_Confianza", index=False
            )
            _excel_safe(recent_frame).to_excel(
                writer, sheet_name="Desempeno_Reciente", index=False
            )
            _excel_safe(predictions_frame).to_excel(
                writer, sheet_name="Predicciones", index=False
            )
            _excel_safe(errors_frame).to_excel(writer, sheet_name="Errores", index=False)
        print(f"Excel generado: {output}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="market-engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict_gap = subparsers.add_parser(
        "predict-gap", help="Predice gap up/down significativo de la siguiente sesión"
    )
    predict_gap.add_argument("--ticker", required=True)
    predict_gap.add_argument("--years", type=int, default=5)
    predict_gap.add_argument("--to-date", default=date.today().isoformat())
    predict_gap.add_argument("--export", action="store_true")
    predict_gap.add_argument("--output", default="prediccion_gap_manana_FMP.xlsx")
    predict_gap.add_argument("--force-refresh", action="store_true")
    predict_gap.set_defaults(handler=_predict_gap)

    walk_forward = subparsers.add_parser(
        "walk-forward",
        help="Evalúa cronológicamente el predictor de gap sin fuga de información",
    )
    walk_forward.add_argument("--ticker", required=True)
    walk_forward.add_argument("--from-date", required=True)
    walk_forward.add_argument("--to-date", required=True)
    walk_forward.add_argument("--training-years", type=int, default=5)
    walk_forward.add_argument("--min-history-rows", type=int, default=180)
    walk_forward.add_argument(
        "--step",
        type=int,
        default=1,
        help="Predice cada N sesiones; use 1 para todas las sesiones",
    )
    walk_forward.add_argument("--decision-threshold", type=float, default=0.50)
    walk_forward.add_argument("--export", action="store_true")
    walk_forward.add_argument("--output", default="walk_forward_gap_FMP.xlsx")
    walk_forward.add_argument("--force-refresh", action="store_true")
    walk_forward.set_defaults(handler=_walk_forward)

    evaluate = subparsers.add_parser(
        "evaluate-models",
        help="Puntúa la predictibilidad del modelo por ticker mediante walk-forward",
    )
    source = evaluate.add_mutually_exclusive_group(required=True)
    source.add_argument("--ticker")
    source.add_argument("--tickers", help="TXT con un ticker por línea")
    evaluate.add_argument("--from-date", required=True)
    evaluate.add_argument("--to-date", required=True)
    evaluate.add_argument("--training-years", type=int, default=5)
    evaluate.add_argument("--min-history-rows", type=int, default=180)
    evaluate.add_argument("--step", type=int, default=1)
    evaluate.add_argument("--decision-threshold", type=float, default=0.50)
    evaluate.add_argument("--high-confidence-threshold", type=float, default=0.60)
    evaluate.add_argument(
        "--include-predictions",
        action="store_true",
        help="Incluye el detalle de todas las predicciones en el Excel",
    )
    evaluate.add_argument("--export", action="store_true")
    evaluate.add_argument("--output", default="ranking_predictibilidad_FMP.xlsx")
    evaluate.add_argument("--force-refresh", action="store_true")
    evaluate.set_defaults(handler=_evaluate_models)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

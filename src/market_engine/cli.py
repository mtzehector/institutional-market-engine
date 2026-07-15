from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from market_engine.backtesting.walk_forward import walk_forward_gap
from market_engine.config import PROJECT_ROOT, Settings
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

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from market_engine.config import PROJECT_ROOT, Settings
from market_engine.gaps.features import build_gap_features
from market_engine.gaps.predictor import predict_next_gap
from market_engine.providers.fmp import FMPProvider


def _predict_gap(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    provider = FMPProvider(settings)
    end = date.fromisoformat(args.to_date)
    start = end - timedelta(days=365 * args.years + 180)

    print(f"Descargando {args.ticker} y QQQ: {start} -> {end}")
    ticker_data = provider.historical_eod(
        args.ticker.upper(), start, end, args.force_refresh
    )
    qqq_data = provider.historical_eod("QQQ", start, end, args.force_refresh)

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
        audit = features.copy()
        for column in audit.columns:
            if isinstance(audit[column].dtype, pd.DatetimeTZDtype):
                audit[column] = audit[column].dt.tz_localize(None)
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            pd.DataFrame([row]).to_excel(writer, sheet_name="Prediccion", index=False)
            audit.to_excel(writer, sheet_name="Auditoria", index=False)
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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from market_engine.cli import _build_features_for_ticker, _excel_safe, _load_tickers
from market_engine.config import PROJECT_ROOT
from market_engine.evaluation.smart_money_regimes import (
    analyze_smart_money_regimes,
    correlation_report,
)


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_evaluation(path_text: str | None, sheet: str) -> pd.DataFrame:
    if not path_text:
        return pd.DataFrame()
    path = _resolve(path_text)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de evaluación: {path}")
    frame = pd.read_excel(path, sheet_name=sheet)
    if "ticker" not in frame.columns:
        raise ValueError("La hoja de evaluación debe contener la columna ticker")
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    return frame


def _ticker_memory_rows(
    ticker: str,
    evaluation_by_ticker: pd.DataFrame,
    default_memory_rows: int,
) -> int:
    if evaluation_by_ticker.empty or ticker not in evaluation_by_ticker.index:
        return default_memory_rows
    row = evaluation_by_ticker.loc[ticker]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    value = row.get("memory_rows", default_memory_rows)
    if pd.isna(value):
        return default_memory_rows
    return max(1, int(value))


def _evaluate(args: argparse.Namespace) -> int:
    start = date.fromisoformat(args.from_date)
    end = date.fromisoformat(args.to_date)
    if end < start:
        raise ValueError("--to-date no puede ser anterior a --from-date")

    tickers = [args.ticker.upper()] if args.ticker else _load_tickers(_resolve(args.tickers))
    evaluation = _load_evaluation(args.evaluation_input, args.evaluation_sheet)
    evaluation_by_ticker = evaluation.set_index("ticker") if not evaluation.empty else pd.DataFrame()

    maximum_memory = args.memory_rows
    if not evaluation.empty and "memory_rows" in evaluation.columns:
        usable = pd.to_numeric(evaluation["memory_rows"], errors="coerce").dropna()
        if not usable.empty:
            maximum_memory = max(maximum_memory, int(usable.max()))

    summaries: list[pd.DataFrame] = []
    regimes: list[pd.DataFrame] = []
    crossings: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []
    download_start = start - timedelta(days=max(365 * 6, maximum_memory * 2))

    for position, ticker in enumerate(tickers, 1):
        print(f"[{position}/{len(tickers)}] Regímenes Smart Money para {ticker}")
        try:
            memory_rows = _ticker_memory_rows(ticker, evaluation_by_ticker, args.memory_rows)
            features, _ = _build_features_for_ticker(
                ticker, download_start, end, args.force_refresh
            )
            features = features.loc[pd.to_datetime(features["date"]).dt.date <= end].copy()
            result = analyze_smart_money_regimes(
                features,
                ticker=ticker,
                memory_rows=memory_rows,
                lower_equilibrium=args.lower_equilibrium,
                upper_equilibrium=args.upper_equilibrium,
                minimum_persistence=args.minimum_persistence,
                lookaround=args.lookaround,
                false_regime_days=args.false_regime_days,
                high_volume_threshold=args.high_volume_threshold,
            )
            summary = result.summary.copy()
            summary["requested_from_date"] = start
            summary["requested_to_date"] = end
            if not evaluation.empty and ticker in evaluation_by_ticker.index:
                evaluation_row = evaluation_by_ticker.loc[[ticker]].reset_index()
                summary = summary.merge(
                    evaluation_row,
                    on="ticker",
                    how="left",
                    suffixes=("", "_evaluation"),
                )
            summaries.append(summary)
            if not result.regimes.empty:
                regimes.append(result.regimes)
            if not result.crossings.empty:
                crossings.append(result.crossings)

            row = summary.iloc[0]
            print(
                f"{ticker}: memoria={memory_rows}, cruces={int(row['equilibrium_cross_count'])}, "
                f"duración mediana={row['median_regime_duration']}, "
                f"fuerza={row['median_institutional_regime_strength']}"
            )
        except Exception as exc:
            errors.append({"ticker": ticker, "error": str(exc)})
            print(f"{ticker}: ERROR - {exc}")

    if not summaries:
        raise ValueError(f"No se evaluó ningún ticker. Errores: {errors[:5]}")

    summary_frame = pd.concat(summaries, ignore_index=True)
    regimes_frame = pd.concat(regimes, ignore_index=True) if regimes else pd.DataFrame()
    crossings_frame = pd.concat(crossings, ignore_index=True) if crossings else pd.DataFrame()
    errors_frame = pd.DataFrame(errors)
    correlations = correlation_report(summary_frame)

    display_columns = [
        "ticker",
        "memory_rows",
        "equilibrium_cross_count",
        "crossings_per_100_sessions",
        "median_regime_duration",
        "median_relative_volume",
        "median_institutional_regime_strength",
        "volume_weighted_memory_regime_load",
        "smart_money_inertia_index",
        "smart_money_regime_coherence",
    ]
    print("\nRESUMEN DE REGÍMENES")
    print(summary_frame[[c for c in display_columns if c in summary_frame.columns]].to_string(index=False))

    if args.export:
        output = _resolve(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            _excel_safe(summary_frame).to_excel(writer, sheet_name="Resumen_Ticker", index=False)
            _excel_safe(regimes_frame).to_excel(writer, sheet_name="Regimenes", index=False)
            _excel_safe(crossings_frame).to_excel(writer, sheet_name="Cruces", index=False)
            _excel_safe(correlations).to_excel(writer, sheet_name="Correlaciones", index=False)
            _excel_safe(errors_frame).to_excel(writer, sheet_name="Errores", index=False)
        print(f"Excel generado: {output}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="market-regime")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ticker")
    source.add_argument("--tickers")
    parser.add_argument("--from-date", required=True)
    parser.add_argument("--to-date", required=True)
    parser.add_argument(
        "--memory-rows",
        type=int,
        default=180,
        help="Memoria por defecto; si el Excel contiene memory_rows se usa la óptima por ticker",
    )
    parser.add_argument("--lower-equilibrium", type=float, default=49.0)
    parser.add_argument("--upper-equilibrium", type=float, default=51.0)
    parser.add_argument("--minimum-persistence", type=int, default=2)
    parser.add_argument("--lookaround", type=int, default=3)
    parser.add_argument("--false-regime-days", type=int, default=3)
    parser.add_argument("--high-volume-threshold", type=float, default=1.25)
    parser.add_argument("--evaluation-input")
    parser.add_argument("--evaluation-sheet", default="Mejor_Memoria")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--output", default="smart_money_regimes_v082.xlsx")
    parser.add_argument("--force-refresh", action="store_true")
    parser.set_defaults(handler=_evaluate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

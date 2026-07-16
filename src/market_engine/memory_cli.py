from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from market_engine.cli import _build_features_for_ticker, _excel_safe, _load_tickers
from market_engine.config import PROJECT_ROOT
from market_engine.evaluation.adaptive_memory import (
    evaluate_memory_windows,
    rank_best_memories,
)


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _parse_windows(text: str) -> list[int | None]:
    windows: list[int | None] = []
    for token in text.split(","):
        value = token.strip().lower()
        if not value:
            continue
        if value in {"expanding", "full", "none"}:
            windows.append(None)
        else:
            windows.append(int(value))
    return windows


def _evaluate(args: argparse.Namespace) -> int:
    evaluation_start = date.fromisoformat(args.from_date)
    evaluation_end = date.fromisoformat(args.to_date)
    if evaluation_end < evaluation_start:
        raise ValueError("--to-date no puede ser anterior a --from-date")

    if args.ticker:
        tickers = [args.ticker.upper()]
    else:
        tickers = _load_tickers(_resolve(args.tickers))

    windows = _parse_windows(args.windows)
    finite_windows = [window for window in windows if window is not None]
    maximum_rows = max(finite_windows) if finite_windows else 1260
    download_start = evaluation_start - timedelta(days=max(365 * 6, maximum_rows * 2))

    all_rankings: list[pd.DataFrame] = []
    best_rows: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []

    for position, ticker in enumerate(tickers, 1):
        print(f"\n[{position}/{len(tickers)}] Memoria adaptativa para {ticker}")
        try:
            features, _ = _build_features_for_ticker(
                ticker, download_start, evaluation_end, args.force_refresh
            )
            result = evaluate_memory_windows(
                features,
                ticker=ticker,
                from_date=evaluation_start,
                to_date=evaluation_end,
                memory_windows=windows,
                min_history_rows=args.min_history_rows,
                step=args.step,
                decision_threshold=args.decision_threshold,
                high_confidence_threshold=args.high_confidence_threshold,
                include_predictions=args.include_predictions,
            )
            ranking = result.ranking.copy()
            ranking["regime_label"] = args.regime_label
            all_rankings.append(ranking)

            best = result.best_by_ticker.copy()
            best["regime_label"] = args.regime_label
            best_rows.append(best)

            if not result.predictions.empty:
                all_predictions.append(result.predictions)

            top = best.iloc[0]
            print(
                f"{ticker}: mejor={top['memory_label']}, "
                f"score={float(top['predictability_score']):.2f}"
            )
        except Exception as exc:
            errors.append({"ticker": ticker, "error": str(exc)})
            print(f"{ticker}: ERROR - {exc}")

    if not best_rows:
        raise ValueError(f"No se evaluó ningún ticker. Errores: {errors[:5]}")

    comparison = pd.concat(all_rankings, ignore_index=True)
    best_frame = rank_best_memories(pd.concat(best_rows, ignore_index=True))
    predictions = (
        pd.concat(all_predictions, ignore_index=True)
        if all_predictions
        else pd.DataFrame()
    )
    errors_frame = pd.DataFrame(errors)

    print("\nMEJOR MEMORIA POR TICKER")
    columns = [
        "adaptive_rank",
        "ticker",
        "memory_label",
        "predictability_score",
        "predictability_grade",
        "observations",
        "directional_accuracy",
        "mean_brier_skill",
        "mean_calibration_error",
        "regime_label",
    ]
    print(best_frame[[c for c in columns if c in best_frame.columns]].to_string(index=False))

    if args.export:
        output = _resolve(args.output)
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            _excel_safe(best_frame).to_excel(writer, sheet_name="Mejor_Memoria", index=False)
            _excel_safe(comparison).to_excel(writer, sheet_name="Comparacion_Ventanas", index=False)
            _excel_safe(predictions).to_excel(writer, sheet_name="Predicciones", index=False)
            _excel_safe(errors_frame).to_excel(writer, sheet_name="Errores", index=False)
        print(f"Excel generado: {output}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="market-memory")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ticker")
    source.add_argument("--tickers")
    parser.add_argument("--from-date", required=True)
    parser.add_argument("--to-date", required=True)
    parser.add_argument(
        "--windows",
        default="180,252,378,504,756,expanding",
        help="Ventanas en sesiones separadas por coma; use expanding para historia completa",
    )
    parser.add_argument("--regime-label", default="REGIME_2026")
    parser.add_argument("--min-history-rows", type=int, default=180)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--decision-threshold", type=float, default=0.50)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.60)
    parser.add_argument("--include-predictions", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--output", default="adaptive_memory_FMP.xlsx")
    parser.add_argument("--force-refresh", action="store_true")
    parser.set_defaults(handler=_evaluate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

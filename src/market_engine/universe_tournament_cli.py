from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_engine.cli import _excel_safe
from market_engine.config import PROJECT_ROOT
from market_engine.evaluation.universe_tournament import run_universe_tournament


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _parse_sizes(text: str) -> tuple[int, ...]:
    sizes = tuple(sorted({int(token.strip()) for token in text.split(",") if token.strip()}))
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("--cohort-sizes debe contener enteros positivos")
    return sizes


def _load_prediction_universe(path_text: str, sheet: str) -> pd.DataFrame:
    path = _resolve(path_text)
    if not path.exists():
        raise FileNotFoundError(f"No existe el Excel del Adaptive Universe: {path}")
    frame = pd.read_excel(path, sheet_name=sheet)
    if frame.empty:
        raise ValueError(f"La hoja {sheet} está vacía")

    # The v0.8.3 workbook contains the same observation in TOP_5, TOP_10,
    # TOP_20 and UNIVERSE. Use only UNIVERSE to avoid duplicated evidence.
    if "cohort" in frame.columns:
        universe = frame.loc[frame["cohort"].astype(str).str.upper() == "UNIVERSE"].copy()
        if not universe.empty:
            frame = universe

    key = [column for column in ("ticker", "origin_date", "target_date") if column in frame.columns]
    if len(key) == 3:
        frame = frame.drop_duplicates(key, keep="last")
    return frame.reset_index(drop=True)


def _evaluate(args: argparse.Namespace) -> int:
    sizes = _parse_sizes(args.cohort_sizes)
    prediction_features = _load_prediction_universe(args.input, args.sheet)
    result = run_universe_tournament(prediction_features, sizes)

    display = [
        "tournament_rank",
        "tournament_entry",
        "strategy",
        "cohort_size",
        "observations",
        "unique_origin_dates",
        "tournament_score",
        "predictability_score",
        "balanced_accuracy",
        "macro_f1",
        "rare_event_precision",
        "rare_event_recall",
        "rare_event_f1",
        "mean_brier_skill",
        "mean_calibration_error",
    ]
    print("\nADAPTIVE UNIVERSE TOURNAMENT")
    print(result.leaderboard[[c for c in display if c in result.leaderboard.columns]].to_string(index=False))

    if args.export:
        output = _resolve(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            _excel_safe(result.leaderboard).to_excel(writer, sheet_name="Leaderboard", index=False)
            _excel_safe(result.strategy_metrics).to_excel(writer, sheet_name="Metricas", index=False)
            _excel_safe(result.selections).to_excel(writer, sheet_name="Scores_Diarios", index=False)
            _excel_safe(result.predictions).to_excel(writer, sheet_name="Predicciones_Torneo", index=False)
            _excel_safe(result.daily_diagnostics).to_excel(writer, sheet_name="Diagnostico_Diario", index=False)
        print(f"Excel generado: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="market-universe-tournament")
    parser.add_argument("--input", required=True, help="Excel generado por market-adaptive-universe")
    parser.add_argument("--sheet", default="Predicciones_Cohorte")
    parser.add_argument("--cohort-sizes", default="5,10,20")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--output", default="adaptive_universe_tournament_v084.xlsx")
    parser.set_defaults(handler=_evaluate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

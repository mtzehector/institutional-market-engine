from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_engine.cli import _excel_safe
from market_engine.config import PROJECT_ROOT
from market_engine.evaluation.champion_validation import run_champion_validation


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_prediction_universe(path_text: str, sheet: str) -> pd.DataFrame:
    path = _resolve(path_text)
    if not path.exists():
        raise FileNotFoundError(f"No existe el Excel de entrada: {path}")
    frame = pd.read_excel(path, sheet_name=sheet)
    if frame.empty:
        raise ValueError(f"La hoja {sheet} está vacía")

    if "tournament_entry" in frame.columns:
        universal = frame.loc[
            frame["tournament_entry"].astype(str).str.upper() == "UNIVERSE"
        ].copy()
        if not universal.empty:
            frame = universal
    if "cohort" in frame.columns:
        universal = frame.loc[frame["cohort"].astype(str).str.upper() == "UNIVERSE"].copy()
        if not universal.empty:
            frame = universal

    key = [column for column in ("ticker", "origin_date", "target_date") if column in frame.columns]
    if len(key) == 3:
        frame = frame.drop_duplicates(key, keep="last")
    return frame.reset_index(drop=True)


def _evaluate(args: argparse.Namespace) -> int:
    prediction_features = _load_prediction_universe(args.input, args.sheet)
    result = run_champion_validation(prediction_features, frequency=args.frequency)

    display = [
        "stability_rank",
        "champion",
        "windows",
        "stability_score",
        "mean_quality",
        "minimum_quality",
        "quality_std",
        "mean_rare_event_f1",
        "minimum_rare_event_f1",
        "positive_brier_windows",
    ]
    print("\nCHAMPION VALIDATION LABORATORY — ESTABILIDAD TEMPORAL")
    print(result.stability_ranking[[c for c in display if c in result.stability_ranking.columns]].to_string(index=False))

    if args.export:
        output = _resolve(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            _excel_safe(result.stability_ranking).to_excel(writer, sheet_name="Ranking_Estabilidad", index=False)
            _excel_safe(result.window_metrics).to_excel(writer, sheet_name="Metricas_Ventanas", index=False)
            _excel_safe(result.degradation).to_excel(writer, sheet_name="Degradacion", index=False)
            _excel_safe(result.window_definitions).to_excel(writer, sheet_name="Ventanas", index=False)
            _excel_safe(result.selections).to_excel(writer, sheet_name="Selecciones", index=False)
        print(f"Excel generado: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="market-champion-validation")
    parser.add_argument("--input", required=True, help="Excel v0.8.4 con Predicciones_Torneo")
    parser.add_argument("--sheet", default="Predicciones_Torneo")
    parser.add_argument("--frequency", choices=["M", "Q"], default="M")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--output", default="champion_validation_v092.xlsx")
    parser.set_defaults(handler=_evaluate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

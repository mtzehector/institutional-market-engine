from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_engine.cli import _excel_safe
from market_engine.config import PROJECT_ROOT
from market_engine.evaluation.evolutionary_universe import run_evolutionary_universe


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
        raise FileNotFoundError(f"No existe el Excel de entrada: {path}")
    frame = pd.read_excel(path, sheet_name=sheet)
    if frame.empty:
        raise ValueError(f"La hoja {sheet} está vacía")

    # Accept v0.8.3 or v0.8.4 workbooks. Both contain repeated evidence for
    # several cohorts/strategies; retain only the universal control rows.
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
    sizes = _parse_sizes(args.cohort_sizes)
    prediction_features = _load_prediction_universe(args.input, args.sheet)
    result = run_evolutionary_universe(
        prediction_features,
        cohort_sizes=sizes,
        max_factors=args.max_factors,
        validation_fraction=args.validation_fraction,
        hall_of_fame_size=args.hall_of_fame_size,
    )

    display = [
        "evolution_rank",
        "candidate_id",
        "factor_signature",
        "factor_count",
        "cohort_size",
        "evolution_score",
        "validation_quality_score",
        "generalization_gap",
        "validation_observations",
        "validation_unique_origin_dates",
        "validation_rare_event_f1",
        "validation_balanced_accuracy",
        "validation_macro_f1",
        "validation_mean_brier_skill",
        "validation_mean_calibration_error",
    ]
    print("\nEVOLUTIONARY UNIVERSE LABORATORY — LEADERBOARD")
    print(result.leaderboard[[c for c in display if c in result.leaderboard.columns]].head(args.top).to_string(index=False))

    print("\nHALL OF FAME DIVERSO")
    hall_display = ["hall_of_fame_rank", *display[1:]]
    print(result.hall_of_fame[[c for c in hall_display if c in result.hall_of_fame.columns]].to_string(index=False))

    if args.export:
        output = _resolve(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            _excel_safe(result.leaderboard).to_excel(writer, sheet_name="Leaderboard", index=False)
            _excel_safe(result.hall_of_fame).to_excel(writer, sheet_name="Hall_of_Fame", index=False)
            _excel_safe(result.candidate_metrics).to_excel(writer, sheet_name="Metricas_Candidatos", index=False)
            _excel_safe(result.candidate_definitions).to_excel(writer, sheet_name="Definiciones", index=False)
            _excel_safe(result.selections).to_excel(writer, sheet_name="Selecciones", index=False)
            _excel_safe(result.daily_diagnostics).to_excel(writer, sheet_name="Diagnostico_Diario", index=False)
        print(f"Excel generado: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="market-evolutionary-universe")
    parser.add_argument("--input", required=True, help="Excel v0.8.3 o v0.8.4")
    parser.add_argument("--sheet", default="Predicciones_Torneo")
    parser.add_argument("--cohort-sizes", default="5,10,20")
    parser.add_argument("--max-factors", type=int, default=3)
    parser.add_argument("--validation-fraction", type=float, default=0.30)
    parser.add_argument("--hall-of-fame-size", type=int, default=12)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--output", default="evolutionary_universe_v090.xlsx")
    parser.set_defaults(handler=_evaluate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

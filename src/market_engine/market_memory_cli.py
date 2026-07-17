from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_engine.cli import _excel_safe
from market_engine.config import PROJECT_ROOT
from market_engine.evaluation.market_memory import run_market_memory_laboratory


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _evaluate(args: argparse.Namespace) -> int:
    path = _resolve(args.input)
    if not path.exists():
        raise FileNotFoundError(f"No existe el Excel de entrada: {path}")

    daily_states = pd.read_excel(path, sheet_name=args.states_sheet)
    selections = pd.read_excel(path, sheet_name=args.selections_sheet)
    if daily_states.empty:
        raise ValueError(f"La hoja {args.states_sheet} está vacía")
    if selections.empty:
        raise ValueError(f"La hoja {args.selections_sheet} está vacía")

    result = run_market_memory_laboratory(
        daily_states,
        selections,
        neighbors=args.neighbors,
        minimum_history=args.minimum_history,
    )

    print("\nMARKET MEMORY ENGINE — RESUMEN")
    if result.summary.empty:
        print("No hubo fechas suficientes para generar recomendaciones.")
    else:
        print(result.summary.to_string(index=False))

    print("\nRECOMENDACIONES CAUSALES")
    display = [
        "target_date",
        "recommended_champion",
        "memory_score",
        "recommendation_margin",
        "neighbor_dates_used",
        "mean_neighbor_distance",
        "actual_selected_quality",
        "actual_oracle_champion",
        "actual_oracle_quality",
        "actual_universe_quality",
        "advantage_vs_universe",
        "oracle_regret",
        "selected_was_oracle",
    ]
    if result.recommendations.empty:
        print("Sin recomendaciones.")
    else:
        print(
            result.recommendations[
                [column for column in display if column in result.recommendations.columns]
            ].to_string(index=False)
        )

    if args.export:
        output = _resolve(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            _excel_safe(result.summary).to_excel(writer, sheet_name="Resumen", index=False)
            _excel_safe(result.recommendations).to_excel(
                writer, sheet_name="Recomendaciones", index=False
            )
            _excel_safe(result.champion_scores).to_excel(
                writer, sheet_name="Scores_Campeones", index=False
            )
            _excel_safe(result.neighbor_states).to_excel(
                writer, sheet_name="Estados_Similares", index=False
            )
            _excel_safe(result.date_champion_quality).to_excel(
                writer, sheet_name="Calidad_Real_Fecha", index=False
            )
        print(f"Excel generado: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="market-market-memory")
    parser.add_argument(
        "--input", required=True, help="Excel v0.9.3.1 con Estados_Diarios y Selecciones"
    )
    parser.add_argument("--states-sheet", default="Estados_Diarios")
    parser.add_argument("--selections-sheet", default="Selecciones")
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--minimum-history", type=int, default=8)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--output", default="market_memory_v094.xlsx")
    parser.set_defaults(handler=_evaluate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

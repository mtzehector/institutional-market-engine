from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_engine.cli import _excel_safe
from market_engine.config import PROJECT_ROOT
from market_engine.evaluation.champion_lifecycle import run_champion_lifecycle_laboratory


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _evaluate(args: argparse.Namespace) -> int:
    path = _resolve(args.input)
    if not path.exists():
        raise FileNotFoundError(f"No existe el Excel de entrada: {path}")

    selections = pd.read_excel(path, sheet_name=args.selections_sheet)
    daily_states = pd.read_excel(path, sheet_name=args.states_sheet)
    if selections.empty:
        raise ValueError(f"La hoja {args.selections_sheet} está vacía")

    result = run_champion_lifecycle_laboratory(
        selections,
        daily_states,
        short_window=args.short_window,
        long_window=args.long_window,
        minimum_history=args.minimum_history,
        minimum_state_persistence=args.minimum_state_persistence,
    )

    print("\nCHAMPION LIFE CYCLE ENGINE — ESTADO ACTUAL")
    if result.current_status.empty:
        print("No hubo datos suficientes para clasificar campeones.")
    else:
        columns = [
            "champion",
            "lifecycle_state",
            "performance_level",
            "performance_direction",
            "recommended_action",
            "lifecycle_health_score",
            "deployment_score",
            "short_advantage",
            "advantage_slope",
        ]
        print(result.current_status[columns].to_string(index=False))

    if args.export:
        output = _resolve(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            _excel_safe(result.summary).to_excel(writer, sheet_name="Resumen", index=False)
            _excel_safe(result.current_status).to_excel(
                writer, sheet_name="Estado_Actual", index=False
            )
            _excel_safe(result.lifecycle_history).to_excel(
                writer, sheet_name="Historial_Ciclo", index=False
            )
            _excel_safe(result.transitions).to_excel(
                writer, sheet_name="Transiciones", index=False
            )
            _excel_safe(result.regime_performance).to_excel(
                writer, sheet_name="Desempeno_Regimen", index=False
            )
        print(f"Excel generado: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="market-champion-lifecycle")
    parser.add_argument(
        "--input", required=True, help="Excel con Estados_Diarios y Selecciones"
    )
    parser.add_argument("--states-sheet", default="Estados_Diarios")
    parser.add_argument("--selections-sheet", default="Selecciones")
    parser.add_argument("--short-window", type=int, default=3)
    parser.add_argument("--long-window", type=int, default=6)
    parser.add_argument("--minimum-history", type=int, default=4)
    parser.add_argument("--minimum-state-persistence", type=int, default=2)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--output", default="champion_lifecycle_v0961.xlsx")
    parser.set_defaults(handler=_evaluate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_engine.cli import _excel_safe
from market_engine.config import PROJECT_ROOT
from market_engine.evaluation.survival_utility import run_survival_utility_laboratory


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _parse_ints(value: str) -> tuple[int, ...]:
    values = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError("horizons debe contener enteros positivos")
    return values


def _evaluate(args: argparse.Namespace) -> int:
    path = _resolve(args.input)
    if not path.exists():
        raise FileNotFoundError(f"No existe el Excel de entrada: {path}")

    selections = pd.read_excel(path, sheet_name=args.selections_sheet)
    daily_states = pd.read_excel(path, sheet_name=args.states_sheet)
    if selections.empty:
        raise ValueError(f"La hoja {args.selections_sheet} está vacía")

    result = run_survival_utility_laboratory(
        selections,
        daily_states,
        short_window=args.short_window,
        long_window=args.long_window,
        minimum_history=args.minimum_history,
        minimum_state_persistence=args.minimum_state_persistence,
        horizons=args.horizons,
        minimum_completed_spells=args.minimum_completed_spells,
    )

    print("\nSURVIVAL UTILITY CALIBRATION — ESTADO ACTUAL")
    if result.current_utility.empty:
        print("No hubo datos suficientes para calibrar utilidad de supervivencia.")
    else:
        columns = [
            "champion",
            "lifecycle_state",
            "evidence_strength",
            "short_term_authority",
            "medium_term_authority",
            "utility_adjusted_authority",
            "utility_outlook",
            "adverse_state_persistence",
        ]
        print(result.current_utility[columns].to_string(index=False))

    if args.export:
        output = _resolve(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            _excel_safe(result.summary).to_excel(writer, sheet_name="Resumen", index=False)
            _excel_safe(result.current_utility).to_excel(
                writer, sheet_name="Utilidad_Actual", index=False
            )
            _excel_safe(result.utility_history).to_excel(
                writer, sheet_name="Historial_Utilidad", index=False
            )
            _excel_safe(result.evidence_report).to_excel(
                writer, sheet_name="Evidencia_Supervivencia", index=False
            )
            _excel_safe(result.survival_result.current_survival).to_excel(
                writer, sheet_name="Supervivencia_Base", index=False
            )
            _excel_safe(result.survival_result.survival_curves).to_excel(
                writer, sheet_name="Curvas_Supervivencia", index=False
            )
            _excel_safe(result.survival_result.lifecycle_result.current_status).to_excel(
                writer, sheet_name="Ciclo_Actual", index=False
            )
        print(f"Excel generado: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="market-survival-utility")
    parser.add_argument("--input", required=True, help="Excel con Estados_Diarios y Selecciones")
    parser.add_argument("--states-sheet", default="Estados_Diarios")
    parser.add_argument("--selections-sheet", default="Selecciones")
    parser.add_argument("--short-window", type=int, default=3)
    parser.add_argument("--long-window", type=int, default=6)
    parser.add_argument("--minimum-history", type=int, default=4)
    parser.add_argument("--minimum-state-persistence", type=int, default=2)
    parser.add_argument("--horizons", type=_parse_ints, default=(1, 3, 5))
    parser.add_argument("--minimum-completed-spells", type=int, default=3)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--output", default="champion_survival_utility_v0971.xlsx")
    parser.set_defaults(handler=_evaluate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

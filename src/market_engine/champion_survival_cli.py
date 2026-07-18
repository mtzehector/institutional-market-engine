from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_engine.cli import _excel_safe
from market_engine.config import PROJECT_ROOT
from market_engine.evaluation.champion_survival import run_champion_survival_laboratory


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

    result = run_champion_survival_laboratory(
        selections,
        daily_states,
        short_window=args.short_window,
        long_window=args.long_window,
        minimum_history=args.minimum_history,
        minimum_state_persistence=args.minimum_state_persistence,
        horizons=args.horizons,
        minimum_completed_spells=args.minimum_completed_spells,
    )

    print("\nCHAMPION SURVIVAL ENGINE — ESTADO ACTUAL")
    if result.current_survival.empty:
        print("No hubo datos suficientes para estimar supervivencia.")
    else:
        horizon = max(args.horizons)
        columns = [
            "champion",
            "lifecycle_state",
            "state_age",
            f"survival_probability_{horizon}",
            "survival_outlook",
            "survival_confidence_score",
            "survival_adjusted_authority",
        ]
        print(result.current_survival[columns].to_string(index=False))

    if args.export:
        output = _resolve(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            _excel_safe(result.summary).to_excel(writer, sheet_name="Resumen", index=False)
            _excel_safe(result.current_survival).to_excel(
                writer, sheet_name="Supervivencia_Actual", index=False
            )
            _excel_safe(result.survival_history).to_excel(
                writer, sheet_name="Historial_Supervivencia", index=False
            )
            _excel_safe(result.state_duration_spells).to_excel(
                writer, sheet_name="Episodios_Estado", index=False
            )
            _excel_safe(result.survival_curves).to_excel(
                writer, sheet_name="Curvas_Supervivencia", index=False
            )
            _excel_safe(result.lifecycle_result.current_status).to_excel(
                writer, sheet_name="Ciclo_Actual", index=False
            )
        print(f"Excel generado: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="market-champion-survival")
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
    parser.add_argument("--output", default="champion_survival_v097.xlsx")
    parser.set_defaults(handler=_evaluate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_engine.cli import _excel_safe
from market_engine.config import PROJECT_ROOT
from market_engine.evaluation.champion_stress import run_champion_stress_laboratory


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

    result = run_champion_stress_laboratory(
        selections,
        daily_states,
        short_window=args.short_window,
        long_window=args.long_window,
        minimum_history=args.minimum_history,
        minimum_state_persistence=args.minimum_state_persistence,
        minimum_own_episodes=args.minimum_own_episodes,
        minimum_family_episodes=args.minimum_family_episodes,
        trigger_lookback=args.trigger_lookback,
        relapse_horizon=args.relapse_horizon,
    )

    print("\nCHAMPION STRESS ENGINE v0.9.8.2 — ESTADO ACTUAL")
    if result.current_stress.empty:
        print("No hubo datos suficientes para medir estrés.")
    else:
        columns = [
            "champion",
            "lifecycle_state",
            "historical_stress_susceptibility",
            "current_active_stress_score",
            "current_stress_outlook",
            "stress_risk_score",
            "stress_risk_outlook",
            "relapse_rate",
        ]
        print(result.current_stress[columns].to_string(index=False))

    if args.export:
        output = _resolve(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            _excel_safe(result.summary).to_excel(writer, sheet_name="Resumen", index=False)
            _excel_safe(result.current_stress).to_excel(writer, sheet_name="Estres_Actual", index=False)
            _excel_safe(result.active_episodes).to_excel(writer, sheet_name="Episodios_Activos", index=False)
            _excel_safe(result.stress_episodes).to_excel(writer, sheet_name="Episodios_Estres", index=False)
            _excel_safe(result.champion_summary).to_excel(writer, sheet_name="Estres_Campeon", index=False)
            _excel_safe(result.family_summary).to_excel(writer, sheet_name="Estres_Familia", index=False)
            _excel_safe(result.trigger_analysis).to_excel(writer, sheet_name="Coherencia_Disparadores", index=False)
            _excel_safe(result.relapse_analysis).to_excel(writer, sheet_name="Recaida_Censurada", index=False)
            _excel_safe(result.resilience_result.current_resilience).to_excel(
                writer, sheet_name="Resiliencia_Actual", index=False
            )
            _excel_safe(result.resilience_result.lifecycle_result.current_status).to_excel(
                writer, sheet_name="Ciclo_Actual", index=False
            )
        print(f"Excel generado: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="market-champion-stress")
    parser.add_argument("--input", required=True, help="Excel con Estados_Diarios y Selecciones")
    parser.add_argument("--states-sheet", default="Estados_Diarios")
    parser.add_argument("--selections-sheet", default="Selecciones")
    parser.add_argument("--short-window", type=int, default=3)
    parser.add_argument("--long-window", type=int, default=6)
    parser.add_argument("--minimum-history", type=int, default=4)
    parser.add_argument("--minimum-state-persistence", type=int, default=2)
    parser.add_argument("--minimum-own-episodes", type=int, default=3)
    parser.add_argument("--minimum-family-episodes", type=int, default=5)
    parser.add_argument("--trigger-lookback", type=int, default=2)
    parser.add_argument("--relapse-horizon", type=int, default=3)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--output", default="champion_stress_v0982.xlsx")
    parser.set_defaults(handler=_evaluate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

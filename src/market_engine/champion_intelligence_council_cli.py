from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_engine.cli import _excel_safe
from market_engine.config import PROJECT_ROOT
from market_engine.evaluation.champion_intelligence_council import (
    COUNCIL_POLICIES,
    run_champion_intelligence_council,
)


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _parse_ints(value: str) -> tuple[int, ...]:
    values = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError("Debe contener enteros positivos")
    return values


def _evaluate(args: argparse.Namespace) -> int:
    path = _resolve(args.input)
    if not path.exists():
        raise FileNotFoundError(f"No existe el Excel de entrada: {path}")

    selections = pd.read_excel(path, sheet_name=args.selections_sheet)
    daily_states = pd.read_excel(path, sheet_name=args.states_sheet)
    if selections.empty:
        raise ValueError(f"La hoja {args.selections_sheet} está vacía")

    result = run_champion_intelligence_council(
        selections,
        daily_states,
        short_window=args.short_window,
        long_window=args.long_window,
        minimum_history=args.minimum_history,
        minimum_state_persistence=args.minimum_state_persistence,
        horizons=args.horizons,
        minimum_completed_spells=args.minimum_completed_spells,
        minimum_own_episodes=args.minimum_own_episodes,
        minimum_family_episodes=args.minimum_family_episodes,
        trigger_lookback=args.trigger_lookback,
        relapse_horizon=args.relapse_horizon,
        ks=args.neighbors_grid,
        baseline_k=args.baseline_neighbors,
        memory_minimum_history=args.memory_minimum_history,
        calibration_history=args.calibration_history,
        novelty_percentile=args.novelty_percentile,
    )

    print("\nCHAMPION INTELLIGENCE COUNCIL — POLÍTICA BALANCED")
    balanced = result.council_current.loc[result.council_current["policy"] == "BALANCED"]
    columns = [
        "champion",
        "council_authority_score",
        "council_decision",
        "council_confidence",
        "council_agreement",
        "risk_veto",
        "dominant_supporting_engine",
        "dominant_warning_engine",
    ]
    print(balanced[columns].to_string(index=False))

    print("\nCOMPARACIÓN DE POLÍTICAS")
    print(result.policy_comparison.to_string(index=False))

    if args.export:
        output = _resolve(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            _excel_safe(result.summary).to_excel(writer, sheet_name="Resumen", index=False)
            _excel_safe(result.council_current).to_excel(writer, sheet_name="Consejo_Actual", index=False)
            _excel_safe(result.engine_votes).to_excel(writer, sheet_name="Votos_Motores", index=False)
            _excel_safe(result.explanations).to_excel(writer, sheet_name="Explicaciones", index=False)
            _excel_safe(result.disagreements).to_excel(writer, sheet_name="Desacuerdos", index=False)
            _excel_safe(result.policy_comparison).to_excel(writer, sheet_name="Comparacion_Politicas", index=False)
            _excel_safe(result.stress_result.current_stress).to_excel(writer, sheet_name="Estres_Actual", index=False)
            _excel_safe(result.utility_result.current_utility).to_excel(writer, sheet_name="Utilidad_Actual", index=False)
            _excel_safe(result.stress_result.resilience_result.current_resilience).to_excel(
                writer, sheet_name="Resiliencia_Actual", index=False
            )
            _excel_safe(result.utility_result.survival_result.current_survival).to_excel(
                writer, sheet_name="Supervivencia_Actual", index=False
            )
            _excel_safe(result.stress_result.resilience_result.lifecycle_result.current_status).to_excel(
                writer, sheet_name="Ciclo_Actual", index=False
            )
        print(f"Excel generado: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="market-champion-council")
    parser.add_argument("--input", required=True, help="Excel con Estados_Diarios y Selecciones")
    parser.add_argument("--states-sheet", default="Estados_Diarios")
    parser.add_argument("--selections-sheet", default="Selecciones")
    parser.add_argument("--short-window", type=int, default=3)
    parser.add_argument("--long-window", type=int, default=6)
    parser.add_argument("--minimum-history", type=int, default=4)
    parser.add_argument("--minimum-state-persistence", type=int, default=2)
    parser.add_argument("--horizons", type=_parse_ints, default=(1, 3, 5))
    parser.add_argument("--minimum-completed-spells", type=int, default=3)
    parser.add_argument("--minimum-own-episodes", type=int, default=3)
    parser.add_argument("--minimum-family-episodes", type=int, default=5)
    parser.add_argument("--trigger-lookback", type=int, default=2)
    parser.add_argument("--relapse-horizon", type=int, default=3)
    parser.add_argument("--neighbors-grid", type=_parse_ints, default=(3, 5, 7, 10))
    parser.add_argument("--baseline-neighbors", type=int, default=5)
    parser.add_argument("--memory-minimum-history", type=int, default=10)
    parser.add_argument("--calibration-history", type=int, default=6)
    parser.add_argument("--novelty-percentile", type=float, default=0.80)
    parser.add_argument("--policies", default=",".join(COUNCIL_POLICIES), help="Reservado para futuras políticas personalizadas")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--output", default="champion_intelligence_council_v099.xlsx")
    parser.set_defaults(handler=_evaluate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_engine.cli import _excel_safe
from market_engine.config import PROJECT_ROOT
from market_engine.evaluation.market_memory_decision import run_market_memory_decision_laboratory


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _parse_ks(value: str) -> tuple[int, ...]:
    ks = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not ks or min(ks) < 1:
        raise argparse.ArgumentTypeError("neighbors-grid debe contener enteros positivos")
    return ks


def _evaluate(args: argparse.Namespace) -> int:
    path = _resolve(args.input)
    if not path.exists():
        raise FileNotFoundError(f"No existe el Excel de entrada: {path}")
    states = pd.read_excel(path, sheet_name=args.states_sheet)
    selections = pd.read_excel(path, sheet_name=args.selections_sheet)

    result = run_market_memory_decision_laboratory(
        states,
        selections,
        ks=args.neighbors_grid,
        baseline_k=args.baseline_neighbors,
        minimum_history=args.minimum_history,
        calibration_history=args.calibration_history,
        novelty_percentile=args.novelty_percentile,
        similarity_percentile=args.similarity_percentile,
        stability_percentile=args.stability_percentile,
    )

    print("\nMARKET MEMORY DECISION LABORATORY — COMPARACIÓN DE POLÍTICAS")
    if result.policy_comparison.empty:
        print("No hubo fechas suficientes para comparar políticas.")
    else:
        print(result.policy_comparison.to_string(index=False))

    if args.export:
        output = _resolve(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            _excel_safe(result.summary).to_excel(writer, sheet_name="Resumen", index=False)
            _excel_safe(result.policy_comparison).to_excel(writer, sheet_name="Comparacion_Politicas", index=False)
            _excel_safe(result.policy_by_date).to_excel(writer, sheet_name="Politicas_por_Fecha", index=False)
            _excel_safe(result.recommendations).to_excel(writer, sheet_name="Recomendaciones", index=False)
            _excel_safe(result.threshold_audit).to_excel(writer, sheet_name="Auditoria_Umbrales", index=False)
            _excel_safe(result.confidence_result.confidence_validation).to_excel(writer, sheet_name="Validacion_Confianza", index=False)
            _excel_safe(result.confidence_result.sensitivity_k).to_excel(writer, sheet_name="Sensibilidad_K", index=False)
            _excel_safe(result.confidence_result.novelty_dimensions).to_excel(writer, sheet_name="Novedad_Dimensiones", index=False)
        print(f"Excel generado: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="market-memory-decision")
    parser.add_argument("--input", required=True, help="Excel v0.9.3.1 con Estados_Diarios y Selecciones")
    parser.add_argument("--states-sheet", default="Estados_Diarios")
    parser.add_argument("--selections-sheet", default="Selecciones")
    parser.add_argument("--neighbors-grid", type=_parse_ks, default=(3, 5, 7, 10))
    parser.add_argument("--baseline-neighbors", type=int, default=5)
    parser.add_argument("--minimum-history", type=int, default=10)
    parser.add_argument("--calibration-history", type=int, default=6)
    parser.add_argument("--novelty-percentile", type=float, default=0.80)
    parser.add_argument("--similarity-percentile", type=float, default=0.25)
    parser.add_argument("--stability-percentile", type=float, default=0.25)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--output", default="market_memory_decision_v0942.xlsx")
    parser.set_defaults(handler=_evaluate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

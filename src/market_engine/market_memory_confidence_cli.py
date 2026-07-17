from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_engine.cli import _excel_safe
from market_engine.config import PROJECT_ROOT
from market_engine.evaluation.market_memory_confidence import (
    run_market_memory_confidence_laboratory,
)


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _parse_ks(text: str) -> tuple[int, ...]:
    values = tuple(sorted({int(token.strip()) for token in text.split(",") if token.strip()}))
    if not values or any(value < 1 for value in values):
        raise ValueError("--neighbors-grid debe contener enteros positivos")
    return values


def _evaluate(args: argparse.Namespace) -> int:
    path = _resolve(args.input)
    if not path.exists():
        raise FileNotFoundError(f"No existe el Excel de entrada: {path}")

    daily_states = pd.read_excel(path, sheet_name=args.states_sheet)
    selections = pd.read_excel(path, sheet_name=args.selections_sheet)
    result = run_market_memory_confidence_laboratory(
        daily_states,
        selections,
        ks=_parse_ks(args.neighbors_grid),
        baseline_k=args.baseline_neighbors,
        minimum_history=args.minimum_history,
    )

    print("\nMARKET MEMORY CONFIDENCE LABORATORY — RESUMEN")
    print(result.summary.to_string(index=False) if not result.summary.empty else "Sin resultados")

    print("\nVALIDACION POR NIVEL DE CONFIANZA")
    print(
        result.confidence_validation.to_string(index=False)
        if not result.confidence_validation.empty
        else "Sin validacion"
    )

    if args.export:
        output = _resolve(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            _excel_safe(result.summary).to_excel(writer, sheet_name="Resumen", index=False)
            _excel_safe(result.recommendations).to_excel(writer, sheet_name="Recomendaciones", index=False)
            _excel_safe(result.confidence_validation).to_excel(writer, sheet_name="Validacion_Confianza", index=False)
            _excel_safe(result.sensitivity_k).to_excel(writer, sheet_name="Sensibilidad_K", index=False)
            _excel_safe(result.novelty_dimensions).to_excel(writer, sheet_name="Novedad_Dimensiones", index=False)
            _excel_safe(result.abstention_counterfactual).to_excel(writer, sheet_name="Contrafactual_Abstencion", index=False)
        print(f"Excel generado: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="market-memory-confidence")
    parser.add_argument("--input", required=True, help="Excel v0.9.3.1 con Estados_Diarios y Selecciones")
    parser.add_argument("--states-sheet", default="Estados_Diarios")
    parser.add_argument("--selections-sheet", default="Selecciones")
    parser.add_argument("--neighbors-grid", default="3,5,7,10")
    parser.add_argument("--baseline-neighbors", type=int, default=5)
    parser.add_argument("--minimum-history", type=int, default=10)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--output", default="market_memory_confidence_v0941.xlsx")
    parser.set_defaults(handler=_evaluate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

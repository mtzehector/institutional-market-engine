from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_engine.cli import _excel_safe
from market_engine.config import PROJECT_ROOT
from market_engine.evaluation.market_morphology import run_market_morphology_laboratory


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _evaluate(args: argparse.Namespace) -> int:
    input_path = _resolve(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(input_path, sheet_name=args.sheet)
    elif suffix == ".csv":
        frame = pd.read_csv(input_path)
    else:
        raise ValueError("El archivo de entrada debe ser CSV o Excel")

    result = run_market_morphology_laboratory(
        frame,
        smoothing_window=args.smoothing_window,
        minimum_drawdown_pct=args.minimum_drawdown_pct,
    )

    print("\nMARKET MORPHOLOGY M0 — GEOMETRÍA DEL MERCADO")
    print(result.summary.to_string(index=False))
    if result.episodes.empty:
        print("\nNo se detectaron episodios con el umbral indicado.")
    else:
        print("\nEPISODIOS DETECTADOS")
        print(result.episodes.to_string(index=False))

    if args.export:
        output = _resolve(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            _excel_safe(result.summary).to_excel(writer, sheet_name="Resumen", index=False)
            _excel_safe(result.geometry).to_excel(writer, sheet_name="Geometria", index=False)
            _excel_safe(result.episodes).to_excel(writer, sheet_name="Episodios", index=False)
        print(f"\nExcel generado: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="market-morphology")
    parser.add_argument(
        "--input",
        required=True,
        help="CSV o Excel con columnas date, ticker y market_cap",
    )
    parser.add_argument("--sheet", default=0, help="Hoja de Excel; se ignora para CSV")
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--minimum-drawdown-pct", type=float, default=10.0)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--output", default="reports/market_morphology/m0_market_geometry.xlsx")
    parser.set_defaults(handler=_evaluate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

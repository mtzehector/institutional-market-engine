from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_engine.cli import _excel_safe
from market_engine.config import PROJECT_ROOT
from market_engine.evaluation.sample_morphology_fidelity import compare_sample_to_reference


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read(path: Path, sheet: str | None) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path, sheet_name=sheet or 0)


def _evaluate(args: argparse.Namespace) -> int:
    sample_path = _resolve(args.sample)
    reference_path = _resolve(args.reference)
    if not sample_path.exists():
        raise FileNotFoundError(f"No existe la muestra: {sample_path}")
    if not reference_path.exists():
        raise FileNotFoundError(f"No existe la referencia: {reference_path}")

    result = compare_sample_to_reference(
        _read(sample_path, args.sample_sheet),
        _read(reference_path, args.reference_sheet),
        smoothing_window=args.smoothing_window,
        rolling_window=args.rolling_window,
    )

    print("\nM0.1 — REPRESENTATIVE SAMPLE FIDELITY")
    print(result.summary.to_string(index=False))

    if args.export:
        output = _resolve(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            _excel_safe(result.summary).to_excel(writer, sheet_name="Resumen", index=False)
            _excel_safe(result.metrics).to_excel(writer, sheet_name="Metricas", index=False)
            _excel_safe(result.aligned_geometry).to_excel(writer, sheet_name="Geometria_Comparada", index=False)
            _excel_safe(result.rolling_fidelity).to_excel(writer, sheet_name="Fidelidad_Movil", index=False)
        print(f"Excel generado: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="market-sample-fidelity")
    parser.add_argument("--sample", required=True, help="CSV/XLSX con date,ticker,market_cap")
    parser.add_argument("--reference", required=True, help="CSV/XLSX de referencia")
    parser.add_argument("--sample-sheet")
    parser.add_argument("--reference-sheet")
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--rolling-window", type=int, default=20)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--output", default="sample_morphology_fidelity.xlsx")
    parser.set_defaults(handler=_evaluate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

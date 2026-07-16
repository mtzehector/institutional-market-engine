from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from market_engine import __version__
from market_engine.config import PROJECT_ROOT
from market_engine.evaluation.research_snapshot import (
    compare_snapshots,
    create_snapshot,
    export_snapshot,
    load_snapshot_json,
)


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _create(args: argparse.Namespace) -> int:
    inputs = [_resolve(value) for value in args.input]
    snapshot = create_snapshot(
        input_files=inputs,
        label=args.label,
        as_of_date=date.fromisoformat(args.as_of_date),
        engine_version=args.engine_version,
        regime_label=args.regime_label,
        notes=args.notes,
        preferred_sheet=args.sheet,
    )
    paths = export_snapshot(snapshot, _resolve(args.output_dir))

    print("\nFOTOGRAFÍA DE INVESTIGACIÓN")
    print(snapshot.metadata.to_string(index=False))
    print("\nRESUMEN")
    print(snapshot.aggregate_metrics.to_string(index=False))
    for artifact, path in paths.items():
        print(f"{artifact}: {path}")
    return 0


def _compare(args: argparse.Namespace) -> int:
    baseline = load_snapshot_json(_resolve(args.baseline))
    current = load_snapshot_json(_resolve(args.current))
    comparison = compare_snapshots(baseline, current)

    print("\nCOMPARACIÓN DE FOTOGRAFÍAS")
    print(comparison.to_string(index=False))

    if args.export:
        output = _resolve(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            comparison.to_excel(writer, sheet_name="Comparacion", index=False)
        print(f"Excel generado: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="market-snapshot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser(
        "create",
        help="Crea una fotografía reproducible de las métricas actuales",
    )
    create.add_argument(
        "--input",
        action="append",
        required=True,
        help="Archivo Excel o CSV de ranking/memoria. Se puede repetir.",
    )
    create.add_argument("--sheet", default=None)
    create.add_argument("--label", required=True)
    create.add_argument("--as-of-date", default=date.today().isoformat())
    create.add_argument("--engine-version", default=__version__)
    create.add_argument("--regime-label", default="UNSPECIFIED")
    create.add_argument("--notes", default="")
    create.add_argument("--output-dir", default="snapshots")
    create.set_defaults(handler=_create)

    compare = subparsers.add_parser(
        "compare",
        help="Compara dos fotografías JSON conservando métricas nuevas y retiradas",
    )
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--current", required=True)
    compare.add_argument("--export", action="store_true")
    compare.add_argument("--output", default="snapshot_comparison.xlsx")
    compare.set_defaults(handler=_compare)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

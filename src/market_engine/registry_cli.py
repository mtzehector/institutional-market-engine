from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_engine import __version__
from market_engine.config import PROJECT_ROOT
from market_engine.evaluation.model_registry import (
    build_leaderboard,
    build_score_changes,
    export_registry_workbook,
    load_registry,
    record_evaluation,
)


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _record(args: argparse.Namespace) -> int:
    input_path = _resolve(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {input_path}")

    if input_path.suffix.lower() in {".xlsx", ".xls"}:
        ranking = pd.read_excel(input_path, sheet_name=args.sheet)
    else:
        ranking = pd.read_csv(input_path)

    registry_path = _resolve(args.registry)
    result = record_evaluation(
        ranking=ranking,
        registry_path=registry_path,
        model_version=args.model_version,
        run_label=args.run_label,
    )

    output = _resolve(args.output)
    export_registry_workbook(result, output)
    print("\nHALL OF FAME")
    columns = [
        "hall_of_fame_rank",
        "ticker",
        "predictability_score",
        "teacher_level",
        "predictability_grade",
        "observations",
        "directional_accuracy",
        "mean_brier_skill",
        "mean_calibration_error",
        "model_version",
    ]
    print(result.leaderboard[[c for c in columns if c in result.leaderboard.columns]].to_string(index=False))
    print(f"\nRegistro actualizado: {registry_path}")
    print(f"Excel generado: {output}")
    return 0


def _leaderboard(args: argparse.Namespace) -> int:
    registry_path = _resolve(args.registry)
    history = load_registry(registry_path)
    if history.empty:
        raise ValueError(f"El registro está vacío o no existe: {registry_path}")
    leaderboard = build_leaderboard(history)
    changes = build_score_changes(history)
    print(leaderboard.head(args.top).to_string(index=False))
    if args.export:
        output = _resolve(args.output)
        from market_engine.evaluation.model_registry import RegistryResult

        export_registry_workbook(
            RegistryResult(history=history, leaderboard=leaderboard, changes=changes),
            output,
        )
        print(f"Excel generado: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="market-registry")
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser(
        "record", help="Registra un ranking generado por evaluate-models"
    )
    record.add_argument("--input", required=True)
    record.add_argument("--sheet", default="Ranking")
    record.add_argument("--registry", default="data/model_registry.csv")
    record.add_argument("--model-version", default=__version__)
    record.add_argument("--run-label", default="default")
    record.add_argument("--output", default="hall_of_fame_predictibilidad.xlsx")
    record.set_defaults(handler=_record)

    leaderboard = subparsers.add_parser(
        "leaderboard", help="Muestra el Hall of Fame persistente"
    )
    leaderboard.add_argument("--registry", default="data/model_registry.csv")
    leaderboard.add_argument("--top", type=int, default=30)
    leaderboard.add_argument("--export", action="store_true")
    leaderboard.add_argument("--output", default="hall_of_fame_predictibilidad.xlsx")
    leaderboard.set_defaults(handler=_leaderboard)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

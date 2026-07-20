from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_engine.cli import _excel_safe
from market_engine.config import PROJECT_ROOT
from market_engine.evaluation.historical_outcome_distributions import (
    run_historical_outcome_distribution_laboratory,
)


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def _parse_horizons(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Los horizontes deben ser enteros separados por comas") from exc
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("Los horizontes deben ser enteros positivos")
    return values


def _evaluate(args: argparse.Namespace) -> int:
    sample_path = _resolve(args.sample)
    reference_path = _resolve(args.reference)
    if not sample_path.exists():
        raise FileNotFoundError(f"No existe la muestra: {sample_path}")
    if not reference_path.exists():
        raise FileNotFoundError(f"No existe la referencia: {reference_path}")

    result = run_historical_outcome_distribution_laboratory(
        _read(sample_path),
        _read(reference_path),
        horizons=args.horizons,
        smoothing_window=args.smoothing_window,
        spread_window=args.spread_window,
        extreme_zscore=args.extreme_zscore,
        minimum_episode_observations=args.minimum_episode_observations,
    )

    print("\nHISTORICAL OUTCOME DISTRIBUTION LABORATORY v1.0.0a6")
    print(result.summary.to_string(index=False))
    print("\nTop balanced outcomes:")
    print(result.ranking.head(15).to_string(index=False))

    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        _excel_safe(result.summary).to_excel(writer, sheet_name="Resumen", index=False)
        _excel_safe(result.distributions).to_excel(writer, sheet_name="Distribuciones", index=False)
        _excel_safe(result.probabilities).to_excel(writer, sheet_name="Probabilidades", index=False)
        _excel_safe(result.drawdown_probabilities).to_excel(writer, sheet_name="Drawdowns", index=False)
        _excel_safe(result.ranking).to_excel(writer, sheet_name="Ranking", index=False)
        _excel_safe(result.outcome_result.daily_outcomes).to_excel(
            writer, sheet_name="Resultados_Diarios", index=False
        )
        _excel_safe(result.outcome_result.state_outcomes).to_excel(
            writer, sheet_name="Resultados_Estados", index=False
        )
    print(f"Excel generado: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="market-outcome-distributions")
    parser.add_argument("--sample", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--horizons", type=_parse_horizons, default=(1, 5, 10, 20))
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--spread-window", type=int, default=20)
    parser.add_argument("--extreme-zscore", type=float, default=2.0)
    parser.add_argument("--minimum-episode-observations", type=int, default=3)
    parser.add_argument(
        "--output",
        default="reports/market_morphology/historical_outcome_distributions_v100a6.xlsx",
    )
    parser.set_defaults(handler=_evaluate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

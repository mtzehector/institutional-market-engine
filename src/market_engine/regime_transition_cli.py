from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_engine.champion_validation_cli import _load_prediction_universe
from market_engine.cli import _excel_safe
from market_engine.config import PROJECT_ROOT
from market_engine.evaluation.regime_transition import run_regime_transition_laboratory


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _evaluate(args: argparse.Namespace) -> int:
    prediction_features = _load_prediction_universe(args.input, args.sheet)
    result = run_regime_transition_laboratory(
        prediction_features,
        lookback=args.lookback,
        minimum_observations=args.minimum_observations,
        transition_radius=args.transition_radius,
    )

    display = [
        "regime",
        "regime_rank",
        "champion",
        "regime_quality_score",
        "observations",
        "unique_origin_dates",
        "predictability_score",
        "rare_event_f1",
        "balanced_accuracy",
        "macro_f1",
        "mean_brier_skill",
        "mean_calibration_error",
    ]
    print("\nREGIME TRANSITION LABORATORY — MEJOR PILOTO POR RÉGIMEN")
    if result.regime_ranking.empty:
        print("No hubo observaciones suficientes para construir el ranking.")
    else:
        print(
            result.regime_ranking[
                [column for column in display if column in result.regime_ranking.columns]
            ].to_string(index=False)
        )

    print("\nTRANSICIONES DETECTADAS")
    transition_display = [
        "transition_date",
        "from_regime",
        "to_regime",
        "prior_regime_origin_dates",
        "transition_strength",
    ]
    if result.transitions.empty:
        print("No se detectaron transiciones.")
    else:
        print(
            result.transitions[
                [column for column in transition_display if column in result.transitions.columns]
            ].to_string(index=False)
        )

    if args.export:
        output = _resolve(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            _excel_safe(result.daily_states).to_excel(writer, sheet_name="Estados_Diarios", index=False)
            _excel_safe(result.transitions).to_excel(writer, sheet_name="Transiciones", index=False)
            _excel_safe(result.regime_ranking).to_excel(writer, sheet_name="Ranking_por_Regimen", index=False)
            _excel_safe(result.champion_regime_metrics).to_excel(writer, sheet_name="Metricas_Regimen", index=False)
            _excel_safe(result.transition_impact).to_excel(writer, sheet_name="Impacto_Transicion", index=False)
            _excel_safe(result.selections).to_excel(writer, sheet_name="Selecciones", index=False)
        print(f"Excel generado: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="market-regime-transition")
    parser.add_argument("--input", required=True, help="Excel v0.8.4 con Predicciones_Torneo")
    parser.add_argument("--sheet", default="Predicciones_Torneo")
    parser.add_argument("--lookback", type=int, default=12)
    parser.add_argument("--minimum-observations", type=int, default=10)
    parser.add_argument("--transition-radius", type=int, default=3)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--output", default="regime_transition_v093.xlsx")
    parser.set_defaults(handler=_evaluate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

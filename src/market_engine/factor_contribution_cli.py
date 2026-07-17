from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_engine.cli import _excel_safe
from market_engine.config import PROJECT_ROOT
from market_engine.evaluation.factor_contribution import analyze_factor_contributions


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _evaluate(args: argparse.Namespace) -> int:
    input_path = _resolve(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"No existe el Excel evolutivo: {input_path}")

    leaderboard = pd.read_excel(input_path, sheet_name=args.sheet)
    result = analyze_factor_contributions(
        leaderboard,
        score_column=args.score_column,
        top_n=args.top,
    )

    print("\nFACTOR CONTRIBUTION REPORT — RANKING DE FACTORES")
    factor_display = [
        "factor_rank",
        "factor",
        "appearances_top",
        "top_presence_rate",
        "weighted_top_presence",
        "best_rank",
        "best_score",
        "mean_score",
        "mean_validation_rare_event_f1",
        "mean_generalization_gap",
    ]
    print(
        result.factor_ranking[
            [column for column in factor_display if column in result.factor_ranking.columns]
        ].to_string(index=False)
    )

    print("\nSATURACIÓN POR NÚMERO DE FACTORES")
    saturation_display = [
        "factor_count",
        "candidates",
        "mean_score",
        "best_score",
        "incremental_best_score",
        "mean_validation_rare_event_f1",
        "mean_generalization_gap",
    ]
    print(
        result.saturation[
            [column for column in saturation_display if column in result.saturation.columns]
        ].to_string(index=False)
    )

    print("\nMAYORES SINERGIAS ENTRE PARES")
    synergy_display = [
        "pair_signature",
        "cohort_size",
        "score_synergy",
        "beats_both_singles",
        "validation_rare_event_f1_synergy",
        "validation_balanced_accuracy_synergy",
    ]
    print(
        result.pair_synergies[
            [column for column in synergy_display if column in result.pair_synergies.columns]
        ].head(args.top).to_string(index=False)
    )

    if args.export:
        output = _resolve(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            _excel_safe(result.factor_ranking).to_excel(
                writer, sheet_name="Factor_Ranking", index=False
            )
            _excel_safe(result.marginal_contributions).to_excel(
                writer, sheet_name="Aporte_Marginal", index=False
            )
            _excel_safe(result.factor_interactions).to_excel(
                writer, sheet_name="Interacciones", index=False
            )
            _excel_safe(result.pair_synergies).to_excel(
                writer, sheet_name="Sinergias_Pares", index=False
            )
            _excel_safe(result.saturation).to_excel(
                writer, sheet_name="Saturacion", index=False
            )
            _excel_safe(result.top_candidates).to_excel(
                writer, sheet_name="Top_Candidatos", index=False
            )
        print(f"Excel generado: {output}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="market-factor-contribution")
    parser.add_argument("--input", required=True, help="Excel generado por v0.9")
    parser.add_argument("--sheet", default="Leaderboard")
    parser.add_argument("--score-column", default="evolution_score")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--output", default="factor_contribution_v091.xlsx")
    parser.set_defaults(handler=_evaluate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

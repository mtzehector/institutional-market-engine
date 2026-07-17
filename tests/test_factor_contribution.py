from __future__ import annotations

import pandas as pd

from market_engine.evaluation.factor_contribution import analyze_factor_contributions


def _leaderboard() -> pd.DataFrame:
    rows = [
        (1, "MOMENTUM+VOLUME", 2, 10, 82.0, 0.52, 0.56, 0.50, 0.05, 2.5),
        (2, "VOLUME+STRUCTURE", 2, 10, 79.0, 0.49, 0.54, 0.48, 0.06, 2.5),
        (3, "MOMENTUM", 1, 10, 72.0, 0.44, 0.50, 0.45, 0.07, 0.0),
        (4, "VOLUME", 1, 10, 70.0, 0.42, 0.49, 0.44, 0.08, 0.0),
        (5, "STRUCTURE", 1, 10, 68.0, 0.40, 0.48, 0.43, 0.08, 0.0),
        (6, "MOMENTUM+VOLUME+STRUCTURE", 3, 10, 80.0, 0.50, 0.55, 0.49, 0.06, 5.0),
        (7, "MOMENTUM+VOLUME", 2, 20, 76.0, 0.47, 0.53, 0.47, 0.06, 2.5),
        (8, "MOMENTUM", 1, 20, 69.0, 0.41, 0.49, 0.43, 0.08, 0.0),
        (9, "VOLUME", 1, 20, 68.0, 0.40, 0.48, 0.42, 0.08, 0.0),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "evolution_rank",
            "factor_signature",
            "factor_count",
            "cohort_size",
            "evolution_score",
            "validation_rare_event_f1",
            "validation_balanced_accuracy",
            "validation_macro_f1",
            "generalization_gap",
            "complexity_penalty",
        ],
    )


def test_factor_ranking_detects_dominant_factors() -> None:
    result = analyze_factor_contributions(_leaderboard(), top_n=6)
    assert result.factor_ranking.iloc[0]["factor"] in {"MOMENTUM", "VOLUME"}
    assert set(result.factor_ranking["factor"]) == {"MOMENTUM", "VOLUME", "STRUCTURE"}


def test_marginal_gain_compares_candidate_with_parent() -> None:
    result = analyze_factor_contributions(_leaderboard(), top_n=6)
    row = result.marginal_contributions.loc[
        (result.marginal_contributions["factor_signature"] == "MOMENTUM+VOLUME")
        & (result.marginal_contributions["cohort_size"] == 10)
        & (result.marginal_contributions["factor_added"] == "VOLUME")
    ].iloc[0]
    assert row["parent_signature"] == "MOMENTUM"
    assert row["marginal_score_gain"] == 10.0


def test_pair_synergy_identifies_pair_that_beats_singles() -> None:
    result = analyze_factor_contributions(_leaderboard(), top_n=6)
    pair = result.pair_synergies.loc[
        (result.pair_synergies["pair_signature"] == "MOMENTUM+VOLUME")
        & (result.pair_synergies["cohort_size"] == 10)
    ].iloc[0]
    assert bool(pair["beats_both_singles"])
    assert pair["score_synergy"] == 11.0


def test_saturation_reports_incremental_best_score() -> None:
    result = analyze_factor_contributions(_leaderboard(), top_n=6)
    saturation = result.saturation.set_index("factor_count")
    assert saturation.loc[2, "incremental_best_score"] == 10.0
    assert saturation.loc[3, "incremental_best_score"] == -2.0

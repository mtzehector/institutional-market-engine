from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorContributionResult:
    factor_ranking: pd.DataFrame
    marginal_contributions: pd.DataFrame
    factor_interactions: pd.DataFrame
    pair_synergies: pd.DataFrame
    saturation: pd.DataFrame
    top_candidates: pd.DataFrame


def _factor_tuple(signature: object) -> tuple[str, ...]:
    factors = tuple(
        token.strip().upper()
        for token in str(signature).split("+")
        if token.strip()
    )
    if not factors:
        raise ValueError(f"Firma de factores inválida: {signature!r}")
    return tuple(dict.fromkeys(factors))


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _prepare(leaderboard: pd.DataFrame, score_column: str) -> pd.DataFrame:
    required = {"factor_signature", "factor_count", "cohort_size", score_column}
    missing = required - set(leaderboard.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el leaderboard: {sorted(missing)}")
    if leaderboard.empty:
        raise ValueError("El leaderboard está vacío")

    data = leaderboard.copy().reset_index(drop=True)
    data["factors"] = data["factor_signature"].map(_factor_tuple)
    data["factor_set"] = data["factors"].map(frozenset)
    data["factor_count"] = pd.to_numeric(data["factor_count"], errors="raise").astype(int)
    data["cohort_size"] = pd.to_numeric(data["cohort_size"], errors="raise").astype(int)
    data[score_column] = pd.to_numeric(data[score_column], errors="coerce")

    rank_column = "evolution_rank" if "evolution_rank" in data.columns else None
    if rank_column is None:
        data = data.sort_values(score_column, ascending=False).reset_index(drop=True)
        data["analysis_rank"] = np.arange(1, len(data) + 1)
    else:
        data["analysis_rank"] = pd.to_numeric(data[rank_column], errors="coerce")
        fallback = pd.Series(np.arange(1, len(data) + 1), index=data.index)
        data["analysis_rank"] = data["analysis_rank"].fillna(fallback).astype(int)
    return data


def _factor_ranking(data: pd.DataFrame, score_column: str, top_n: int) -> pd.DataFrame:
    top = data.nsmallest(min(top_n, len(data)), "analysis_rank")
    all_factors = sorted({factor for factors in data["factors"] for factor in factors})
    rows: list[dict[str, object]] = []

    for factor in all_factors:
        full_sample = data.loc[data["factor_set"].map(lambda values: factor in values)]
        top_sample = top.loc[top["factor_set"].map(lambda values: factor in values)]
        weights = 1.0 / np.log2(top_sample["analysis_rank"].astype(float) + 1.0)
        rows.append(
            {
                "factor": factor,
                "appearances_all": int(len(full_sample)),
                "appearances_top": int(len(top_sample)),
                "top_presence_rate": float(len(top_sample) / len(top)) if len(top) else np.nan,
                "weighted_top_presence": float(weights.sum()),
                "best_rank": int(full_sample["analysis_rank"].min()),
                "best_score": float(full_sample[score_column].max()),
                "mean_score": float(full_sample[score_column].mean()),
                "median_score": float(full_sample[score_column].median()),
                "mean_validation_rare_event_f1": float(
                    _numeric(full_sample, "validation_rare_event_f1").mean()
                ),
                "mean_validation_balanced_accuracy": float(
                    _numeric(full_sample, "validation_balanced_accuracy").mean()
                ),
                "mean_generalization_gap": float(
                    _numeric(full_sample, "generalization_gap").mean()
                ),
            }
        )

    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return ranking
    ranking = ranking.sort_values(
        ["weighted_top_presence", "appearances_top", "best_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    ranking.insert(0, "factor_rank", np.arange(1, len(ranking) + 1))
    return ranking


def _marginal_contributions(data: pd.DataFrame, score_column: str) -> pd.DataFrame:
    lookup = {
        (row.factor_set, int(row.cohort_size)): row
        for row in data.itertuples(index=False)
    }
    rows: list[dict[str, object]] = []

    for candidate in data.itertuples(index=False):
        factors = tuple(candidate.factors)
        if len(factors) < 2:
            continue
        for added_factor in factors:
            parent_set = frozenset(set(factors) - {added_factor})
            parent = lookup.get((parent_set, int(candidate.cohort_size)))
            if parent is None:
                continue
            row: dict[str, object] = {
                "candidate_id": getattr(candidate, "candidate_id", None),
                "factor_signature": candidate.factor_signature,
                "cohort_size": int(candidate.cohort_size),
                "factor_added": added_factor,
                "parent_signature": "+".join(sorted(parent_set)),
                "parent_factor_count": len(parent_set),
                "candidate_score": float(getattr(candidate, score_column)),
                "parent_score": float(getattr(parent, score_column)),
                "marginal_score_gain": float(
                    getattr(candidate, score_column) - getattr(parent, score_column)
                ),
            }
            for metric in (
                "validation_rare_event_f1",
                "validation_balanced_accuracy",
                "validation_macro_f1",
                "validation_mean_brier_skill",
                "validation_mean_calibration_error",
                "generalization_gap",
            ):
                candidate_value = getattr(candidate, metric, np.nan)
                parent_value = getattr(parent, metric, np.nan)
                row[f"marginal_{metric}"] = (
                    float(candidate_value - parent_value)
                    if pd.notna(candidate_value) and pd.notna(parent_value)
                    else np.nan
                )
            rows.append(row)

    return pd.DataFrame(rows)


def _factor_interactions(marginal: pd.DataFrame) -> pd.DataFrame:
    if marginal.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for row in marginal.itertuples(index=False):
        context = _factor_tuple(row.parent_signature)
        for context_factor in context:
            rows.append(
                {
                    "factor_added": row.factor_added,
                    "context_factor": context_factor,
                    "cohort_size": int(row.cohort_size),
                    "marginal_score_gain": row.marginal_score_gain,
                    "marginal_validation_rare_event_f1": getattr(
                        row, "marginal_validation_rare_event_f1", np.nan
                    ),
                    "marginal_validation_balanced_accuracy": getattr(
                        row, "marginal_validation_balanced_accuracy", np.nan
                    ),
                }
            )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail
    return (
        detail.groupby(["factor_added", "context_factor"], as_index=False)
        .agg(
            observations=("marginal_score_gain", "size"),
            mean_marginal_score_gain=("marginal_score_gain", "mean"),
            median_marginal_score_gain=("marginal_score_gain", "median"),
            positive_gain_rate=("marginal_score_gain", lambda values: (values > 0).mean()),
            mean_marginal_rare_event_f1=(
                "marginal_validation_rare_event_f1",
                "mean",
            ),
            mean_marginal_balanced_accuracy=(
                "marginal_validation_balanced_accuracy",
                "mean",
            ),
        )
        .sort_values(
            ["mean_marginal_score_gain", "positive_gain_rate"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )


def _pair_synergies(data: pd.DataFrame, score_column: str) -> pd.DataFrame:
    lookup = {
        (row.factor_set, int(row.cohort_size)): row
        for row in data.itertuples(index=False)
    }
    rows: list[dict[str, object]] = []
    pair_rows = data.loc[data["factor_count"] == 2]

    for pair in pair_rows.itertuples(index=False):
        factor_a, factor_b = tuple(pair.factors)
        single_a = lookup.get((frozenset({factor_a}), int(pair.cohort_size)))
        single_b = lookup.get((frozenset({factor_b}), int(pair.cohort_size)))
        if single_a is None or single_b is None:
            continue
        single_score_mean = 0.5 * (
            float(getattr(single_a, score_column)) + float(getattr(single_b, score_column))
        )
        pair_score = float(getattr(pair, score_column))
        row: dict[str, object] = {
            "factor_a": factor_a,
            "factor_b": factor_b,
            "pair_signature": pair.factor_signature,
            "cohort_size": int(pair.cohort_size),
            "pair_score": pair_score,
            "single_score_mean": single_score_mean,
            "score_synergy": pair_score - single_score_mean,
            "beats_both_singles": pair_score
            > max(float(getattr(single_a, score_column)), float(getattr(single_b, score_column))),
        }
        for metric in (
            "validation_rare_event_f1",
            "validation_balanced_accuracy",
            "validation_macro_f1",
        ):
            pair_value = getattr(pair, metric, np.nan)
            a_value = getattr(single_a, metric, np.nan)
            b_value = getattr(single_b, metric, np.nan)
            row[f"{metric}_synergy"] = (
                float(pair_value - 0.5 * (a_value + b_value))
                if pd.notna(pair_value) and pd.notna(a_value) and pd.notna(b_value)
                else np.nan
            )
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["score_synergy", "validation_rare_event_f1_synergy"],
        ascending=[False, False],
    ).reset_index(drop=True)


def _saturation(data: pd.DataFrame, score_column: str, top_n: int) -> pd.DataFrame:
    top_ids = set(data.nsmallest(min(top_n, len(data)), "analysis_rank").index)
    working = data.copy()
    working["is_top_candidate"] = working.index.isin(top_ids)
    saturation = (
        working.groupby("factor_count", as_index=False)
        .agg(
            candidates=(score_column, "size"),
            mean_score=(score_column, "mean"),
            median_score=(score_column, "median"),
            best_score=(score_column, "max"),
            score_std=(score_column, "std"),
            top_candidates=("is_top_candidate", "sum"),
            mean_validation_rare_event_f1=("validation_rare_event_f1", "mean"),
            mean_validation_balanced_accuracy=("validation_balanced_accuracy", "mean"),
            mean_generalization_gap=("generalization_gap", "mean"),
            mean_complexity_penalty=("complexity_penalty", "mean"),
        )
        .sort_values("factor_count")
        .reset_index(drop=True)
    )
    saturation["incremental_best_score"] = saturation["best_score"].diff()
    saturation["incremental_mean_score"] = saturation["mean_score"].diff()
    return saturation


def analyze_factor_contributions(
    leaderboard: pd.DataFrame,
    *,
    score_column: str = "evolution_score",
    top_n: int = 25,
) -> FactorContributionResult:
    if top_n < 1:
        raise ValueError("top_n debe ser mayor o igual que 1")
    data = _prepare(leaderboard, score_column)
    marginal = _marginal_contributions(data, score_column)
    top_candidates = data.nsmallest(min(top_n, len(data)), "analysis_rank").drop(
        columns=["factors", "factor_set"], errors="ignore"
    )
    return FactorContributionResult(
        factor_ranking=_factor_ranking(data, score_column, top_n),
        marginal_contributions=marginal,
        factor_interactions=_factor_interactions(marginal),
        pair_synergies=_pair_synergies(data, score_column),
        saturation=_saturation(data, score_column, top_n),
        top_candidates=top_candidates.reset_index(drop=True),
    )

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd

from market_engine.evaluation.model_evaluation import evaluate_predictions
from market_engine.evaluation.universe_tournament import add_strategy_scores


BASE_FACTORS = (
    "MOMENTUM",
    "VOLUME",
    "SMART_MONEY",
    "MODEL_CONFIDENCE",
    "LOW_VOLATILITY",
    "STRUCTURE",
)


@dataclass(frozen=True)
class EvolutionaryUniverseResult:
    leaderboard: pd.DataFrame
    candidate_metrics: pd.DataFrame
    hall_of_fame: pd.DataFrame
    candidate_definitions: pd.DataFrame
    selections: pd.DataFrame
    daily_diagnostics: pd.DataFrame


def _parse_factor_tuple(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        factors = tuple(token.strip().upper() for token in value.split("+") if token.strip())
    else:
        factors = tuple(str(token).strip().upper() for token in value if str(token).strip())
    unknown = sorted(set(factors) - set(BASE_FACTORS))
    if unknown:
        raise ValueError(f"Factores desconocidos: {unknown}")
    if not factors:
        raise ValueError("La combinación debe contener al menos un factor")
    return tuple(dict.fromkeys(factors))


def generate_candidate_definitions(
    cohort_sizes: Iterable[int] = (5, 10, 20),
    max_factors: int = 3,
) -> pd.DataFrame:
    sizes = sorted({int(value) for value in cohort_sizes if int(value) > 0})
    if not sizes:
        raise ValueError("Debe existir al menos un tamaño de cohorte positivo")
    if max_factors < 1 or max_factors > len(BASE_FACTORS):
        raise ValueError("max_factors debe estar entre 1 y el número de factores disponibles")

    rows: list[dict[str, object]] = []
    candidate_id = 0
    for factor_count in range(1, max_factors + 1):
        for factor_tuple in combinations(BASE_FACTORS, factor_count):
            signature = "+".join(factor_tuple)
            for size in sizes:
                candidate_id += 1
                rows.append(
                    {
                        "candidate_id": f"C{candidate_id:04d}",
                        "factor_signature": signature,
                        "factor_count": factor_count,
                        "cohort_size": size,
                    }
                )
    return pd.DataFrame(rows)


def add_evolutionary_factor_scores(frame: pd.DataFrame) -> pd.DataFrame:
    scored = add_strategy_scores(frame)
    if scored.empty:
        return scored

    # STRUCTURE already exists inside the v0.8.4 hybrid calculation, but it was
    # not exposed as an independent strategy. Rebuild it causally per date.
    rows: list[pd.DataFrame] = []
    for _, daily in scored.groupby("origin_date", sort=True):
        ranked = daily.copy()
        close_vwap = pd.to_numeric(ranked["close_vs_vwap_pct"], errors="coerce")
        close_position = pd.to_numeric(ranked["close_position_pct"], errors="coerce")
        structure = (
            0.55 * close_vwap.rank(pct=True, method="average").fillna(0.5)
            + 0.45 * close_position.rank(pct=True, method="average").fillna(0.5)
        )
        ranked["score_STRUCTURE"] = 100.0 * structure
        rows.append(ranked)
    return pd.concat(rows, ignore_index=True)


def score_candidates(
    scored: pd.DataFrame,
    definitions: pd.DataFrame,
) -> pd.DataFrame:
    if scored.empty or definitions.empty:
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    for _, definition in definitions.iterrows():
        factors = _parse_factor_tuple(str(definition["factor_signature"]))
        score_columns = [f"score_{factor}" for factor in factors]
        missing = [column for column in score_columns if column not in scored.columns]
        if missing:
            raise ValueError(f"Faltan scores para el candidato: {missing}")

        candidate = scored.copy()
        candidate["candidate_score"] = candidate[score_columns].mean(axis=1)
        candidate["candidate_rank"] = candidate.groupby("origin_date")[
            "candidate_score"
        ].rank(ascending=False, method="first")
        candidate = candidate.loc[
            candidate["candidate_rank"] <= int(definition["cohort_size"])
        ].copy()
        candidate["candidate_id"] = definition["candidate_id"]
        candidate["factor_signature"] = definition["factor_signature"]
        candidate["factor_count"] = int(definition["factor_count"])
        candidate["cohort_size"] = int(definition["cohort_size"])
        rows.append(candidate)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _evaluate_sample(sample: pd.DataFrame, label: str) -> pd.DataFrame:
    result = evaluate_predictions(sample, ticker=label)
    summary = result.summary.rename(columns={"ticker": "candidate_id"}).copy()
    summary["unique_tickers"] = int(sample["ticker"].nunique())
    summary["unique_origin_dates"] = int(sample["origin_date"].nunique())
    summary["candidate_score_mean"] = float(sample["candidate_score"].mean())
    summary["candidate_score_std"] = float(sample["candidate_score"].std(ddof=0))
    summary["actual_gap_pct_mean"] = float(sample["actual_gap_pct"].mean())
    summary["actual_gap_pct_median"] = float(sample["actual_gap_pct"].median())
    return summary


def evaluate_candidates(
    selections: pd.DataFrame,
    validation_fraction: float = 0.30,
    minimum_validation_dates: int = 8,
) -> pd.DataFrame:
    if selections.empty:
        return pd.DataFrame()
    if not 0.10 <= validation_fraction <= 0.50:
        raise ValueError("validation_fraction debe estar entre 0.10 y 0.50")

    all_dates = sorted(pd.to_datetime(selections["origin_date"]).dropna().unique())
    validation_count = max(minimum_validation_dates, int(np.ceil(len(all_dates) * validation_fraction)))
    validation_count = min(validation_count, max(1, len(all_dates) - 1))
    split_date = pd.Timestamp(all_dates[-validation_count])

    rows: list[pd.DataFrame] = []
    for candidate_id, sample in selections.groupby("candidate_id", sort=False):
        discovery = sample.loc[pd.to_datetime(sample["origin_date"]) < split_date]
        validation = sample.loc[pd.to_datetime(sample["origin_date"]) >= split_date]
        if discovery.empty or validation.empty:
            continue

        discovery_summary = _evaluate_sample(discovery, str(candidate_id)).add_prefix("discovery_")
        validation_summary = _evaluate_sample(validation, str(candidate_id)).add_prefix("validation_")
        row = pd.concat(
            [discovery_summary.reset_index(drop=True), validation_summary.reset_index(drop=True)],
            axis=1,
        )
        first = sample.iloc[0]
        row["candidate_id"] = candidate_id
        row["factor_signature"] = first["factor_signature"]
        row["factor_count"] = int(first["factor_count"])
        row["cohort_size"] = int(first["cohort_size"])
        row["split_date"] = split_date
        rows.append(row)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _rank_component(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() <= 1:
        return pd.Series(0.5, index=series.index, dtype=float)
    return values.rank(
        pct=True,
        ascending=higher_is_better,
        method="average",
    ).fillna(0.0)


def build_evolutionary_leaderboard(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics.copy()
    board = metrics.copy()

    validation_criteria = {
        "validation_rare_event_f1": True,
        "validation_balanced_accuracy": True,
        "validation_macro_f1": True,
        "validation_mean_brier_skill": True,
        "validation_mean_calibration_error": False,
    }
    board["validation_quality_score"] = 0.0
    for column, higher_is_better in validation_criteria.items():
        component = _rank_component(board[column], higher_is_better)
        board[f"component_{column}"] = component
        board["validation_quality_score"] += 20.0 * component

    discovery_score = (
        0.40 * pd.to_numeric(board["discovery_rare_event_f1"], errors="coerce").fillna(0.0)
        + 0.30 * pd.to_numeric(board["discovery_balanced_accuracy"], errors="coerce").fillna(0.0)
        + 0.30 * pd.to_numeric(board["discovery_macro_f1"], errors="coerce").fillna(0.0)
    )
    validation_score = (
        0.40 * pd.to_numeric(board["validation_rare_event_f1"], errors="coerce").fillna(0.0)
        + 0.30 * pd.to_numeric(board["validation_balanced_accuracy"], errors="coerce").fillna(0.0)
        + 0.30 * pd.to_numeric(board["validation_macro_f1"], errors="coerce").fillna(0.0)
    )
    board["generalization_gap"] = (discovery_score - validation_score).abs()
    board["complexity_penalty"] = 2.5 * (pd.to_numeric(board["factor_count"], errors="coerce") - 1.0)
    board["generalization_penalty"] = 35.0 * board["generalization_gap"].clip(lower=0.0)
    board["evolution_score"] = (
        board["validation_quality_score"]
        - board["complexity_penalty"]
        - board["generalization_penalty"]
    )

    board = board.sort_values(
        [
            "evolution_score",
            "validation_rare_event_f1",
            "validation_balanced_accuracy",
            "factor_count",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    board.insert(0, "evolution_rank", np.arange(1, len(board) + 1))
    return board


def build_diverse_hall_of_fame(
    leaderboard: pd.DataFrame,
    limit: int = 12,
    maximum_factor_overlap: float = 0.75,
) -> pd.DataFrame:
    if leaderboard.empty:
        return leaderboard.copy()
    selected_indices: list[int] = []
    selected_factor_sets: list[set[str]] = []

    for index, row in leaderboard.iterrows():
        factors = set(_parse_factor_tuple(str(row["factor_signature"])))
        sufficiently_different = True
        for prior in selected_factor_sets:
            union = factors | prior
            overlap = len(factors & prior) / len(union) if union else 1.0
            if overlap > maximum_factor_overlap and int(row["cohort_size"]) == int(
                leaderboard.loc[selected_indices[-1], "cohort_size"]
            ):
                sufficiently_different = False
                break
        if sufficiently_different:
            selected_indices.append(index)
            selected_factor_sets.append(factors)
        if len(selected_indices) >= limit:
            break

    hall = leaderboard.loc[selected_indices].copy().reset_index(drop=True)
    hall.insert(0, "hall_of_fame_rank", np.arange(1, len(hall) + 1))
    return hall


def evolutionary_daily_diagnostics(selections: pd.DataFrame) -> pd.DataFrame:
    if selections.empty:
        return pd.DataFrame()
    return (
        selections.groupby(["candidate_id", "origin_date"])
        .agg(
            selected_tickers=("ticker", "nunique"),
            candidate_score_mean=("candidate_score", "mean"),
            candidate_score_std=("candidate_score", "std"),
            actual_rare_events=("actual_direction", lambda values: (values != "SIN_GAP").sum()),
            predicted_rare_events=("predicted_direction", lambda values: (values != "SIN_GAP").sum()),
        )
        .reset_index()
    )


def run_evolutionary_universe(
    prediction_features: pd.DataFrame,
    cohort_sizes: Iterable[int] = (5, 10, 20),
    max_factors: int = 3,
    validation_fraction: float = 0.30,
    hall_of_fame_size: int = 12,
) -> EvolutionaryUniverseResult:
    scored = add_evolutionary_factor_scores(prediction_features)
    definitions = generate_candidate_definitions(cohort_sizes, max_factors)
    selections = score_candidates(scored, definitions)
    metrics = evaluate_candidates(selections, validation_fraction)
    leaderboard = build_evolutionary_leaderboard(metrics)
    hall_of_fame = build_diverse_hall_of_fame(leaderboard, hall_of_fame_size)
    diagnostics = evolutionary_daily_diagnostics(selections)
    return EvolutionaryUniverseResult(
        leaderboard=leaderboard,
        candidate_metrics=metrics,
        hall_of_fame=hall_of_fame,
        candidate_definitions=definitions,
        selections=selections,
        daily_diagnostics=diagnostics,
    )

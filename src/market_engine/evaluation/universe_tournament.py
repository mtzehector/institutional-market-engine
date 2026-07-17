from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from market_engine.evaluation.model_evaluation import evaluate_predictions


STRATEGIES = (
    "MOMENTUM",
    "VOLUME",
    "SMART_MONEY",
    "MODEL_CONFIDENCE",
    "LOW_VOLATILITY",
    "HYBRID",
)


@dataclass(frozen=True)
class UniverseTournamentResult:
    leaderboard: pd.DataFrame
    strategy_metrics: pd.DataFrame
    selections: pd.DataFrame
    predictions: pd.DataFrame
    daily_diagnostics: pd.DataFrame


def _percentile(series: pd.Series, *, ascending: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() <= 1:
        return pd.Series(0.5, index=series.index, dtype=float)
    return numeric.rank(pct=True, method="average", ascending=ascending).fillna(0.5)


def _required_columns() -> set[str]:
    return {
        "ticker",
        "origin_date",
        "target_date",
        "actual_gap_pct",
        "actual_gap_up",
        "actual_gap_down",
        "actual_direction",
        "probability_up",
        "probability_down",
        "probability_no_gap",
        "predicted_direction",
        "correct_direction",
        "return_1d_pct",
        "return_3d_pct",
        "return_5d_pct",
        "relative_volume",
        "close_position_pct",
        "close_vs_vwap_pct",
        "smart_money_pct",
        "smart_money_slope_5d",
        "atr_pct",
    }


def add_strategy_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-safe cross-sectional strategy scores per origin date."""
    missing = _required_columns() - set(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas para el torneo: {sorted(missing)}")
    if frame.empty:
        return frame.copy()

    data = frame.copy()
    data["origin_date"] = pd.to_datetime(data["origin_date"])
    rows: list[pd.DataFrame] = []

    for _, daily in data.groupby("origin_date", sort=True):
        ranked = daily.copy()
        momentum = (
            0.20 * _percentile(ranked["return_1d_pct"])
            + 0.35 * _percentile(ranked["return_3d_pct"])
            + 0.45 * _percentile(ranked["return_5d_pct"])
        )
        volume = (
            0.70 * _percentile(ranked["relative_volume"])
            + 0.30 * _percentile(ranked["close_position_pct"])
        )
        smart_money = (
            0.60 * _percentile(ranked["smart_money_pct"])
            + 0.40 * _percentile(ranked["smart_money_slope_5d"])
        )
        confidence = ranked[
            ["probability_up", "probability_down", "probability_no_gap"]
        ].max(axis=1)
        model_confidence = _percentile(confidence)
        low_volatility = _percentile(ranked["atr_pct"], ascending=False)
        structure = (
            0.55 * _percentile(ranked["close_vs_vwap_pct"])
            + 0.45 * _percentile(ranked["close_position_pct"])
        )

        atr_rank = _percentile(ranked["atr_pct"])
        overextension = np.clip((atr_rank - 0.85) / 0.15, 0.0, 1.0)

        ranked["score_MOMENTUM"] = 100.0 * momentum
        ranked["score_VOLUME"] = 100.0 * volume
        ranked["score_SMART_MONEY"] = 100.0 * smart_money
        ranked["score_MODEL_CONFIDENCE"] = 100.0 * model_confidence
        ranked["score_LOW_VOLATILITY"] = 100.0 * low_volatility
        ranked["score_HYBRID"] = 100.0 * (
            0.25 * momentum
            + 0.20 * volume
            + 0.20 * smart_money
            + 0.15 * model_confidence
            + 0.10 * low_volatility
            + 0.10 * structure
            - 0.05 * overextension
        )
        ranked["overextension_penalty_tournament"] = overextension

        for strategy in STRATEGIES:
            ranked[f"rank_{strategy}"] = ranked[f"score_{strategy}"].rank(
                ascending=False, method="first"
            ).astype(int)
        rows.append(ranked)

    return pd.concat(rows, ignore_index=True)


def build_tournament_selections(
    scored: pd.DataFrame,
    cohort_sizes: Iterable[int] = (5, 10, 20),
) -> pd.DataFrame:
    sizes = sorted({int(value) for value in cohort_sizes if int(value) > 0})
    if not sizes:
        raise ValueError("Debe existir al menos un tamaño de cohorte positivo")

    rows: list[pd.DataFrame] = []
    for strategy in STRATEGIES:
        for size in sizes:
            selected = scored.loc[scored[f"rank_{strategy}"] <= size].copy()
            selected["strategy"] = strategy
            selected["cohort_size"] = size
            selected["strategy_score"] = selected[f"score_{strategy}"]
            selected["strategy_rank"] = selected[f"rank_{strategy}"]
            selected["tournament_entry"] = f"{strategy}_TOP_{size}"
            rows.append(selected)

    universe = scored.copy()
    universe["strategy"] = "UNIVERSE"
    universe["cohort_size"] = universe.groupby("origin_date")["ticker"].transform("nunique")
    universe["strategy_score"] = np.nan
    universe["strategy_rank"] = np.nan
    universe["tournament_entry"] = "UNIVERSE"
    rows.append(universe)
    return pd.concat(rows, ignore_index=True)


def evaluate_tournament(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for entry, sample in predictions.groupby("tournament_entry", sort=False):
        result = evaluate_predictions(sample, ticker=entry)
        summary = result.summary.rename(columns={"ticker": "tournament_entry"}).copy()
        summary["strategy"] = sample["strategy"].iloc[0]
        summary["cohort_size"] = int(sample["cohort_size"].median())
        summary["unique_tickers"] = int(sample["ticker"].nunique())
        summary["unique_origin_dates"] = int(sample["origin_date"].nunique())
        summary["strategy_score_mean"] = float(sample["strategy_score"].mean()) if sample["strategy_score"].notna().any() else np.nan
        summary["strategy_score_std"] = float(sample["strategy_score"].std(ddof=0)) if sample["strategy_score"].notna().any() else np.nan
        summary["actual_gap_pct_mean"] = float(sample["actual_gap_pct"].mean())
        summary["actual_gap_pct_median"] = float(sample["actual_gap_pct"].median())
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_leaderboard(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics.copy()
    board = metrics.copy()
    competitors = board["strategy"] != "UNIVERSE"

    criteria = {
        "rare_event_f1": True,
        "balanced_accuracy": True,
        "macro_f1": True,
        "mean_brier_skill": True,
        "mean_calibration_error": False,
    }
    board["tournament_score"] = 0.0
    for column, higher_is_better in criteria.items():
        values = pd.to_numeric(board[column], errors="coerce")
        if values.notna().sum() <= 1:
            component = pd.Series(0.5, index=board.index)
        else:
            component = values.rank(
                pct=True,
                ascending=higher_is_better,
                method="average",
            ).fillna(0.0)
        board[f"tournament_component_{column}"] = component
        board["tournament_score"] += 20.0 * component

    board["is_competitor"] = competitors
    board = board.sort_values(
        ["is_competitor", "tournament_score", "rare_event_f1", "balanced_accuracy"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    board.insert(0, "tournament_rank", np.arange(1, len(board) + 1))
    return board


def tournament_daily_diagnostics(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    score_columns = [f"score_{strategy}" for strategy in STRATEGIES]
    aggregations: dict[str, tuple[str, str]] = {
        "universe_size": ("ticker", "nunique"),
    }
    for column in score_columns:
        aggregations[f"{column}_mean"] = (column, "mean")
        aggregations[f"{column}_std"] = (column, "std")
    return scored.groupby("origin_date").agg(**aggregations).reset_index()


def run_universe_tournament(
    prediction_features: pd.DataFrame,
    cohort_sizes: Iterable[int] = (5, 10, 20),
) -> UniverseTournamentResult:
    scored = add_strategy_scores(prediction_features)
    selections = build_tournament_selections(scored, cohort_sizes)
    metrics = evaluate_tournament(selections)
    leaderboard = build_leaderboard(metrics)
    diagnostics = tournament_daily_diagnostics(scored)

    selection_columns = [
        "origin_date",
        "ticker",
        *[f"score_{strategy}" for strategy in STRATEGIES],
        *[f"rank_{strategy}" for strategy in STRATEGIES],
        "overextension_penalty_tournament",
    ]
    return UniverseTournamentResult(
        leaderboard=leaderboard,
        strategy_metrics=metrics,
        selections=scored[[column for column in selection_columns if column in scored.columns]].copy(),
        predictions=selections,
        daily_diagnostics=diagnostics,
    )

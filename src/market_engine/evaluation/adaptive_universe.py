from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from market_engine.evaluation.model_evaluation import evaluate_predictions


@dataclass(frozen=True)
class AdaptiveUniverseResult:
    selections: pd.DataFrame
    cohort_predictions: pd.DataFrame
    cohort_metrics: pd.DataFrame
    daily_diagnostics: pd.DataFrame


def _percentile(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() <= 1:
        return pd.Series(0.5, index=series.index, dtype=float)
    return numeric.rank(pct=True, method="average").fillna(0.5)


def add_bullish_score(frame: pd.DataFrame) -> pd.DataFrame:
    """Build a leakage-safe cross-sectional bullish score for each origin date.

    Every input row must represent information available at the close of
    ``origin_date``. Percentile ranking is performed independently per date so
    unlike price scales and volatility levels remain comparable.
    """
    required = {
        "ticker",
        "origin_date",
        "return_1d_pct",
        "return_3d_pct",
        "return_5d_pct",
        "close_position_pct",
        "close_vs_vwap_pct",
        "relative_volume",
        "smart_money_pct",
        "smart_money_slope_5d",
        "qqq_return_1d_pct",
        "atr_pct",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas para construir la cohorte: {sorted(missing)}")

    data = frame.copy()
    data["origin_date"] = pd.to_datetime(data["origin_date"])

    rows: list[pd.DataFrame] = []
    for _, daily in data.groupby("origin_date", sort=True):
        ranked = daily.copy()
        ranked["momentum_component"] = (
            0.20 * _percentile(ranked["return_1d_pct"])
            + 0.35 * _percentile(ranked["return_3d_pct"])
            + 0.45 * _percentile(ranked["return_5d_pct"])
        )
        ranked["participation_component"] = (
            0.65 * _percentile(ranked["relative_volume"])
            + 0.35 * _percentile(ranked["close_position_pct"])
        )
        ranked["institutional_component"] = (
            0.55 * _percentile(ranked["smart_money_pct"])
            + 0.45 * _percentile(ranked["smart_money_slope_5d"])
        )
        ranked["structure_component"] = (
            0.60 * _percentile(ranked["close_vs_vwap_pct"])
            + 0.40 * _percentile(ranked["qqq_return_1d_pct"])
        )

        # Penalize extreme ATR values because the strongest one-day movers can
        # already be exhausted. The penalty is deliberately mild and auditable.
        atr_rank = _percentile(ranked["atr_pct"])
        ranked["overextension_penalty"] = np.clip((atr_rank - 0.85) / 0.15, 0.0, 1.0)
        ranked["bullish_score"] = 100.0 * (
            0.35 * ranked["momentum_component"]
            + 0.20 * ranked["participation_component"]
            + 0.25 * ranked["institutional_component"]
            + 0.20 * ranked["structure_component"]
            - 0.08 * ranked["overextension_penalty"]
        )
        ranked["bullish_rank"] = ranked["bullish_score"].rank(
            ascending=False, method="first"
        ).astype(int)
        rows.append(ranked)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_daily_cohorts(
    scored: pd.DataFrame,
    cohort_sizes: Iterable[int] = (5, 10, 20),
) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    sizes = sorted({int(size) for size in cohort_sizes if int(size) > 0})
    if not sizes:
        raise ValueError("Debe existir al menos un tamaño de cohorte positivo")

    rows: list[pd.DataFrame] = []
    for size in sizes:
        selected = scored.loc[scored["bullish_rank"] <= size].copy()
        selected["cohort"] = f"TOP_{size}"
        selected["cohort_size_requested"] = size
        rows.append(selected)

    universe = scored.copy()
    universe["cohort"] = "UNIVERSE"
    universe["cohort_size_requested"] = universe.groupby("origin_date")["ticker"].transform(
        "count"
    )
    rows.append(universe)
    return pd.concat(rows, ignore_index=True)


def evaluate_cohorts(cohort_predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for cohort, sample in cohort_predictions.groupby("cohort", sort=False):
        if sample.empty:
            continue
        result = evaluate_predictions(sample, ticker=cohort)
        summary = result.summary.copy()
        summary = summary.rename(columns={"ticker": "cohort"})
        summary["unique_tickers"] = int(sample["ticker"].nunique())
        summary["unique_origin_dates"] = int(sample["origin_date"].nunique())
        summary["bullish_score_mean"] = float(sample["bullish_score"].mean())
        summary["bullish_score_std"] = float(sample["bullish_score"].std(ddof=0))
        summary["actual_gap_pct_mean"] = float(sample["actual_gap_pct"].mean())
        summary["actual_gap_pct_median"] = float(sample["actual_gap_pct"].median())
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def daily_diagnostics(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    return (
        scored.groupby("origin_date")
        .agg(
            universe_size=("ticker", "nunique"),
            bullish_score_mean=("bullish_score", "mean"),
            bullish_score_std=("bullish_score", "std"),
            bullish_score_min=("bullish_score", "min"),
            bullish_score_max=("bullish_score", "max"),
            relative_volume_mean=("relative_volume", "mean"),
            smart_money_pct_mean=("smart_money_pct", "mean"),
        )
        .reset_index()
    )


def run_adaptive_universe(
    prediction_features: pd.DataFrame,
    cohort_sizes: Iterable[int] = (5, 10, 20),
) -> AdaptiveUniverseResult:
    scored = add_bullish_score(prediction_features)
    cohorts = build_daily_cohorts(scored, cohort_sizes)
    metrics = evaluate_cohorts(cohorts)
    diagnostics = daily_diagnostics(scored)
    selection_columns = [
        "origin_date",
        "ticker",
        "bullish_rank",
        "bullish_score",
        "momentum_component",
        "participation_component",
        "institutional_component",
        "structure_component",
        "overextension_penalty",
    ]
    selections = scored[[column for column in selection_columns if column in scored.columns]].copy()
    return AdaptiveUniverseResult(
        selections=selections,
        cohort_predictions=cohorts,
        cohort_metrics=metrics,
        daily_diagnostics=diagnostics,
    )

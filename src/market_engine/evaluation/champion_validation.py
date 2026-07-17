from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from market_engine.evaluation.model_evaluation import evaluate_predictions
from market_engine.evaluation.universe_tournament import add_strategy_scores


CHAMPIONS = {
    "CHAMPION_VOLUME_CONFIDENCE_STRUCTURE": (("VOLUME", "MODEL_CONFIDENCE", "STRUCTURE"), 20),
    "PARSIMONIOUS_MOMENTUM_VOLUME": (("MOMENTUM", "VOLUME"), 20),
    "INSTITUTIONAL_STABLE": (("SMART_MONEY", "LOW_VOLATILITY"), 10),
    "INSTITUTIONAL_CONFIRMED": (("VOLUME", "SMART_MONEY", "MODEL_CONFIDENCE"), 10),
    "CONTROL_VOLUME": (("VOLUME",), 20),
    "CONTROL_MOMENTUM": (("MOMENTUM",), 20),
    "CONTROL_SMART_MONEY": (("SMART_MONEY",), 20),
}


@dataclass(frozen=True)
class ChampionValidationResult:
    window_metrics: pd.DataFrame
    stability_ranking: pd.DataFrame
    degradation: pd.DataFrame
    selections: pd.DataFrame
    window_definitions: pd.DataFrame


def _window_label(timestamp: pd.Timestamp, frequency: str) -> str:
    frequency = frequency.upper()
    if frequency == "M":
        return timestamp.strftime("%Y-%m")
    if frequency == "Q":
        return f"{timestamp.year}-Q{timestamp.quarter}"
    raise ValueError("frequency debe ser M o Q")


def build_window_definitions(frame: pd.DataFrame, frequency: str = "M") -> pd.DataFrame:
    dates = pd.to_datetime(frame["origin_date"], errors="coerce").dropna()
    if dates.empty:
        raise ValueError("No existen fechas válidas para construir ventanas")
    work = pd.DataFrame({"origin_date": dates})
    work["window"] = work["origin_date"].map(lambda value: _window_label(value, frequency))
    return (
        work.groupby("window")
        .agg(window_start=("origin_date", "min"), window_end=("origin_date", "max"), origin_dates=("origin_date", "nunique"))
        .reset_index()
        .sort_values("window_start")
        .reset_index(drop=True)
    )


def _add_structure_score(scored: pd.DataFrame) -> pd.DataFrame:
    data = scored.copy()
    rows: list[pd.DataFrame] = []
    for _, daily in data.groupby("origin_date", sort=True):
        ranked = daily.copy()
        close_vwap = pd.to_numeric(ranked["close_vs_vwap_pct"], errors="coerce")
        close_position = pd.to_numeric(ranked["close_position_pct"], errors="coerce")
        vwap_rank = close_vwap.rank(pct=True, method="average").fillna(0.5)
        position_rank = close_position.rank(pct=True, method="average").fillna(0.5)
        ranked["score_STRUCTURE"] = 100.0 * (0.55 * vwap_rank + 0.45 * position_rank)
        rows.append(ranked)
    return pd.concat(rows, ignore_index=True) if rows else data


def add_champion_scores(frame: pd.DataFrame) -> pd.DataFrame:
    scored = _add_structure_score(add_strategy_scores(frame))
    for name, (factors, cohort_size) in CHAMPIONS.items():
        columns = [f"score_{factor}" for factor in factors]
        scored[f"score_{name}"] = scored[columns].mean(axis=1)
        scored[f"rank_{name}"] = scored.groupby("origin_date")[f"score_{name}"].rank(
            ascending=False, method="first"
        )
        scored[f"cohort_size_{name}"] = cohort_size
    return scored


def build_champion_selections(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for name, (_, cohort_size) in CHAMPIONS.items():
        selected = scored.loc[scored[f"rank_{name}"] <= cohort_size].copy()
        selected["champion"] = name
        selected["cohort_size"] = cohort_size
        selected["champion_score"] = selected[f"score_{name}"]
        selected["champion_rank"] = selected[f"rank_{name}"]
        rows.append(selected)

    universe = scored.copy()
    universe["champion"] = "UNIVERSE"
    universe["cohort_size"] = universe.groupby("origin_date")["ticker"].transform("nunique")
    universe["champion_score"] = np.nan
    universe["champion_rank"] = np.nan
    rows.append(universe)
    return pd.concat(rows, ignore_index=True)


def _evaluate_sample(sample: pd.DataFrame, label: str) -> pd.DataFrame:
    summary = evaluate_predictions(sample, ticker=label).summary.copy()
    summary = summary.rename(columns={"ticker": "champion"})
    summary["unique_tickers"] = int(sample["ticker"].nunique())
    summary["unique_origin_dates"] = int(sample["origin_date"].nunique())
    summary["actual_gap_pct_mean"] = float(sample["actual_gap_pct"].mean())
    summary["actual_gap_pct_median"] = float(sample["actual_gap_pct"].median())
    return summary


def evaluate_windows(selections: pd.DataFrame, frequency: str = "M") -> pd.DataFrame:
    data = selections.copy()
    data["origin_date"] = pd.to_datetime(data["origin_date"])
    data["window"] = data["origin_date"].map(lambda value: _window_label(value, frequency))
    rows: list[pd.DataFrame] = []
    for (champion, window), sample in data.groupby(["champion", "window"], sort=True):
        if sample.empty:
            continue
        summary = _evaluate_sample(sample, str(champion))
        summary["window"] = window
        summary["window_start"] = sample["origin_date"].min()
        summary["window_end"] = sample["origin_date"].max()
        summary["cohort_size"] = int(sample["cohort_size"].median())
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_degradation(window_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ordered = window_metrics.sort_values(["champion", "window_start"])
    metrics = [
        "predictability_score",
        "rare_event_f1",
        "balanced_accuracy",
        "macro_f1",
        "mean_brier_skill",
        "mean_calibration_error",
    ]
    for champion, sample in ordered.groupby("champion", sort=False):
        sample = sample.reset_index(drop=True)
        for index in range(1, len(sample)):
            row: dict[str, object] = {
                "champion": champion,
                "from_window": sample.loc[index - 1, "window"],
                "to_window": sample.loc[index, "window"],
            }
            for metric in metrics:
                previous = pd.to_numeric(pd.Series([sample.loc[index - 1, metric]]), errors="coerce").iloc[0]
                current = pd.to_numeric(pd.Series([sample.loc[index, metric]]), errors="coerce").iloc[0]
                row[f"delta_{metric}"] = current - previous if pd.notna(previous) and pd.notna(current) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def build_stability_ranking(window_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for champion, sample in window_metrics.groupby("champion", sort=False):
        rare = pd.to_numeric(sample["rare_event_f1"], errors="coerce")
        balanced = pd.to_numeric(sample["balanced_accuracy"], errors="coerce")
        macro = pd.to_numeric(sample["macro_f1"], errors="coerce")
        brier = pd.to_numeric(sample["mean_brier_skill"], errors="coerce")
        calibration = pd.to_numeric(sample["mean_calibration_error"], errors="coerce")
        quality = 0.35 * rare + 0.25 * balanced + 0.20 * macro + 0.10 * brier.fillna(0.0) + 0.10 * (1.0 - calibration)
        stability_penalty = float(quality.std(ddof=0)) if len(quality) > 1 else 0.0
        rows.append(
            {
                "champion": champion,
                "windows": int(sample["window"].nunique()),
                "mean_quality": float(quality.mean()),
                "minimum_quality": float(quality.min()),
                "quality_std": stability_penalty,
                "mean_rare_event_f1": float(rare.mean()),
                "minimum_rare_event_f1": float(rare.min()),
                "positive_brier_windows": int((brier > 0).sum()),
                "stability_score": float(100.0 * (quality.mean() - 0.75 * stability_penalty)),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        ["stability_score", "minimum_quality", "mean_rare_event_f1"], ascending=False
    ).reset_index(drop=True)
    ranking.insert(0, "stability_rank", np.arange(1, len(ranking) + 1))
    return ranking


def run_champion_validation(
    prediction_features: pd.DataFrame,
    frequency: str = "M",
) -> ChampionValidationResult:
    scored = add_champion_scores(prediction_features)
    selections = build_champion_selections(scored)
    windows = build_window_definitions(scored, frequency)
    metrics = evaluate_windows(selections, frequency)
    degradation = build_degradation(metrics)
    stability = build_stability_ranking(metrics)
    return ChampionValidationResult(
        window_metrics=metrics,
        stability_ranking=stability,
        degradation=degradation,
        selections=selections,
        window_definitions=windows,
    )

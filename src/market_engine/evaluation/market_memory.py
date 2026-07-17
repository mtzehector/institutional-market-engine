from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from market_engine.evaluation.model_evaluation import evaluate_predictions


STATE_FEATURES = (
    "qqq_return_1d_pct",
    "breadth_positive_1d",
    "breadth_positive_5d",
    "median_relative_volume",
    "median_atr_pct",
    "institutional_breadth",
)


@dataclass(frozen=True)
class MarketMemoryResult:
    recommendations: pd.DataFrame
    neighbor_states: pd.DataFrame
    champion_scores: pd.DataFrame
    date_champion_quality: pd.DataFrame
    summary: pd.DataFrame


def _safe_float(value: object, default: float = 0.0) -> float:
    return float(value) if pd.notna(value) else default


def _quality(summary: pd.Series) -> float:
    """Comparable 0-100 quality score used by the regime laboratories."""
    return 100.0 * (
        0.35 * _safe_float(summary.get("rare_event_f1"), 0.0)
        + 0.25 * _safe_float(summary.get("balanced_accuracy"), 0.0)
        + 0.20 * _safe_float(summary.get("macro_f1"), 0.0)
        + 0.10 * _safe_float(summary.get("mean_brier_skill"), 0.0)
        + 0.10 * (1.0 - _safe_float(summary.get("mean_calibration_error"), 1.0))
    )


def validate_market_memory_inputs(
    daily_states: pd.DataFrame,
    selections: pd.DataFrame,
) -> None:
    state_required = {"origin_date", *STATE_FEATURES}
    missing_states = state_required - set(daily_states.columns)
    if missing_states:
        raise ValueError(f"Faltan columnas en Estados_Diarios: {sorted(missing_states)}")

    selection_required = {
        "origin_date",
        "champion",
        "ticker",
        "actual_direction",
        "predicted_direction",
        "probability_up",
        "probability_down",
        "probability_no_gap",
    }
    missing_selections = selection_required - set(selections.columns)
    if missing_selections:
        raise ValueError(f"Faltan columnas en Selecciones: {sorted(missing_selections)}")


def build_date_champion_quality(selections: pd.DataFrame) -> pd.DataFrame:
    """Evaluate every champion independently on every synchronized date."""
    data = selections.copy()
    data["origin_date"] = pd.to_datetime(data["origin_date"])
    rows: list[dict[str, object]] = []

    for (origin_date, champion), sample in data.groupby(
        ["origin_date", "champion"], sort=True
    ):
        if sample.empty:
            continue
        summary = evaluate_predictions(sample, ticker=str(champion)).summary.iloc[0]
        rows.append(
            {
                "origin_date": origin_date,
                "champion": str(champion),
                "observations": int(len(sample)),
                "unique_tickers": int(sample["ticker"].nunique()),
                "quality_score": _quality(summary),
                "predictability_score": _safe_float(
                    summary.get("predictability_score"), np.nan
                ),
                "rare_event_f1": _safe_float(summary.get("rare_event_f1"), 0.0),
                "balanced_accuracy": _safe_float(
                    summary.get("balanced_accuracy"), 0.0
                ),
                "macro_f1": _safe_float(summary.get("macro_f1"), 0.0),
                "mean_brier_skill": _safe_float(
                    summary.get("mean_brier_skill"), 0.0
                ),
                "mean_calibration_error": _safe_float(
                    summary.get("mean_calibration_error"), 1.0
                ),
            }
        )
    return pd.DataFrame(rows)


def _robust_scale(history: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    numeric = history.loc[:, STATE_FEATURES].apply(pd.to_numeric, errors="coerce")
    target_numeric = pd.to_numeric(target.loc[list(STATE_FEATURES)], errors="coerce")

    median = numeric.median()
    mad = (numeric - median).abs().median()
    fallback = numeric.std(ddof=0)
    scale = (1.4826 * mad).where(mad > 1e-12, fallback)
    scale = scale.where(scale > 1e-12, 1.0).fillna(1.0)

    scaled_history = ((numeric - median) / scale).replace([np.inf, -np.inf], np.nan)
    scaled_target = ((target_numeric - median) / scale).replace(
        [np.inf, -np.inf], np.nan
    )
    return scaled_history.fillna(0.0), scaled_target.fillna(0.0)


def find_causal_neighbors(
    daily_states: pd.DataFrame,
    *,
    neighbors: int = 5,
    minimum_history: int = 8,
) -> pd.DataFrame:
    if neighbors < 1:
        raise ValueError("neighbors debe ser positivo")
    if minimum_history < neighbors:
        raise ValueError("minimum_history debe ser mayor o igual que neighbors")

    states = daily_states.copy()
    states["origin_date"] = pd.to_datetime(states["origin_date"])
    states = states.sort_values("origin_date").drop_duplicates("origin_date", keep="last")

    rows: list[dict[str, object]] = []
    for position in range(minimum_history, len(states)):
        target = states.iloc[position]
        history = states.iloc[:position].copy()
        scaled_history, scaled_target = _robust_scale(history, target)
        differences = scaled_history.subtract(scaled_target, axis=1)
        distance = np.sqrt(np.square(differences).sum(axis=1))
        nearest_positions = distance.nsmallest(min(neighbors, len(distance))).index

        for rank, history_index in enumerate(nearest_positions, start=1):
            neighbor = history.loc[history_index]
            value = float(distance.loc[history_index])
            rows.append(
                {
                    "target_date": target["origin_date"],
                    "neighbor_rank": rank,
                    "neighbor_date": neighbor["origin_date"],
                    "distance": value,
                    "similarity_weight": 1.0 / (value + 1e-6),
                    "target_regime": target.get("regime"),
                    "neighbor_regime": neighbor.get("regime"),
                    **{
                        f"target_{feature}": target.get(feature)
                        for feature in STATE_FEATURES
                    },
                    **{
                        f"neighbor_{feature}": neighbor.get(feature)
                        for feature in STATE_FEATURES
                    },
                }
            )
    return pd.DataFrame(rows)


def score_champions_from_neighbors(
    neighbors: pd.DataFrame,
    date_quality: pd.DataFrame,
) -> pd.DataFrame:
    if neighbors.empty or date_quality.empty:
        return pd.DataFrame()

    qualities = date_quality.rename(columns={"origin_date": "neighbor_date"})
    merged = neighbors.merge(qualities, on="neighbor_date", how="inner")
    if merged.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for (target_date, champion), sample in merged.groupby(
        ["target_date", "champion"], sort=True
    ):
        weights = pd.to_numeric(sample["similarity_weight"], errors="coerce").fillna(0.0)
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            continue

        def weighted(column: str) -> float:
            values = pd.to_numeric(sample[column], errors="coerce").fillna(0.0)
            return float(np.average(values, weights=weights))

        rows.append(
            {
                "target_date": target_date,
                "champion": champion,
                "memory_score": weighted("quality_score"),
                "memory_rare_event_f1": weighted("rare_event_f1"),
                "memory_balanced_accuracy": weighted("balanced_accuracy"),
                "memory_macro_f1": weighted("macro_f1"),
                "memory_brier_skill": weighted("mean_brier_skill"),
                "memory_calibration_error": weighted("mean_calibration_error"),
                "neighbor_dates_used": int(sample["neighbor_date"].nunique()),
                "mean_neighbor_distance": float(sample["distance"].mean()),
                "closest_neighbor_distance": float(sample["distance"].min()),
            }
        )

    scores = pd.DataFrame(rows)
    if scores.empty:
        return scores
    scores["memory_rank"] = scores.groupby("target_date")["memory_score"].rank(
        ascending=False, method="first"
    )
    return scores.sort_values(["target_date", "memory_rank"]).reset_index(drop=True)


def build_memory_recommendations(
    champion_scores: pd.DataFrame,
    date_quality: pd.DataFrame,
) -> pd.DataFrame:
    if champion_scores.empty:
        return pd.DataFrame()

    actual = date_quality.rename(
        columns={
            "origin_date": "target_date",
            "quality_score": "actual_quality_score",
            "rare_event_f1": "actual_rare_event_f1",
            "balanced_accuracy": "actual_balanced_accuracy",
            "macro_f1": "actual_macro_f1",
        }
    )
    recommendations: list[dict[str, object]] = []

    for target_date, scores in champion_scores.groupby("target_date", sort=True):
        ordered = scores.sort_values("memory_rank").reset_index(drop=True)
        winner = ordered.iloc[0]
        runner_up_score = (
            float(ordered.iloc[1]["memory_score"]) if len(ordered) > 1 else np.nan
        )
        actual_date = actual.loc[actual["target_date"] == target_date].copy()
        selected_actual = actual_date.loc[
            actual_date["champion"] == winner["champion"]
        ]
        oracle = actual_date.sort_values("actual_quality_score", ascending=False).head(1)
        universe = actual_date.loc[actual_date["champion"] == "UNIVERSE"]

        selected_quality = (
            float(selected_actual.iloc[0]["actual_quality_score"])
            if not selected_actual.empty
            else np.nan
        )
        oracle_quality = (
            float(oracle.iloc[0]["actual_quality_score"]) if not oracle.empty else np.nan
        )
        universe_quality = (
            float(universe.iloc[0]["actual_quality_score"])
            if not universe.empty
            else np.nan
        )

        recommendations.append(
            {
                "target_date": target_date,
                "recommended_champion": winner["champion"],
                "memory_score": float(winner["memory_score"]),
                "runner_up_score": runner_up_score,
                "recommendation_margin": (
                    float(winner["memory_score"]) - runner_up_score
                    if pd.notna(runner_up_score)
                    else np.nan
                ),
                "neighbor_dates_used": int(winner["neighbor_dates_used"]),
                "mean_neighbor_distance": float(winner["mean_neighbor_distance"]),
                "closest_neighbor_distance": float(
                    winner["closest_neighbor_distance"]
                ),
                "actual_selected_quality": selected_quality,
                "actual_oracle_champion": (
                    oracle.iloc[0]["champion"] if not oracle.empty else None
                ),
                "actual_oracle_quality": oracle_quality,
                "actual_universe_quality": universe_quality,
                "oracle_regret": (
                    oracle_quality - selected_quality
                    if pd.notna(oracle_quality) and pd.notna(selected_quality)
                    else np.nan
                ),
                "advantage_vs_universe": (
                    selected_quality - universe_quality
                    if pd.notna(selected_quality) and pd.notna(universe_quality)
                    else np.nan
                ),
                "selected_was_oracle": (
                    bool(winner["champion"] == oracle.iloc[0]["champion"])
                    if not oracle.empty
                    else False
                ),
            }
        )
    return pd.DataFrame(recommendations)


def build_memory_summary(recommendations: pd.DataFrame) -> pd.DataFrame:
    if recommendations.empty:
        return pd.DataFrame()
    advantage = pd.to_numeric(
        recommendations["advantage_vs_universe"], errors="coerce"
    )
    regret = pd.to_numeric(recommendations["oracle_regret"], errors="coerce")
    return pd.DataFrame(
        [
            {
                "recommendation_dates": int(len(recommendations)),
                "oracle_hit_rate": float(recommendations["selected_was_oracle"].mean()),
                "mean_actual_selected_quality": float(
                    pd.to_numeric(
                        recommendations["actual_selected_quality"], errors="coerce"
                    ).mean()
                ),
                "mean_actual_universe_quality": float(
                    pd.to_numeric(
                        recommendations["actual_universe_quality"], errors="coerce"
                    ).mean()
                ),
                "mean_advantage_vs_universe": float(advantage.mean()),
                "positive_advantage_rate": float((advantage > 0).mean()),
                "mean_oracle_regret": float(regret.mean()),
                "median_oracle_regret": float(regret.median()),
                "mean_recommendation_margin": float(
                    pd.to_numeric(
                        recommendations["recommendation_margin"], errors="coerce"
                    ).mean()
                ),
                "mean_neighbor_distance": float(
                    pd.to_numeric(
                        recommendations["mean_neighbor_distance"], errors="coerce"
                    ).mean()
                ),
            }
        ]
    )


def run_market_memory_laboratory(
    daily_states: pd.DataFrame,
    selections: pd.DataFrame,
    *,
    neighbors: int = 5,
    minimum_history: int = 8,
) -> MarketMemoryResult:
    validate_market_memory_inputs(daily_states, selections)
    date_quality = build_date_champion_quality(selections)
    neighbor_states = find_causal_neighbors(
        daily_states,
        neighbors=neighbors,
        minimum_history=minimum_history,
    )
    champion_scores = score_champions_from_neighbors(neighbor_states, date_quality)
    recommendations = build_memory_recommendations(champion_scores, date_quality)
    summary = build_memory_summary(recommendations)
    return MarketMemoryResult(
        recommendations=recommendations,
        neighbor_states=neighbor_states,
        champion_scores=champion_scores,
        date_champion_quality=date_quality,
        summary=summary,
    )

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from market_engine.evaluation.market_memory import run_market_memory_laboratory


@dataclass(frozen=True)
class MarketMemoryConfidenceResult:
    recommendations: pd.DataFrame
    confidence_validation: pd.DataFrame
    sensitivity_k: pd.DataFrame
    novelty_dimensions: pd.DataFrame
    abstention_counterfactual: pd.DataFrame
    summary: pd.DataFrame


def _safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _decision_level(score: float) -> str:
    if score >= 80:
        return "RECOMMEND"
    if score >= 60:
        return "CAUTIOUS_RECOMMEND"
    if score >= 40:
        return "WATCH"
    return "ABSTAIN"


def _confidence_from_run(result, ks: tuple[int, ...]) -> pd.DataFrame:
    rec = result.recommendations.copy()
    if rec.empty:
        return rec

    neighbors = result.neighbor_states.copy()
    scores = result.champion_scores.copy()
    quality = result.date_champion_quality.copy()

    rows: list[dict[str, object]] = []
    for _, row in rec.iterrows():
        date = pd.Timestamp(row["target_date"])
        champion = row["recommended_champion"]
        neigh = neighbors.loc[pd.to_datetime(neighbors["target_date"]) == date].copy()
        champ_quality = quality.loc[quality["champion"] == champion].rename(
            columns={"origin_date": "neighbor_date"}
        )
        merged = neigh.merge(champ_quality[["neighbor_date", "quality_score"]], on="neighbor_date", how="left")

        distances = _safe_numeric(neigh.get("distance", pd.Series(dtype=float)), 0.0)
        mean_distance = float(distances.mean()) if not distances.empty else np.nan
        closest_distance = float(distances.min()) if not distances.empty else np.nan
        similarity_confidence = float(100.0 / (1.0 + max(mean_distance, 0.0))) if pd.notna(mean_distance) else 0.0

        qualities = _safe_numeric(merged.get("quality_score", pd.Series(dtype=float)), 0.0)
        outcome_std = float(qualities.std(ddof=0)) if len(qualities) else np.nan
        outcome_iqr = float(qualities.quantile(0.75) - qualities.quantile(0.25)) if len(qualities) else np.nan
        outcome_stability = float(100.0 / (1.0 + max(outcome_std, 0.0))) if pd.notna(outcome_std) else 0.0

        neighbor_dates = pd.to_datetime(neigh.get("neighbor_date", pd.Series(dtype="datetime64[ns]")), errors="coerce")
        unique_months = int(neighbor_dates.dt.to_period("M").nunique()) if not neighbor_dates.empty else 0
        temporal_diversity = min(100.0, 25.0 * unique_months)
        date_span = int((neighbor_dates.max() - neighbor_dates.min()).days) if len(neighbor_dates.dropna()) > 1 else 0

        score_rows = scores.loc[pd.to_datetime(scores["target_date"]) == date].sort_values("memory_rank")
        margin = float(row.get("recommendation_margin", 0.0) or 0.0)
        margin_confidence = float(100.0 * (1.0 - np.exp(-max(margin, 0.0) / 10.0)))

        rows.append({
            **row.to_dict(),
            "similarity_confidence": similarity_confidence,
            "outcome_std": outcome_std,
            "outcome_iqr": outcome_iqr,
            "outcome_stability": outcome_stability,
            "unique_neighbor_months": unique_months,
            "neighbor_date_span_days": date_span,
            "temporal_diversity": temporal_diversity,
            "margin_confidence": margin_confidence,
            "candidate_count": int(len(score_rows)),
        })
    return pd.DataFrame(rows)


def _build_multi_k_stability(
    daily_states: pd.DataFrame,
    selections: pd.DataFrame,
    ks: tuple[int, ...],
    minimum_history: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs: list[pd.DataFrame] = []
    for k in ks:
        result = run_market_memory_laboratory(
            daily_states,
            selections,
            neighbors=k,
            minimum_history=max(minimum_history, k),
        )
        frame = result.recommendations[["target_date", "recommended_champion", "advantage_vs_universe", "oracle_regret"]].copy()
        frame["k"] = k
        runs.append(frame)
    sensitivity = pd.concat(runs, ignore_index=True) if runs else pd.DataFrame()

    stability_rows: list[dict[str, object]] = []
    for target_date, sample in sensitivity.groupby("target_date", sort=True):
        counts = sample["recommended_champion"].value_counts(dropna=False)
        winner = counts.index[0]
        share = float(counts.iloc[0] / len(sample))
        stability_rows.append({
            "target_date": target_date,
            "multi_k_consensus_champion": winner,
            "multi_k_stability": 100.0 * share,
            "unique_recommendations_across_k": int(counts.size),
        })
    return sensitivity, pd.DataFrame(stability_rows)


def _novelty_by_dimension(neighbors: pd.DataFrame) -> pd.DataFrame:
    if neighbors.empty:
        return pd.DataFrame()
    features = [column.removeprefix("target_") for column in neighbors.columns if column.startswith("target_")]
    rows: list[dict[str, object]] = []
    for target_date, sample in neighbors.groupby("target_date", sort=True):
        for feature in features:
            target_col = f"target_{feature}"
            neighbor_col = f"neighbor_{feature}"
            if neighbor_col not in sample.columns:
                continue
            target_value = pd.to_numeric(sample[target_col], errors="coerce").iloc[0]
            neighbor_values = pd.to_numeric(sample[neighbor_col], errors="coerce")
            spread = float(neighbor_values.std(ddof=0))
            spread = spread if np.isfinite(spread) and spread > 1e-12 else 1.0
            novelty = float(abs(target_value - neighbor_values.median()) / spread) if pd.notna(target_value) else np.nan
            rows.append({"target_date": target_date, "feature": feature, "dimension_novelty_z": novelty})
    return pd.DataFrame(rows)


def run_market_memory_confidence_laboratory(
    daily_states: pd.DataFrame,
    selections: pd.DataFrame,
    *,
    ks: tuple[int, ...] = (3, 5, 7, 10),
    baseline_k: int = 5,
    minimum_history: int = 10,
) -> MarketMemoryConfidenceResult:
    if baseline_k not in ks:
        ks = tuple(sorted(set((*ks, baseline_k))))
    baseline = run_market_memory_laboratory(
        daily_states,
        selections,
        neighbors=baseline_k,
        minimum_history=max(minimum_history, baseline_k),
    )
    recommendations = _confidence_from_run(baseline, ks)
    sensitivity, stability = _build_multi_k_stability(daily_states, selections, ks, minimum_history)
    recommendations = recommendations.merge(stability, on="target_date", how="left")

    novelty = _novelty_by_dimension(baseline.neighbor_states)
    novelty_summary = novelty.groupby("target_date")["dimension_novelty_z"].max().rename("global_novelty_score").reset_index() if not novelty.empty else pd.DataFrame(columns=["target_date", "global_novelty_score"])
    recommendations = recommendations.merge(novelty_summary, on="target_date", how="left")
    recommendations["novelty_confidence"] = 100.0 / (1.0 + _safe_numeric(recommendations["global_novelty_score"], 0.0))

    recommendations["memory_confidence_score"] = (
        0.25 * _safe_numeric(recommendations["similarity_confidence"]) +
        0.20 * _safe_numeric(recommendations["margin_confidence"]) +
        0.20 * _safe_numeric(recommendations["outcome_stability"]) +
        0.20 * _safe_numeric(recommendations["multi_k_stability"]) +
        0.10 * _safe_numeric(recommendations["temporal_diversity"]) +
        0.05 * _safe_numeric(recommendations["novelty_confidence"])
    )
    recommendations["decision_level"] = recommendations["memory_confidence_score"].map(_decision_level)
    recommendations["accepted_decision"] = recommendations["decision_level"].isin(["RECOMMEND", "CAUTIOUS_RECOMMEND"])

    validation = recommendations.groupby("decision_level", dropna=False).agg(
        dates=("target_date", "count"),
        mean_confidence=("memory_confidence_score", "mean"),
        mean_advantage_vs_universe=("advantage_vs_universe", "mean"),
        positive_advantage_rate=("advantage_vs_universe", lambda s: float((_safe_numeric(s) > 0).mean())),
        mean_oracle_regret=("oracle_regret", "mean"),
        oracle_hit_rate=("selected_was_oracle", "mean"),
    ).reset_index()

    accepted = recommendations.loc[recommendations["accepted_decision"]]
    rejected = recommendations.loc[~recommendations["accepted_decision"]]
    abstention = pd.DataFrame([{
        "total_dates": int(len(recommendations)),
        "accepted_dates": int(len(accepted)),
        "rejected_dates": int(len(rejected)),
        "coverage_rate": float(len(accepted) / len(recommendations)) if len(recommendations) else np.nan,
        "accepted_mean_advantage": float(pd.to_numeric(accepted["advantage_vs_universe"], errors="coerce").mean()),
        "rejected_mean_advantage": float(pd.to_numeric(rejected["advantage_vs_universe"], errors="coerce").mean()),
        "accepted_mean_oracle_regret": float(pd.to_numeric(accepted["oracle_regret"], errors="coerce").mean()),
        "rejected_mean_oracle_regret": float(pd.to_numeric(rejected["oracle_regret"], errors="coerce").mean()),
    }])

    summary = pd.DataFrame([{
        "recommendation_dates": int(len(recommendations)),
        "mean_confidence": float(recommendations["memory_confidence_score"].mean()),
        "accepted_dates": int(recommendations["accepted_decision"].sum()),
        "coverage_rate": float(recommendations["accepted_decision"].mean()),
        "accepted_positive_advantage_rate": float((_safe_numeric(accepted["advantage_vs_universe"]) > 0).mean()) if len(accepted) else np.nan,
        "accepted_mean_advantage": float(pd.to_numeric(accepted["advantage_vs_universe"], errors="coerce").mean()),
        "accepted_mean_oracle_regret": float(pd.to_numeric(accepted["oracle_regret"], errors="coerce").mean()),
    }])

    return MarketMemoryConfidenceResult(
        recommendations=recommendations,
        confidence_validation=validation,
        sensitivity_k=sensitivity,
        novelty_dimensions=novelty,
        abstention_counterfactual=abstention,
        summary=summary,
    )

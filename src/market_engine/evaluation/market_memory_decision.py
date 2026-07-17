from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from market_engine.evaluation.market_memory_confidence import (
    MarketMemoryConfidenceResult,
    run_market_memory_confidence_laboratory,
)


POLICIES = (
    "BASELINE_ACCEPT_ALL",
    "COMPOSITE_SCORE",
    "NOVELTY_ONLY",
    "NOVELTY_STABILITY",
    "HIERARCHICAL_VETO",
)


@dataclass(frozen=True)
class MarketMemoryDecisionResult:
    recommendations: pd.DataFrame
    policy_comparison: pd.DataFrame
    policy_by_date: pd.DataFrame
    threshold_audit: pd.DataFrame
    summary: pd.DataFrame
    confidence_result: MarketMemoryConfidenceResult


def _numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _causal_percentile(values: pd.Series, minimum_history: int, percentile: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=numeric.index, dtype=float)
    for index in range(len(numeric)):
        history = numeric.iloc[:index].dropna()
        if len(history) >= minimum_history:
            result.iloc[index] = float(history.quantile(percentile))
    return result


def _causal_lower_percentile(values: pd.Series, minimum_history: int, percentile: float) -> pd.Series:
    return _causal_percentile(values, minimum_history, percentile)


def add_causal_thresholds(
    recommendations: pd.DataFrame,
    *,
    calibration_history: int = 6,
    novelty_percentile: float = 0.80,
    similarity_percentile: float = 0.25,
    stability_percentile: float = 0.25,
) -> pd.DataFrame:
    data = recommendations.sort_values("target_date").reset_index(drop=True).copy()
    data["novelty_veto_threshold"] = _causal_percentile(
        data["global_novelty_score"], calibration_history, novelty_percentile
    )
    data["similarity_floor"] = _causal_lower_percentile(
        data["similarity_confidence"], calibration_history, similarity_percentile
    )
    data["stability_floor"] = _causal_lower_percentile(
        data["outcome_stability"], calibration_history, stability_percentile
    )
    return data


def _policy_decisions(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in data.iterrows():
        novelty_threshold = row.get("novelty_veto_threshold")
        similarity_floor = row.get("similarity_floor")
        stability_floor = row.get("stability_floor")
        calibrated = pd.notna(novelty_threshold) and pd.notna(similarity_floor) and pd.notna(stability_floor)

        novelty_extreme = bool(calibrated and row["global_novelty_score"] > novelty_threshold)
        similarity_weak = bool(calibrated and row["similarity_confidence"] < similarity_floor)
        stability_weak = bool(calibrated and row["outcome_stability"] < stability_floor)

        decisions = {
            "BASELINE_ACCEPT_ALL": True,
            "COMPOSITE_SCORE": bool(row["accepted_decision"]),
            "NOVELTY_ONLY": bool(calibrated and not novelty_extreme),
            "NOVELTY_STABILITY": bool(calibrated and not novelty_extreme and not stability_weak),
            "HIERARCHICAL_VETO": bool(
                calibrated
                and not novelty_extreme
                and not similarity_weak
                and not stability_weak
            ),
        }
        for policy, accepted in decisions.items():
            reason = "ACCEPT"
            if not calibrated and policy not in {"BASELINE_ACCEPT_ALL", "COMPOSITE_SCORE"}:
                reason = "INSUFFICIENT_CALIBRATION_HISTORY"
            elif novelty_extreme and policy in {"NOVELTY_ONLY", "NOVELTY_STABILITY", "HIERARCHICAL_VETO"}:
                reason = "NOVELTY_VETO"
            elif stability_weak and policy in {"NOVELTY_STABILITY", "HIERARCHICAL_VETO"}:
                reason = "OUTCOME_STABILITY_VETO"
            elif similarity_weak and policy == "HIERARCHICAL_VETO":
                reason = "SIMILARITY_VETO"
            elif policy == "COMPOSITE_SCORE" and not accepted:
                reason = "COMPOSITE_REJECT"
            rows.append({
                "target_date": row["target_date"],
                "policy": policy,
                "accepted": accepted,
                "decision_reason": reason,
                "recommended_champion": row["recommended_champion"],
                "advantage_vs_universe": row["advantage_vs_universe"],
                "oracle_regret": row["oracle_regret"],
                "selected_was_oracle": row["selected_was_oracle"],
                "global_novelty_score": row["global_novelty_score"],
                "similarity_confidence": row["similarity_confidence"],
                "outcome_stability": row["outcome_stability"],
                "memory_confidence_score": row["memory_confidence_score"],
                "novelty_veto_threshold": novelty_threshold,
                "similarity_floor": similarity_floor,
                "stability_floor": stability_floor,
            })
    return pd.DataFrame(rows)


def compare_policies(policy_by_date: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for policy, sample in policy_by_date.groupby("policy", sort=False):
        accepted = sample.loc[sample["accepted"]]
        rejected = sample.loc[~sample["accepted"]]
        advantage = pd.to_numeric(accepted["advantage_vs_universe"], errors="coerce")
        regret = pd.to_numeric(accepted["oracle_regret"], errors="coerce")
        rows.append({
            "policy": policy,
            "total_dates": int(len(sample)),
            "accepted_dates": int(len(accepted)),
            "coverage_rate": float(len(accepted) / len(sample)) if len(sample) else np.nan,
            "accepted_mean_advantage": float(advantage.mean()),
            "accepted_positive_advantage_rate": float((advantage > 0).mean()) if len(accepted) else np.nan,
            "accepted_mean_oracle_regret": float(regret.mean()),
            "accepted_max_oracle_regret": float(regret.max()),
            "accepted_oracle_hit_rate": float(accepted["selected_was_oracle"].mean()) if len(accepted) else np.nan,
            "rejected_mean_advantage": float(pd.to_numeric(rejected["advantage_vs_universe"], errors="coerce").mean()),
            "rejected_mean_oracle_regret": float(pd.to_numeric(rejected["oracle_regret"], errors="coerce").mean()),
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["policy_score"] = (
        0.30 * result["coverage_rate"].fillna(0.0)
        + 0.30 * result["accepted_positive_advantage_rate"].fillna(0.0)
        + 0.20 * (result["accepted_mean_advantage"].fillna(0.0).clip(-50, 50) + 50) / 100
        + 0.20 * (1.0 - result["accepted_mean_oracle_regret"].fillna(100.0).clip(0, 100) / 100)
    ) * 100.0
    return result.sort_values("policy_score", ascending=False).reset_index(drop=True)


def run_market_memory_decision_laboratory(
    daily_states: pd.DataFrame,
    selections: pd.DataFrame,
    *,
    ks: tuple[int, ...] = (3, 5, 7, 10),
    baseline_k: int = 5,
    minimum_history: int = 10,
    calibration_history: int = 6,
    novelty_percentile: float = 0.80,
    similarity_percentile: float = 0.25,
    stability_percentile: float = 0.25,
) -> MarketMemoryDecisionResult:
    confidence = run_market_memory_confidence_laboratory(
        daily_states,
        selections,
        ks=ks,
        baseline_k=baseline_k,
        minimum_history=minimum_history,
    )
    recommendations = add_causal_thresholds(
        confidence.recommendations,
        calibration_history=calibration_history,
        novelty_percentile=novelty_percentile,
        similarity_percentile=similarity_percentile,
        stability_percentile=stability_percentile,
    )
    policy_by_date = _policy_decisions(recommendations)
    comparison = compare_policies(policy_by_date)
    thresholds = recommendations[[
        "target_date", "global_novelty_score", "novelty_veto_threshold",
        "similarity_confidence", "similarity_floor",
        "outcome_stability", "stability_floor",
    ]].copy()
    summary = comparison.head(1).copy()
    if not summary.empty:
        summary = summary.rename(columns={"policy": "recommended_policy"})
    return MarketMemoryDecisionResult(
        recommendations=recommendations,
        policy_comparison=comparison,
        policy_by_date=policy_by_date,
        threshold_audit=thresholds,
        summary=summary,
        confidence_result=confidence,
    )

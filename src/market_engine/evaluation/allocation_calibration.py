from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from market_engine.evaluation.adaptive_intelligence_allocation import (
    run_adaptive_intelligence_allocation_laboratory,
)
from market_engine.evaluation.market_memory import build_date_champion_quality


CALIBRATION_POLICIES = (
    "LINEAR",
    "LOGISTIC",
    "EXPONENTIAL",
    "PIECEWISE",
    "BOUNDED_ADAPTIVE",
)


@dataclass(frozen=True)
class AllocationCalibrationResult:
    recommendations: pd.DataFrame
    allocation_by_date: pd.DataFrame
    policy_comparison: pd.DataFrame
    calibration_report: pd.DataFrame
    conviction_report: pd.DataFrame
    disagreement_report: pd.DataFrame
    summary: pd.DataFrame


def _numeric(value: object, default: float = 0.0) -> float:
    return float(value) if pd.notna(value) else default


def _novelty_ratio(row: pd.Series) -> float:
    novelty = max(_numeric(row.get("global_novelty_score"), 0.0), 0.0)
    threshold = _numeric(row.get("novelty_veto_threshold"), np.nan)
    if pd.notna(threshold) and threshold > 1e-12:
        return max(novelty / threshold, 0.0)
    similarity = np.clip(
        _numeric(row.get("similarity_confidence"), 50.0) / 100.0, 0.0, 1.0
    )
    return max(1.0 - similarity, 0.0)


def memory_weight_from_curve(row: pd.Series, policy: str) -> float:
    ratio = _novelty_ratio(row)
    similarity = np.clip(
        _numeric(row.get("similarity_confidence"), 50.0) / 100.0, 0.0, 1.0
    )
    stability = np.clip(
        _numeric(row.get("outcome_stability"), 50.0) / 100.0, 0.0, 1.0
    )

    if policy == "LINEAR":
        weight = 1.0 - 0.60 * ratio
    elif policy == "LOGISTIC":
        weight = 0.10 + 0.80 / (1.0 + np.exp(4.0 * (ratio - 1.0)))
    elif policy == "EXPONENTIAL":
        weight = 0.90 * np.exp(-1.25 * ratio)
    elif policy == "PIECEWISE":
        if ratio <= 0.50:
            weight = 0.90
        elif ratio <= 0.90:
            weight = 0.75
        elif ratio <= 1.25:
            weight = 0.50
        elif ratio <= 1.75:
            weight = 0.25
        else:
            weight = 0.10
    elif policy == "BOUNDED_ADAPTIVE":
        novelty_component = 1.0 - 0.55 * min(ratio, 1.8)
        weight = 0.50 * novelty_component + 0.30 * similarity + 0.20 * stability
    else:
        raise ValueError(f"Política de calibración desconocida: {policy}")

    lower, upper = (0.15, 0.90) if policy == "BOUNDED_ADAPTIVE" else (0.05, 0.95)
    return float(np.clip(weight, lower, upper))


def _date_diagnostics(base: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target_date, sample in base.groupby("target_date", sort=True):
        memory_order = sample.sort_values("memory_score_percentile", ascending=False)
        current_order = sample.sort_values("current_intelligence_score", ascending=False)
        memory_top = memory_order.iloc[0]
        current_top = current_order.iloc[0]

        memory_second = (
            _numeric(memory_order.iloc[1]["memory_score_percentile"], 0.0)
            if len(memory_order) > 1
            else 0.0
        )
        current_second = (
            _numeric(current_order.iloc[1]["current_intelligence_score"], 0.0)
            if len(current_order) > 1
            else 0.0
        )
        memory_margin = max(
            _numeric(memory_top["memory_score_percentile"]) - memory_second, 0.0
        )
        current_margin = max(
            _numeric(current_top["current_intelligence_score"]) - current_second, 0.0
        )

        vectors = sample[["memory_score_percentile", "current_intelligence_score"]].apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0.0)
        vector_disagreement = float(
            (vectors["memory_score_percentile"] - vectors["current_intelligence_score"])
            .abs()
            .mean()
            / 100.0
        )
        top_mismatch = float(memory_top["champion"] != current_top["champion"])
        disagreement = 100.0 * np.clip(
            0.65 * vector_disagreement + 0.35 * top_mismatch, 0.0, 1.0
        )

        rows.append(
            {
                "target_date": target_date,
                "memory_top_champion": memory_top["champion"],
                "current_top_champion": current_top["champion"],
                "memory_internal_margin": memory_margin,
                "memory_internal_ambiguity": 100.0 - min(memory_margin, 100.0),
                "current_intelligence_margin": current_margin,
                "current_intelligence_consistency": min(current_margin, 100.0),
                "memory_current_disagreement": disagreement,
                "top_recommendations_agree": bool(top_mismatch == 0.0),
            }
        )
    return pd.DataFrame(rows)


def build_calibrated_allocations(
    base_allocations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if base_allocations.empty:
        return pd.DataFrame(), pd.DataFrame()
    base = base_allocations.loc[base_allocations["policy"] == "MEMORY_ONLY"].copy()
    if base.empty:
        raise ValueError("No existe la política MEMORY_ONLY para calibrar asignaciones")
    diagnostics = _date_diagnostics(base)
    base = base.merge(diagnostics, on="target_date", how="left")

    frames: list[pd.DataFrame] = []
    for policy in CALIBRATION_POLICIES:
        sample = base.copy()
        sample["calibration_policy"] = policy
        sample["memory_weight"] = sample.apply(
            lambda row: memory_weight_from_curve(row, policy), axis=1
        )
        sample["current_intelligence_weight"] = 1.0 - sample["memory_weight"]
        sample["allocation_score"] = (
            sample["memory_weight"] * sample["memory_score_percentile"]
            + sample["current_intelligence_weight"]
            * sample["current_intelligence_score"]
        )
        sample["allocation_rank"] = sample.groupby("target_date")[
            "allocation_score"
        ].rank(ascending=False, method="first")
        frames.append(sample)
    return pd.concat(frames, ignore_index=True), diagnostics


def build_calibration_recommendations(
    allocations: pd.DataFrame,
    date_quality: pd.DataFrame,
) -> pd.DataFrame:
    if allocations.empty:
        return pd.DataFrame()
    winners = allocations.loc[allocations["allocation_rank"] == 1].copy()
    quality = date_quality.rename(
        columns={"origin_date": "target_date", "quality_score": "quality_score_actual"}
    )
    oracle = (
        quality.sort_values(
            ["target_date", "quality_score_actual"], ascending=[True, False]
        )
        .groupby("target_date", as_index=False)
        .first()[["target_date", "champion", "quality_score_actual"]]
        .rename(
            columns={
                "champion": "oracle_champion",
                "quality_score_actual": "oracle_quality",
            }
        )
    )
    universe = quality.loc[
        quality["champion"] == "UNIVERSE",
        ["target_date", "quality_score_actual"],
    ].rename(columns={"quality_score_actual": "universe_quality"})
    result = winners.merge(oracle, on="target_date", how="left").merge(
        universe, on="target_date", how="left"
    )
    result["advantage_vs_universe"] = (
        result["actual_quality_score"] - result["universe_quality"]
    )
    result["oracle_regret"] = result["oracle_quality"] - result["actual_quality_score"]
    result["selected_was_oracle"] = result["champion"] == result["oracle_champion"]

    strength = pd.to_numeric(result["allocation_score"], errors="coerce").clip(0, 100)
    agreement = 100.0 - pd.to_numeric(
        result["memory_current_disagreement"], errors="coerce"
    ).clip(0, 100)
    novelty_confidence = 100.0 / (
        1.0
        + pd.to_numeric(result["global_novelty_score"], errors="coerce").clip(
            lower=0
        )
    )
    memory_certainty = 100.0 - pd.to_numeric(
        result["memory_internal_ambiguity"], errors="coerce"
    ).clip(0, 100)
    current_consistency = pd.to_numeric(
        result["current_intelligence_consistency"], errors="coerce"
    ).clip(0, 100)
    result["adaptive_conviction_index"] = (
        0.35 * strength
        + 0.25 * agreement
        + 0.15 * novelty_confidence
        + 0.125 * memory_certainty
        + 0.125 * current_consistency
    ).clip(0, 100)

    result = result.rename(columns={"champion": "recommended_champion"})
    if result.columns.duplicated().any():
        duplicated = result.columns[result.columns.duplicated()].tolist()
        raise ValueError(
            f"Columnas duplicadas en recomendaciones calibradas: {duplicated}"
        )
    return result.sort_values(
        ["target_date", "calibration_policy"]
    ).reset_index(drop=True)


def compare_calibration_policies(recommendations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for policy, sample in recommendations.groupby("calibration_policy", sort=False):
        advantage = pd.to_numeric(sample["advantage_vs_universe"], errors="coerce")
        regret = pd.to_numeric(sample["oracle_regret"], errors="coerce")
        rows.append(
            {
                "calibration_policy": policy,
                "dates": int(len(sample)),
                "mean_memory_weight": float(sample["memory_weight"].mean()),
                "minimum_memory_weight": float(sample["memory_weight"].min()),
                "maximum_memory_weight": float(sample["memory_weight"].max()),
                "mean_adaptive_conviction_index": float(
                    sample["adaptive_conviction_index"].mean()
                ),
                "mean_advantage_vs_universe": float(advantage.mean()),
                "positive_advantage_rate": float((advantage > 0).mean()),
                "mean_oracle_regret": float(regret.mean()),
                "maximum_oracle_regret": float(regret.max()),
                "oracle_hit_rate": float(sample["selected_was_oracle"].mean()),
            }
        )
    comparison = pd.DataFrame(rows)
    if comparison.empty:
        return comparison
    comparison["calibration_policy_score"] = 100.0 * (
        0.30 * comparison["positive_advantage_rate"].fillna(0.0)
        + 0.25
        * (comparison["mean_advantage_vs_universe"].clip(-50, 50) + 50)
        / 100
        + 0.20 * (1.0 - comparison["mean_oracle_regret"].clip(0, 100) / 100)
        + 0.15
        * (1.0 - comparison["maximum_oracle_regret"].clip(0, 100) / 100)
        + 0.10 * comparison["oracle_hit_rate"].fillna(0.0)
    )
    return comparison.sort_values(
        "calibration_policy_score", ascending=False
    ).reset_index(drop=True)


def run_allocation_calibration_laboratory(
    daily_states: pd.DataFrame,
    selections: pd.DataFrame,
    *,
    ks: tuple[int, ...] = (3, 5, 7, 10),
    baseline_k: int = 5,
    minimum_history: int = 10,
    calibration_history: int = 6,
    novelty_percentile: float = 0.80,
) -> AllocationCalibrationResult:
    base = run_adaptive_intelligence_allocation_laboratory(
        daily_states,
        selections,
        ks=ks,
        baseline_k=baseline_k,
        minimum_history=minimum_history,
        calibration_history=calibration_history,
        novelty_percentile=novelty_percentile,
    )
    allocations, disagreement = build_calibrated_allocations(
        base.allocation_by_date
    )
    date_quality = build_date_champion_quality(selections)
    recommendations = build_calibration_recommendations(
        allocations, date_quality
    )
    comparison = compare_calibration_policies(recommendations)
    calibration_report = recommendations[
        [
            "target_date",
            "calibration_policy",
            "memory_weight",
            "current_intelligence_weight",
            "global_novelty_score",
            "novelty_veto_threshold",
            "recommended_champion",
            "advantage_vs_universe",
            "oracle_regret",
        ]
    ].copy()
    conviction_report = recommendations[
        [
            "target_date",
            "calibration_policy",
            "recommended_champion",
            "adaptive_conviction_index",
            "allocation_score",
            "memory_current_disagreement",
            "memory_internal_ambiguity",
            "current_intelligence_consistency",
            "advantage_vs_universe",
            "oracle_regret",
        ]
    ].copy()
    summary = comparison.head(1).copy()
    if not summary.empty:
        summary = summary.rename(
            columns={
                "calibration_policy": "recommended_calibration_policy"
            }
        )
    return AllocationCalibrationResult(
        recommendations=recommendations,
        allocation_by_date=allocations,
        policy_comparison=comparison,
        calibration_report=calibration_report,
        conviction_report=conviction_report,
        disagreement_report=disagreement,
        summary=summary,
    )

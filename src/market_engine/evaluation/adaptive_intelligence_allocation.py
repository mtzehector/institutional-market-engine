from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from market_engine.evaluation.market_memory import run_market_memory_laboratory
from market_engine.evaluation.market_memory_confidence import (
    run_market_memory_confidence_laboratory,
)
from market_engine.evaluation.market_memory_decision import add_causal_thresholds


ALLOCATION_POLICIES = (
    "MEMORY_ONLY",
    "NOVELTY_LINEAR",
    "CONFIDENCE_BLEND",
    "ADAPTIVE_ALLOCATION",
)

CHAMPION_FACTOR_MAP: dict[str, tuple[str, ...]] = {
    "CHAMPION_VOLUME_CONFIDENCE_STRUCTURE": ("VOLUME", "STRUCTURE"),
    "PARSIMONIOUS_MOMENTUM_VOLUME": ("MOMENTUM", "VOLUME"),
    "INSTITUTIONAL_STABLE": ("SMART_MONEY", "DEFENSIVE"),
    "INSTITUTIONAL_CONFIRMED": ("VOLUME", "SMART_MONEY"),
    "CONTROL_VOLUME": ("VOLUME",),
    "CONTROL_MOMENTUM": ("MOMENTUM",),
    "CONTROL_SMART_MONEY": ("SMART_MONEY",),
}


@dataclass(frozen=True)
class AdaptiveIntelligenceAllocationResult:
    recommendations: pd.DataFrame
    allocation_by_date: pd.DataFrame
    policy_comparison: pd.DataFrame
    intelligence_states: pd.DataFrame
    champion_scores: pd.DataFrame
    summary: pd.DataFrame


def _numeric(value: object, default: float = 0.0) -> float:
    return float(value) if pd.notna(value) else default


def _causal_percentile_rank(history: pd.Series, value: object) -> float:
    numeric = pd.to_numeric(history, errors="coerce").dropna()
    current = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if numeric.empty or pd.isna(current):
        return 0.5
    return float((numeric <= current).mean())


def build_current_intelligence_states(
    daily_states: pd.DataFrame,
    *,
    minimum_history: int = 6,
) -> pd.DataFrame:
    """Build causal 0-100 scores for non-memory intelligence sources."""
    states = daily_states.copy()
    states["origin_date"] = pd.to_datetime(states["origin_date"])
    states = states.sort_values("origin_date").drop_duplicates("origin_date", keep="last")
    rows: list[dict[str, object]] = []

    for position in range(minimum_history, len(states)):
        current = states.iloc[position]
        history = states.iloc[:position]

        qqq_rank = _causal_percentile_rank(
            history["qqq_return_1d_pct"], current["qqq_return_1d_pct"]
        )
        breadth_1d_rank = _causal_percentile_rank(
            history["breadth_positive_1d"], current["breadth_positive_1d"]
        )
        breadth_5d_rank = _causal_percentile_rank(
            history["breadth_positive_5d"], current["breadth_positive_5d"]
        )
        volume_rank = _causal_percentile_rank(
            history["median_relative_volume"], current["median_relative_volume"]
        )
        atr_rank = _causal_percentile_rank(
            history["median_atr_pct"], current["median_atr_pct"]
        )
        institutional_rank = _causal_percentile_rank(
            history["institutional_breadth"], current["institutional_breadth"]
        )

        momentum = 100.0 * (
            0.35 * qqq_rank + 0.30 * breadth_1d_rank + 0.35 * breadth_5d_rank
        )
        volume = 100.0 * (0.65 * volume_rank + 0.35 * breadth_1d_rank)
        smart_money = 100.0 * institutional_rank
        defensive = 100.0 * (
            0.45 * (1.0 - atr_rank)
            + 0.30 * institutional_rank
            + 0.25 * (1.0 - max(0.0, 0.5 - breadth_5d_rank) * 2.0)
        )
        structure = 100.0 * (
            0.40 * breadth_1d_rank + 0.35 * breadth_5d_rank + 0.25 * qqq_rank
        )

        rows.append(
            {
                "target_date": current["origin_date"],
                "score_MOMENTUM": momentum,
                "score_VOLUME": volume,
                "score_SMART_MONEY": smart_money,
                "score_DEFENSIVE": defensive,
                "score_STRUCTURE": structure,
                "qqq_percentile": qqq_rank,
                "breadth_1d_percentile": breadth_1d_rank,
                "breadth_5d_percentile": breadth_5d_rank,
                "volume_percentile": volume_rank,
                "atr_percentile": atr_rank,
                "institutional_percentile": institutional_rank,
            }
        )
    return pd.DataFrame(rows)


def score_champions_from_current_state(
    intelligence_states: pd.DataFrame,
    champions: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, state in intelligence_states.iterrows():
        for champion in champions:
            factors = CHAMPION_FACTOR_MAP.get(champion)
            if not factors:
                continue
            values = [_numeric(state.get(f"score_{factor}"), 50.0) for factor in factors]
            rows.append(
                {
                    "target_date": state["target_date"],
                    "champion": champion,
                    "current_intelligence_score": float(np.mean(values)),
                    "intelligence_factors": "+".join(factors),
                }
            )
    return pd.DataFrame(rows)


def _normalize_memory_scores(scores: pd.DataFrame) -> pd.DataFrame:
    data = scores.copy()
    data["memory_score_percentile"] = data.groupby("target_date")["memory_score"].rank(
        pct=True, ascending=True, method="average"
    ) * 100.0
    return data


def _memory_weight(row: pd.Series, policy: str) -> float:
    if policy == "MEMORY_ONLY":
        return 1.0

    confidence = np.clip(_numeric(row.get("memory_confidence_score"), 50.0) / 100.0, 0.0, 1.0)
    similarity = np.clip(_numeric(row.get("similarity_confidence"), 50.0) / 100.0, 0.0, 1.0)
    stability = np.clip(_numeric(row.get("outcome_stability"), 50.0) / 100.0, 0.0, 1.0)
    novelty = max(_numeric(row.get("global_novelty_score"), 0.0), 0.0)
    threshold = _numeric(row.get("novelty_veto_threshold"), np.nan)
    if pd.notna(threshold) and threshold > 0:
        novelty_ratio = novelty / threshold
        novelty_weight = float(np.clip(1.0 - 0.60 * novelty_ratio, 0.05, 0.95))
    else:
        novelty_weight = similarity

    if policy == "NOVELTY_LINEAR":
        return novelty_weight
    if policy == "CONFIDENCE_BLEND":
        return float(np.clip(confidence, 0.10, 0.90))
    if policy == "ADAPTIVE_ALLOCATION":
        return float(
            np.clip(
                0.50 * novelty_weight + 0.30 * similarity + 0.20 * stability,
                0.05,
                0.95,
            )
        )
    raise ValueError(f"Política desconocida: {policy}")


def build_allocations(
    recommendations: pd.DataFrame,
    memory_scores: pd.DataFrame,
    current_scores: pd.DataFrame,
    date_quality: pd.DataFrame,
) -> pd.DataFrame:
    memory = _normalize_memory_scores(memory_scores)
    combined = memory.merge(current_scores, on=["target_date", "champion"], how="inner")
    combined = combined.merge(
        recommendations,
        on="target_date",
        how="inner",
        suffixes=("", "_recommendation"),
    )
    actual = date_quality.rename(
        columns={"origin_date": "target_date", "quality_score": "actual_quality_score"}
    )[["target_date", "champion", "actual_quality_score"]]
    combined = combined.merge(actual, on=["target_date", "champion"], how="left")

    rows: list[dict[str, object]] = []
    for policy in ALLOCATION_POLICIES:
        sample = combined.copy()
        sample["memory_weight"] = sample.apply(lambda row: _memory_weight(row, policy), axis=1)
        sample["current_intelligence_weight"] = 1.0 - sample["memory_weight"]
        sample["allocation_score"] = (
            sample["memory_weight"] * sample["memory_score_percentile"]
            + sample["current_intelligence_weight"] * sample["current_intelligence_score"]
        )
        sample["allocation_rank"] = sample.groupby("target_date")["allocation_score"].rank(
            ascending=False, method="first"
        )
        sample["policy"] = policy
        rows.append(sample)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_policy_recommendations(
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
        quality.sort_values(["target_date", "quality_score_actual"], ascending=[True, False])
        .groupby("target_date", as_index=False)
        .first()[["target_date", "champion", "quality_score_actual"]]
        .rename(
            columns={
                "champion": "oracle_champion",
                "quality_score_actual": "oracle_quality",
            }
        )
    )
    universe = quality.loc[quality["champion"] == "UNIVERSE", ["target_date", "quality_score_actual"]].rename(
        columns={"quality_score_actual": "universe_quality"}
    )
    result = winners.merge(oracle, on="target_date", how="left").merge(
        universe, on="target_date", how="left"
    )
    result["advantage_vs_universe"] = result["actual_quality_score"] - result["universe_quality"]
    result["oracle_regret"] = result["oracle_quality"] - result["actual_quality_score"]
    result["selected_was_oracle"] = result["champion"] == result["oracle_champion"]
    result = result.rename(columns={"champion": "recommended_champion"})
    return result.sort_values(["target_date", "policy"]).reset_index(drop=True)


def compare_allocation_policies(recommendations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for policy, sample in recommendations.groupby("policy", sort=False):
        advantage = pd.to_numeric(sample["advantage_vs_universe"], errors="coerce")
        regret = pd.to_numeric(sample["oracle_regret"], errors="coerce")
        rows.append(
            {
                "policy": policy,
                "dates": int(len(sample)),
                "mean_memory_weight": float(sample["memory_weight"].mean()),
                "minimum_memory_weight": float(sample["memory_weight"].min()),
                "maximum_memory_weight": float(sample["memory_weight"].max()),
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
    comparison["allocation_policy_score"] = 100.0 * (
        0.35 * comparison["positive_advantage_rate"].fillna(0.0)
        + 0.25 * (comparison["mean_advantage_vs_universe"].clip(-50, 50) + 50) / 100
        + 0.25 * (1.0 - comparison["mean_oracle_regret"].clip(0, 100) / 100)
        + 0.15 * (1.0 - comparison["maximum_oracle_regret"].clip(0, 100) / 100)
    )
    return comparison.sort_values("allocation_policy_score", ascending=False).reset_index(drop=True)


def run_adaptive_intelligence_allocation_laboratory(
    daily_states: pd.DataFrame,
    selections: pd.DataFrame,
    *,
    ks: tuple[int, ...] = (3, 5, 7, 10),
    baseline_k: int = 5,
    minimum_history: int = 10,
    calibration_history: int = 6,
    novelty_percentile: float = 0.80,
) -> AdaptiveIntelligenceAllocationResult:
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
    )
    memory = run_market_memory_laboratory(
        daily_states,
        selections,
        neighbors=baseline_k,
        minimum_history=max(minimum_history, baseline_k),
    )
    intelligence_states = build_current_intelligence_states(
        daily_states, minimum_history=minimum_history
    )
    champions = sorted(
        set(memory.champion_scores["champion"].astype(str)) - {"UNIVERSE"}
    )
    current_scores = score_champions_from_current_state(intelligence_states, champions)
    allocations = build_allocations(
        recommendations,
        memory.champion_scores,
        current_scores,
        memory.date_champion_quality,
    )
    policy_recommendations = build_policy_recommendations(
        allocations, memory.date_champion_quality
    )
    comparison = compare_allocation_policies(policy_recommendations)
    summary = comparison.head(1).copy()
    if not summary.empty:
        summary = summary.rename(columns={"policy": "recommended_allocation_policy"})
    return AdaptiveIntelligenceAllocationResult(
        recommendations=policy_recommendations,
        allocation_by_date=allocations,
        policy_comparison=comparison,
        intelligence_states=intelligence_states,
        champion_scores=current_scores,
        summary=summary,
    )

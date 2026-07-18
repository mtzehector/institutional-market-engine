from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from market_engine.evaluation.adaptive_intelligence_allocation import (
    run_adaptive_intelligence_allocation_laboratory,
)
from market_engine.evaluation.champion_stress import (
    ChampionStressResult,
    run_champion_stress_laboratory,
)
from market_engine.evaluation.survival_utility import (
    SurvivalUtilityResult,
    run_survival_utility_laboratory,
)

COUNCIL_POLICIES = ("CONSERVATIVE", "BALANCED", "EXPLORATORY")

POLICY_CONFIG: dict[str, dict[str, float]] = {
    "CONSERVATIVE": {
        "stress_veto": 65.0,
        "stress_cap": 42.0,
        "evidence_penalty_low": 0.72,
        "evidence_penalty_medium": 0.90,
        "exploration_bonus": 0.00,
    },
    "BALANCED": {
        "stress_veto": 78.0,
        "stress_cap": 58.0,
        "evidence_penalty_low": 0.82,
        "evidence_penalty_medium": 0.94,
        "exploration_bonus": 0.00,
    },
    "EXPLORATORY": {
        "stress_veto": 90.0,
        "stress_cap": 72.0,
        "evidence_penalty_low": 0.92,
        "evidence_penalty_medium": 0.98,
        "exploration_bonus": 6.0,
    },
}

LIFECYCLE_VOTE = {
    "MATURE": 95.0,
    "RECOVERING": 76.0,
    "EMERGING": 66.0,
    "DISCOVERY": 48.0,
    "DETERIORATING": 22.0,
    "OBSOLETE": 5.0,
}

ENGINE_BASE_WEIGHTS = {
    "LIFECYCLE": 0.20,
    "UTILITY": 0.24,
    "SURVIVAL": 0.14,
    "RESILIENCE": 0.15,
    "STRESS": 0.20,
    "MEMORY_ALLOCATION": 0.07,
}

EVIDENCE_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}


@dataclass(frozen=True)
class ChampionIntelligenceCouncilResult:
    council_current: pd.DataFrame
    engine_votes: pd.DataFrame
    explanations: pd.DataFrame
    disagreements: pd.DataFrame
    policy_comparison: pd.DataFrame
    summary: pd.DataFrame
    stress_result: ChampionStressResult
    utility_result: SurvivalUtilityResult


def _numeric(value: object, default: float = 50.0) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else default


def _evidence_multiplier(strength: str, policy: str) -> float:
    config = POLICY_CONFIG[policy]
    if strength == "HIGH":
        return 1.0
    if strength == "MEDIUM":
        return config["evidence_penalty_medium"]
    return config["evidence_penalty_low"]


def _combine_evidence(*strengths: object) -> str:
    valid = [str(value) for value in strengths if str(value) in EVIDENCE_RANK]
    if not valid:
        return "LOW"
    return min(valid, key=lambda value: EVIDENCE_RANK[value])


def _decision(score: float, veto: bool) -> str:
    if veto or score < 25:
        return "SUSPEND_OR_AVOID"
    if score < 45:
        return "REDUCE_AUTHORITY"
    if score < 62:
        return "MAINTAIN_CAUTIOUSLY"
    if score < 78:
        return "INCREASE_GRADUALLY"
    return "INCREASE_AUTHORITY"


def _confidence(agreement: float, evidence: str, veto: bool) -> float:
    evidence_component = {"LOW": 45.0, "MEDIUM": 72.0, "HIGH": 92.0}.get(evidence, 45.0)
    score = 0.65 * agreement + 0.35 * evidence_component
    if veto:
        score = max(score, 80.0)
    return float(np.clip(score, 0.0, 100.0))


def _reason(engine: str, score: float, row: pd.Series) -> str:
    if engine == "LIFECYCLE":
        return f"Estado {row.get('lifecycle_state')} con salud {row.get('lifecycle_health_score', np.nan):.1f}."
    if engine == "UTILITY":
        return f"Utilidad ajustada {score:.1f}; perspectiva {row.get('utility_outlook', 'N/D')}."
    if engine == "SURVIVAL":
        return f"Persistencia útil de mediano plazo {score:.1f}."
    if engine == "RESILIENCE":
        return f"Resiliencia {score:.1f}; evidencia {row.get('resilience_evidence_strength', 'LOW')}."
    if engine == "STRESS":
        return f"Seguridad frente al estrés {score:.1f}; estrés activo {row.get('current_active_stress_score', np.nan):.1f}."
    return f"Asignación adaptativa respalda al campeón con {score:.1f}."


def _memory_votes(
    daily_states: pd.DataFrame,
    selections: pd.DataFrame,
    champions: list[str],
    *,
    ks: tuple[int, ...],
    baseline_k: int,
    memory_minimum_history: int,
    calibration_history: int,
    novelty_percentile: float,
) -> pd.DataFrame:
    required = {
        "qqq_return_1d_pct",
        "breadth_positive_1d",
        "breadth_positive_5d",
        "median_relative_volume",
        "median_atr_pct",
        "institutional_breadth",
    }
    if daily_states.empty or not required.issubset(daily_states.columns):
        return pd.DataFrame({"champion": champions, "memory_allocation_vote": 50.0})

    allocation = run_adaptive_intelligence_allocation_laboratory(
        daily_states,
        selections,
        ks=ks,
        baseline_k=baseline_k,
        minimum_history=memory_minimum_history,
        calibration_history=calibration_history,
        novelty_percentile=novelty_percentile,
    )
    data = allocation.allocation_by_date
    if data.empty:
        return pd.DataFrame({"champion": champions, "memory_allocation_vote": 50.0})
    sample = data.loc[data["policy"] == "ADAPTIVE_ALLOCATION"].copy()
    latest_date = pd.to_datetime(sample["target_date"]).max()
    sample = sample.loc[pd.to_datetime(sample["target_date"]) == latest_date]
    values = sample[["champion", "allocation_score"]].rename(
        columns={"allocation_score": "memory_allocation_vote"}
    )
    result = pd.DataFrame({"champion": champions}).merge(values, on="champion", how="left")
    result["memory_allocation_vote"] = pd.to_numeric(
        result["memory_allocation_vote"], errors="coerce"
    ).fillna(50.0).clip(0, 100)
    return result


def build_council_inputs(
    stress: ChampionStressResult,
    utility: SurvivalUtilityResult,
    memory_votes: pd.DataFrame,
) -> pd.DataFrame:
    stress_current = stress.current_stress.copy()
    utility_current = utility.current_utility.copy()
    resilience = stress.resilience_result.current_resilience.copy()

    utility_columns = [
        "champion",
        "utility_adjusted_authority",
        "utility_outlook",
        "medium_term_authority",
        "medium_term_survival_probability",
        "favorable_state_survival",
        "adverse_state_persistence",
        "evidence_strength",
    ]
    resilience_columns = [
        "champion",
        "resilience_score",
        "resilience_outlook",
        "evidence_strength",
        "resilience_reference_scope",
        "completed_recovery_episodes",
    ]
    utility_current = utility_current[utility_columns].rename(
        columns={"evidence_strength": "utility_evidence_strength"}
    )
    resilience = resilience[resilience_columns].rename(
        columns={"evidence_strength": "resilience_evidence_strength"}
    )

    combined = stress_current.merge(utility_current, on="champion", how="left")
    combined = combined.merge(resilience, on="champion", how="left")
    combined = combined.merge(memory_votes, on="champion", how="left")
    combined["memory_allocation_vote"] = combined["memory_allocation_vote"].fillna(50.0)
    return combined


def deliberate_council(inputs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    council_rows: list[dict[str, object]] = []
    vote_rows: list[dict[str, object]] = []
    explanation_rows: list[dict[str, object]] = []

    for policy in COUNCIL_POLICIES:
        config = POLICY_CONFIG[policy]
        for _, row in inputs.iterrows():
            lifecycle_vote = LIFECYCLE_VOTE.get(str(row.get("lifecycle_state")), 40.0)
            utility_vote = _numeric(row.get("utility_adjusted_authority"))
            survival_vote = _numeric(row.get("medium_term_authority"))
            resilience_vote = _numeric(row.get("resilience_score"))
            stress_vote = 100.0 - _numeric(row.get("stress_risk_score"))
            memory_vote = _numeric(row.get("memory_allocation_vote"))

            votes = {
                "LIFECYCLE": lifecycle_vote,
                "UTILITY": utility_vote,
                "SURVIVAL": survival_vote,
                "RESILIENCE": resilience_vote,
                "STRESS": stress_vote,
                "MEMORY_ALLOCATION": memory_vote,
            }
            evidence_by_engine = {
                "LIFECYCLE": "HIGH",
                "UTILITY": str(row.get("utility_evidence_strength", "LOW")),
                "SURVIVAL": str(row.get("utility_evidence_strength", "LOW")),
                "RESILIENCE": str(row.get("resilience_evidence_strength", "LOW")),
                "STRESS": "MEDIUM" if _numeric(row.get("completed_stress_episodes"), 0) >= 2 else "LOW",
                "MEMORY_ALLOCATION": "MEDIUM",
            }

            weighted_sum = 0.0
            effective_weight = 0.0
            for engine, vote in votes.items():
                weight = ENGINE_BASE_WEIGHTS[engine] * _evidence_multiplier(
                    evidence_by_engine[engine], policy
                )
                weighted_sum += vote * weight
                effective_weight += weight
                vote_rows.append(
                    {
                        "policy": policy,
                        "champion": row["champion"],
                        "engine": engine,
                        "vote_score": vote,
                        "base_weight": ENGINE_BASE_WEIGHTS[engine],
                        "effective_weight": weight,
                        "evidence_strength": evidence_by_engine[engine],
                        "vote_reason": _reason(engine, vote, row),
                    }
                )

            raw_score = weighted_sum / effective_weight if effective_weight else 50.0
            raw_score += config["exploration_bonus"] if str(row.get("lifecycle_state")) in {"EMERGING", "RECOVERING"} else 0.0

            active_stress = _numeric(row.get("current_active_stress_score"), 0.0)
            obsolete = str(row.get("lifecycle_state")) == "OBSOLETE"
            risk_veto = bool(active_stress >= config["stress_veto"] or obsolete)
            veto_reason = None
            score = raw_score
            if risk_veto:
                score = min(score, config["stress_cap"])
                veto_reason = (
                    "OBSOLETE" if obsolete else f"ACTIVE_STRESS_{active_stress:.1f}"
                )

            vote_values = np.array(list(votes.values()), dtype=float)
            agreement = float(np.clip(100.0 - 2.0 * vote_values.std(ddof=0), 0.0, 100.0))
            evidence = _combine_evidence(
                row.get("utility_evidence_strength"),
                row.get("resilience_evidence_strength"),
            )
            confidence = _confidence(agreement, evidence, risk_veto)
            decision = _decision(score, risk_veto)

            strongest_engine = max(votes, key=votes.get)
            warning_engine = min(votes, key=votes.get)
            supporting = [engine for engine, value in votes.items() if value >= 65]
            warnings = [engine for engine, value in votes.items() if value < 40]

            council_rows.append(
                {
                    "policy": policy,
                    "champion": row["champion"],
                    "lifecycle_state": row.get("lifecycle_state"),
                    "council_authority_score": float(np.clip(score, 0, 100)),
                    "council_decision": decision,
                    "council_confidence": confidence,
                    "council_agreement": agreement,
                    "risk_veto": risk_veto,
                    "risk_veto_reason": veto_reason,
                    "dominant_supporting_engine": strongest_engine,
                    "dominant_warning_engine": warning_engine,
                    "supporting_engines": ", ".join(supporting) if supporting else "NONE",
                    "warning_engines": ", ".join(warnings) if warnings else "NONE",
                    "evidence_strength": evidence,
                    "current_active_stress_score": active_stress,
                    "utility_adjusted_authority": utility_vote,
                    "resilience_score": resilience_vote,
                    "memory_allocation_vote": memory_vote,
                }
            )
            explanation_rows.append(
                {
                    "policy": policy,
                    "champion": row["champion"],
                    "decision": decision,
                    "supporting_arguments": " | ".join(
                        _reason(engine, votes[engine], row) for engine in supporting
                    ) or "Sin argumentos fuertes.",
                    "warning_arguments": " | ".join(
                        _reason(engine, votes[engine], row) for engine in warnings
                    ) or "Sin advertencias fuertes.",
                    "veto_explanation": veto_reason,
                }
            )

    council = pd.DataFrame(council_rows).sort_values(
        ["policy", "council_authority_score"], ascending=[True, False]
    ).reset_index(drop=True)
    votes_frame = pd.DataFrame(vote_rows)
    explanations = pd.DataFrame(explanation_rows)
    return council, votes_frame, explanations


def build_disagreements(votes: pd.DataFrame) -> pd.DataFrame:
    if votes.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (policy, champion), sample in votes.groupby(["policy", "champion"], sort=True):
        ordered = sample.sort_values("vote_score")
        low = ordered.iloc[0]
        high = ordered.iloc[-1]
        spread = float(high["vote_score"] - low["vote_score"])
        rows.append(
            {
                "policy": policy,
                "champion": champion,
                "highest_engine": high["engine"],
                "highest_vote": high["vote_score"],
                "lowest_engine": low["engine"],
                "lowest_vote": low["vote_score"],
                "vote_spread": spread,
                "disagreement_level": "HIGH" if spread >= 45 else "MEDIUM" if spread >= 25 else "LOW",
            }
        )
    return pd.DataFrame(rows).sort_values("vote_spread", ascending=False).reset_index(drop=True)


def compare_policies(council: pd.DataFrame) -> pd.DataFrame:
    if council.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for policy, sample in council.groupby("policy", sort=False):
        rows.append(
            {
                "policy": policy,
                "mean_authority": float(sample["council_authority_score"].mean()),
                "mean_confidence": float(sample["council_confidence"].mean()),
                "mean_agreement": float(sample["council_agreement"].mean()),
                "risk_vetoes": int(sample["risk_veto"].sum()),
                "increase_recommendations": int(sample["council_decision"].isin(["INCREASE_GRADUALLY", "INCREASE_AUTHORITY"]).sum()),
                "reduce_or_suspend": int(sample["council_decision"].isin(["REDUCE_AUTHORITY", "SUSPEND_OR_AVOID"]).sum()),
                "top_champion": sample.sort_values("council_authority_score", ascending=False).iloc[0]["champion"],
                "top_authority": float(sample["council_authority_score"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_authority", ascending=False).reset_index(drop=True)


def run_champion_intelligence_council(
    selections: pd.DataFrame,
    daily_states: pd.DataFrame,
    *,
    short_window: int = 3,
    long_window: int = 6,
    minimum_history: int = 4,
    minimum_state_persistence: int = 2,
    horizons: tuple[int, ...] = (1, 3, 5),
    minimum_completed_spells: int = 3,
    minimum_own_episodes: int = 3,
    minimum_family_episodes: int = 5,
    trigger_lookback: int = 2,
    relapse_horizon: int = 3,
    ks: tuple[int, ...] = (3, 5, 7, 10),
    baseline_k: int = 5,
    memory_minimum_history: int = 10,
    calibration_history: int = 6,
    novelty_percentile: float = 0.80,
) -> ChampionIntelligenceCouncilResult:
    stress = run_champion_stress_laboratory(
        selections,
        daily_states,
        short_window=short_window,
        long_window=long_window,
        minimum_history=minimum_history,
        minimum_state_persistence=minimum_state_persistence,
        minimum_own_episodes=minimum_own_episodes,
        minimum_family_episodes=minimum_family_episodes,
        trigger_lookback=trigger_lookback,
        relapse_horizon=relapse_horizon,
    )
    utility = run_survival_utility_laboratory(
        selections,
        daily_states,
        short_window=short_window,
        long_window=long_window,
        minimum_history=minimum_history,
        minimum_state_persistence=minimum_state_persistence,
        horizons=horizons,
        minimum_completed_spells=minimum_completed_spells,
    )
    champions = sorted(stress.current_stress["champion"].astype(str).unique().tolist())
    memory = _memory_votes(
        daily_states,
        selections,
        champions,
        ks=ks,
        baseline_k=baseline_k,
        memory_minimum_history=memory_minimum_history,
        calibration_history=calibration_history,
        novelty_percentile=novelty_percentile,
    )
    inputs = build_council_inputs(stress, utility, memory)
    council, votes, explanations = deliberate_council(inputs)
    disagreements = build_disagreements(votes)
    comparison = compare_policies(council)

    summary = pd.DataFrame()
    if not council.empty:
        balanced = council.loc[council["policy"] == "BALANCED"].sort_values(
            "council_authority_score", ascending=False
        )
        summary = pd.DataFrame([
            {
                "champions": int(council["champion"].nunique()),
                "policies": int(council["policy"].nunique()),
                "balanced_top_champion": balanced.iloc[0]["champion"],
                "balanced_top_authority": float(balanced.iloc[0]["council_authority_score"]),
                "balanced_top_decision": balanced.iloc[0]["council_decision"],
                "balanced_mean_agreement": float(balanced["council_agreement"].mean()),
                "balanced_risk_vetoes": int(balanced["risk_veto"].sum()),
                "highest_disagreement_champion": disagreements.iloc[0]["champion"] if not disagreements.empty else None,
            }
        ])

    return ChampionIntelligenceCouncilResult(
        council_current=council,
        engine_votes=votes,
        explanations=explanations,
        disagreements=disagreements,
        policy_comparison=comparison,
        summary=summary,
        stress_result=stress,
        utility_result=utility,
    )

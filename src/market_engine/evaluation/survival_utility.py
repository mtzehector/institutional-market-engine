from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd

from market_engine.evaluation.champion_survival import (
    ChampionSurvivalResult,
    run_champion_survival_laboratory,
)

STATE_UTILITY: dict[str, float] = {
    "MATURE": 1.00,
    "RECOVERING": 0.80,
    "EMERGING": 0.60,
    "DISCOVERY": 0.35,
    "DETERIORATING": 0.15,
    "OBSOLETE": 0.00,
}

FAVORABLE_STATES = {"MATURE", "RECOVERING", "EMERGING"}
ADVERSE_STATES = {"DETERIORATING", "OBSOLETE"}


@dataclass(frozen=True)
class SurvivalUtilityResult:
    utility_history: pd.DataFrame
    current_utility: pd.DataFrame
    evidence_report: pd.DataFrame
    summary: pd.DataFrame
    survival_result: ChampionSurvivalResult


def _wilson_interval(successes: float, support: int, *, z: float = 1.96) -> tuple[float, float]:
    if support <= 0:
        return 0.0, 1.0
    proportion = float(np.clip(successes / support, 0.0, 1.0))
    denominator = 1.0 + (z * z) / support
    centre = proportion + (z * z) / (2.0 * support)
    spread = z * sqrt((proportion * (1.0 - proportion) + (z * z) / (4.0 * support)) / support)
    lower = (centre - spread) / denominator
    upper = (centre + spread) / denominator
    return float(np.clip(lower, 0.0, 1.0)), float(np.clip(upper, 0.0, 1.0))


def _evidence_strength(support: int, completed_spells: int, scope: str) -> str:
    effective = min(support, completed_spells)
    if scope == "CHAMPION" and effective >= 10:
        return "HIGH"
    if effective >= 6:
        return "MEDIUM"
    return "LOW"


def _evidence_multiplier(strength: str) -> float:
    return {"HIGH": 1.0, "MEDIUM": 0.85, "LOW": 0.65}.get(strength, 0.65)


def _mean_probability(row: pd.Series, horizons: tuple[int, ...]) -> float:
    values = [float(row.get(f"survival_probability_{h}", np.nan)) for h in horizons]
    numeric = [value for value in values if np.isfinite(value)]
    return float(np.mean(numeric)) if numeric else 0.5


def build_survival_utility_history(
    survival_history: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = (1, 3, 5),
) -> pd.DataFrame:
    if survival_history.empty:
        return pd.DataFrame()
    ordered = tuple(sorted(set(horizons)))
    short_horizons = tuple(h for h in ordered if h <= 3) or (ordered[0],)
    medium_horizon = ordered[-1]

    rows: list[dict[str, object]] = []
    for _, row in survival_history.iterrows():
        state = str(row.get("lifecycle_state"))
        utility = STATE_UTILITY.get(state, 0.25)
        short_probability = _mean_probability(row, short_horizons)
        medium_probability = float(row.get(f"survival_probability_{medium_horizon}", 0.5))
        support = int(row.get(f"at_risk_support_{medium_horizon}", 0) or 0)
        completed = int(row.get("completed_spells_available", 0) or 0)
        scope = str(row.get("survival_reference_scope", "POOLED_STATE"))
        evidence = _evidence_strength(support, completed, scope)
        evidence_multiplier = _evidence_multiplier(evidence)

        successes = medium_probability * support
        lower, upper = _wilson_interval(successes, support)
        health = float(pd.to_numeric(pd.Series([row.get("lifecycle_health_score")]), errors="coerce").fillna(50.0).iloc[0]) / 100.0

        favorable_survival = medium_probability * utility if state in FAVORABLE_STATES else 0.0
        adverse_persistence = medium_probability * (1.0 - utility) if state in ADVERSE_STATES else 0.0

        short_term_authority = 100.0 * (
            0.50 * health
            + 0.35 * short_probability * utility
            + 0.15 * evidence_multiplier
        )
        medium_term_authority = 100.0 * (
            0.40 * health
            + 0.45 * medium_probability * utility
            + 0.15 * evidence_multiplier
        )
        utility_adjusted_authority = 0.60 * short_term_authority + 0.40 * medium_term_authority
        risk_penalty = 35.0 * adverse_persistence
        utility_adjusted_authority = float(np.clip(utility_adjusted_authority - risk_penalty, 0.0, 100.0))

        record = row.to_dict()
        record.update(
            {
                "state_utility": utility,
                "favorable_state_survival": favorable_survival,
                "adverse_state_persistence": adverse_persistence,
                "survival_probability_lower": lower,
                "survival_probability_upper": upper,
                "evidence_strength": evidence,
                "evidence_multiplier": evidence_multiplier,
                "short_term_survival_probability": short_probability,
                "medium_term_survival_probability": medium_probability,
                "short_term_authority": float(np.clip(short_term_authority, 0.0, 100.0)),
                "medium_term_authority": float(np.clip(medium_term_authority, 0.0, 100.0)),
                "utility_adjusted_authority": utility_adjusted_authority,
                "survival_risk_penalty": risk_penalty,
            }
        )
        rows.append(record)

    return pd.DataFrame(rows).sort_values(["origin_date", "champion"]).reset_index(drop=True)


def build_current_survival_utility(utility_history: pd.DataFrame) -> pd.DataFrame:
    if utility_history.empty:
        return pd.DataFrame()
    latest = (
        utility_history.sort_values("origin_date")
        .groupby("champion", as_index=False)
        .tail(1)
        .copy()
    )
    latest["utility_outlook"] = pd.cut(
        latest["utility_adjusted_authority"],
        bins=[-np.inf, 25, 45, 65, np.inf],
        labels=["LOW_UTILITY", "CAUTIOUS", "USEFUL", "HIGH_UTILITY"],
    ).astype(str)
    return latest.sort_values("utility_adjusted_authority", ascending=False).reset_index(drop=True)


def build_evidence_report(current_utility: pd.DataFrame) -> pd.DataFrame:
    if current_utility.empty:
        return pd.DataFrame()
    columns = [
        "champion",
        "lifecycle_state",
        "state_age",
        "survival_reference_scope",
        "completed_spells_available",
        "evidence_strength",
        "survival_probability_lower",
        "medium_term_survival_probability",
        "survival_probability_upper",
        "favorable_state_survival",
        "adverse_state_persistence",
    ]
    return current_utility.loc[:, columns].copy()


def run_survival_utility_laboratory(
    selections: pd.DataFrame,
    daily_states: pd.DataFrame | None = None,
    *,
    short_window: int = 3,
    long_window: int = 6,
    minimum_history: int = 4,
    minimum_state_persistence: int = 2,
    horizons: tuple[int, ...] = (1, 3, 5),
    minimum_completed_spells: int = 3,
) -> SurvivalUtilityResult:
    survival = run_champion_survival_laboratory(
        selections,
        daily_states,
        short_window=short_window,
        long_window=long_window,
        minimum_history=minimum_history,
        minimum_state_persistence=minimum_state_persistence,
        horizons=horizons,
        minimum_completed_spells=minimum_completed_spells,
    )
    history = build_survival_utility_history(survival.survival_history, horizons=horizons)
    current = build_current_survival_utility(history)
    evidence = build_evidence_report(current)

    summary = pd.DataFrame()
    if not current.empty:
        favorable = current["lifecycle_state"].isin(FAVORABLE_STATES)
        adverse = current["lifecycle_state"].isin(ADVERSE_STATES)
        summary = pd.DataFrame(
            [
                {
                    "champions": int(current["champion"].nunique()),
                    "favorable_state_champions": int(favorable.sum()),
                    "adverse_state_champions": int(adverse.sum()),
                    "mean_utility_adjusted_authority": float(current["utility_adjusted_authority"].mean()),
                    "high_utility_champions": int(current["utility_outlook"].eq("HIGH_UTILITY").sum()),
                    "low_evidence_champions": int(current["evidence_strength"].eq("LOW").sum()),
                    "top_champion": current.iloc[0]["champion"],
                    "top_lifecycle_state": current.iloc[0]["lifecycle_state"],
                    "top_utility_outlook": current.iloc[0]["utility_outlook"],
                    "top_utility_adjusted_authority": float(current.iloc[0]["utility_adjusted_authority"]),
                    "highest_adverse_persistence_champion": current.sort_values("adverse_state_persistence", ascending=False).iloc[0]["champion"],
                }
            ]
        )

    return SurvivalUtilityResult(
        utility_history=history,
        current_utility=current,
        evidence_report=evidence,
        summary=summary,
        survival_result=survival,
    )

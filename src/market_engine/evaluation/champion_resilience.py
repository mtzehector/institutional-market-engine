from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from market_engine.evaluation.champion_lifecycle import (
    ChampionLifecycleResult,
    run_champion_lifecycle_laboratory,
)

ADVERSE_STATES = {"DETERIORATING", "OBSOLETE"}
FAVORABLE_STATES = {"EMERGING", "RECOVERING", "MATURE"}
FULL_RECOVERY_STATES = {"MATURE"}

CHAMPION_FAMILIES: dict[str, str] = {
    "CHAMPION_VOLUME_CONFIDENCE_STRUCTURE": "VOLUME_STRUCTURE",
    "PARSIMONIOUS_MOMENTUM_VOLUME": "MOMENTUM_VOLUME",
    "INSTITUTIONAL_STABLE": "INSTITUTIONAL",
    "INSTITUTIONAL_CONFIRMED": "INSTITUTIONAL",
    "CONTROL_VOLUME": "VOLUME_STRUCTURE",
    "CONTROL_MOMENTUM": "MOMENTUM_VOLUME",
    "CONTROL_SMART_MONEY": "INSTITUTIONAL",
}


@dataclass(frozen=True)
class ChampionResilienceResult:
    recovery_episodes: pd.DataFrame
    resilience_history: pd.DataFrame
    current_resilience: pd.DataFrame
    champion_summary: pd.DataFrame
    family_summary: pd.DataFrame
    evidence: pd.DataFrame
    summary: pd.DataFrame
    lifecycle_result: ChampionLifecycleResult


def _family(champion: object) -> str:
    return CHAMPION_FAMILIES.get(str(champion), "OTHER")


def build_recovery_episodes(lifecycle_history: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct adverse episodes and their partial/full recoveries."""
    if lifecycle_history.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for champion, sample in lifecycle_history.groupby("champion", sort=True):
        sample = sample.sort_values("origin_date").reset_index(drop=True)
        episode_id = 0
        index = 0
        while index < len(sample):
            if str(sample.loc[index, "lifecycle_state"]) not in ADVERSE_STATES:
                index += 1
                continue

            start = index
            pre_index = max(start - 1, 0)
            pre_quality = float(sample.loc[pre_index, "quality_score"])
            pre_advantage = float(sample.loc[pre_index, "advantage_vs_universe"])
            end = start
            while end + 1 < len(sample) and str(sample.loc[end + 1, "lifecycle_state"]) in ADVERSE_STATES:
                end += 1

            recovery_index: int | None = None
            search = end + 1
            while search < len(sample):
                state = str(sample.loc[search, "lifecycle_state"])
                if state in FAVORABLE_STATES:
                    recovery_index = search
                    break
                search += 1

            adverse_block = sample.iloc[start : end + 1]
            recovered = recovery_index is not None
            recovery_state = (
                str(sample.loc[recovery_index, "lifecycle_state"]) if recovered else None
            )
            full_recovery = bool(
                recovered
                and recovery_state in FULL_RECOVERY_STATES
                and float(sample.loc[recovery_index, "advantage_vs_universe"]) >= 0
            )
            recovery_duration = (
                int(recovery_index - start + 1) if recovered else int(len(sample) - start)
            )
            worst_advantage = float(
                pd.to_numeric(adverse_block["advantage_vs_universe"], errors="coerce").min()
            )
            worst_quality = float(
                pd.to_numeric(adverse_block["quality_score"], errors="coerce").min()
            )
            recovery_quality = (
                float(sample.loc[recovery_index, "quality_score"]) if recovered else np.nan
            )
            recovery_advantage = (
                float(sample.loc[recovery_index, "advantage_vs_universe"])
                if recovered
                else np.nan
            )
            post_persistence = 0
            if recovered:
                cursor = recovery_index
                while cursor < len(sample) and str(sample.loc[cursor, "lifecycle_state"]) in FAVORABLE_STATES:
                    post_persistence += 1
                    cursor += 1

            depth = max(pre_advantage - worst_advantage, 0.0)
            quality_loss = max(pre_quality - worst_quality, 0.0)
            rows.append(
                {
                    "champion": champion,
                    "family": _family(champion),
                    "recovery_episode_id": episode_id,
                    "adverse_start_date": sample.loc[start, "origin_date"],
                    "adverse_end_date": sample.loc[end, "origin_date"],
                    "recovery_date": sample.loc[recovery_index, "origin_date"] if recovered else pd.NaT,
                    "starting_state": sample.loc[start, "lifecycle_state"],
                    "recovery_state": recovery_state,
                    "recovered": recovered,
                    "full_recovery": full_recovery,
                    "episode_completed": recovered,
                    "recovery_duration_observations": recovery_duration,
                    "adverse_duration_observations": int(end - start + 1),
                    "pre_episode_quality": pre_quality,
                    "pre_episode_advantage": pre_advantage,
                    "worst_quality": worst_quality,
                    "worst_advantage": worst_advantage,
                    "recovery_quality": recovery_quality,
                    "recovery_advantage": recovery_advantage,
                    "recovery_depth": depth,
                    "quality_loss": quality_loss,
                    "post_recovery_persistence": post_persistence,
                    "recovered_previous_quality": bool(
                        recovered and pd.notna(recovery_quality) and recovery_quality >= pre_quality
                    ),
                }
            )
            episode_id += 1
            index = max(end + 1, recovery_index + 1 if recovery_index is not None else end + 1)

    return pd.DataFrame(rows)


def _smoothed_probability(successes: int, total: int, *, prior: float = 0.5, strength: float = 2.0) -> float:
    return float((successes + prior * strength) / (total + strength))


def _evidence_strength(total: int) -> str:
    if total >= 12:
        return "HIGH"
    if total >= 6:
        return "MEDIUM"
    return "LOW"


def _summarize_reference(reference: pd.DataFrame) -> dict[str, object]:
    completed = reference.loc[reference["episode_completed"]].copy()
    total = int(len(completed))
    recovered = int(completed["recovered"].sum()) if total else 0
    full = int(completed["full_recovery"].sum()) if total else 0
    probability = _smoothed_probability(recovered, total)
    full_probability = _smoothed_probability(full, total)
    duration = pd.to_numeric(completed["recovery_duration_observations"], errors="coerce")
    depth = pd.to_numeric(completed["recovery_depth"], errors="coerce")
    persistence = pd.to_numeric(completed["post_recovery_persistence"], errors="coerce")
    quality_recovered = pd.to_numeric(completed["recovered_previous_quality"], errors="coerce")

    median_duration = float(duration.median()) if total else np.nan
    adaptation_speed = float(100.0 / (1.0 + median_duration)) if pd.notna(median_duration) else 0.0
    recovery_quality = float(
        100.0
        * (
            0.45 * full_probability
            + 0.30 * (float(quality_recovered.mean()) if total else 0.0)
            + 0.25 * min((float(persistence.mean()) if total else 0.0) / 5.0, 1.0)
        )
    )
    low_damage = float(100.0 / (1.0 + max(float(depth.median()) if total else 0.0, 0.0) / 10.0))
    resilience_score = float(
        np.clip(
            0.35 * probability * 100.0
            + 0.30 * adaptation_speed
            + 0.25 * recovery_quality
            + 0.10 * low_damage,
            0.0,
            100.0,
        )
    )
    return {
        "completed_recovery_episodes": total,
        "recovery_probability": probability,
        "full_recovery_probability": full_probability,
        "median_recovery_duration": median_duration,
        "mean_recovery_duration": float(duration.mean()) if total else np.nan,
        "median_recovery_depth": float(depth.median()) if total else np.nan,
        "mean_post_recovery_persistence": float(persistence.mean()) if total else np.nan,
        "adaptation_speed": adaptation_speed,
        "recovery_quality_score": recovery_quality,
        "low_damage_score": low_damage,
        "resilience_score": resilience_score,
        "evidence_strength": _evidence_strength(total),
    }


def build_resilience_history(
    lifecycle_history: pd.DataFrame,
    recovery_episodes: pd.DataFrame,
    *,
    minimum_own_episodes: int = 3,
    minimum_family_episodes: int = 5,
) -> pd.DataFrame:
    """Build causal resilience estimates for every champion and date."""
    if lifecycle_history.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for champion, sample in lifecycle_history.groupby("champion", sort=True):
        family = _family(champion)
        sample = sample.sort_values("origin_date")
        for _, row in sample.iterrows():
            current_date = pd.to_datetime(row["origin_date"])
            completed_before = recovery_episodes.loc[
                recovery_episodes["episode_completed"]
                & (pd.to_datetime(recovery_episodes["recovery_date"]) < current_date)
            ].copy()
            own = completed_before.loc[completed_before["champion"] == champion]
            family_reference = completed_before.loc[completed_before["family"] == family]
            if len(own) >= minimum_own_episodes:
                reference = own
                scope = "CHAMPION"
            elif len(family_reference) >= minimum_family_episodes:
                reference = family_reference
                scope = "FAMILY"
            else:
                reference = completed_before
                scope = "POOLED"

            metrics = _summarize_reference(reference)
            rows.append(
                {
                    "origin_date": current_date,
                    "champion": champion,
                    "family": family,
                    "lifecycle_state": row["lifecycle_state"],
                    "performance_level": row.get("performance_level"),
                    "performance_direction": row.get("performance_direction"),
                    "lifecycle_health_score": row.get("lifecycle_health_score"),
                    "advantage_vs_universe": row.get("advantage_vs_universe"),
                    "resilience_reference_scope": scope,
                    **metrics,
                }
            )
    return pd.DataFrame(rows).sort_values(["origin_date", "champion"]).reset_index(drop=True)


def build_current_resilience(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    latest = history.sort_values("origin_date").groupby("champion", as_index=False).tail(1).copy()
    latest["resilience_outlook"] = pd.cut(
        latest["resilience_score"],
        bins=[-np.inf, 35, 55, 75, np.inf],
        labels=["FRAGILE", "LIMITED", "RESILIENT", "HIGHLY_RESILIENT"],
    ).astype(str)
    return latest.sort_values("resilience_score", ascending=False).reset_index(drop=True)


def build_group_summary(episodes: pd.DataFrame, group: str) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for key, sample in episodes.groupby(group, sort=True):
        rows.append({group: key, **_summarize_reference(sample)})
    return pd.DataFrame(rows).sort_values("resilience_score", ascending=False).reset_index(drop=True)


def run_champion_resilience_laboratory(
    selections: pd.DataFrame,
    daily_states: pd.DataFrame | None = None,
    *,
    short_window: int = 3,
    long_window: int = 6,
    minimum_history: int = 4,
    minimum_state_persistence: int = 2,
    minimum_own_episodes: int = 3,
    minimum_family_episodes: int = 5,
) -> ChampionResilienceResult:
    lifecycle = run_champion_lifecycle_laboratory(
        selections,
        daily_states,
        short_window=short_window,
        long_window=long_window,
        minimum_history=minimum_history,
        minimum_state_persistence=minimum_state_persistence,
    )
    episodes = build_recovery_episodes(lifecycle.lifecycle_history)
    resilience_history = build_resilience_history(
        lifecycle.lifecycle_history,
        episodes,
        minimum_own_episodes=minimum_own_episodes,
        minimum_family_episodes=minimum_family_episodes,
    )
    current = build_current_resilience(resilience_history)
    champion_summary = build_group_summary(episodes, "champion")
    family_summary = build_group_summary(episodes, "family")
    evidence = current[[
        "champion",
        "family",
        "resilience_reference_scope",
        "completed_recovery_episodes",
        "evidence_strength",
        "recovery_probability",
        "full_recovery_probability",
    ]].copy() if not current.empty else pd.DataFrame()

    summary = pd.DataFrame()
    if not current.empty:
        summary = pd.DataFrame([{
            "champions": int(current["champion"].nunique()),
            "recovery_episodes": int(len(episodes)),
            "completed_recovery_episodes": int(episodes["episode_completed"].sum()) if not episodes.empty else 0,
            "mean_resilience_score": float(current["resilience_score"].mean()),
            "resilient_champions": int(current["resilience_outlook"].isin(["RESILIENT", "HIGHLY_RESILIENT"]).sum()),
            "top_champion": current.iloc[0]["champion"],
            "top_resilience_score": float(current.iloc[0]["resilience_score"]),
            "top_evidence_strength": current.iloc[0]["evidence_strength"],
        }])

    return ChampionResilienceResult(
        recovery_episodes=episodes,
        resilience_history=resilience_history,
        current_resilience=current,
        champion_summary=champion_summary,
        family_summary=family_summary,
        evidence=evidence,
        summary=summary,
        lifecycle_result=lifecycle,
    )

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from market_engine.evaluation.champion_resilience import (
    ChampionResilienceResult,
    build_recovery_episodes,
    run_champion_resilience_laboratory,
)


@dataclass(frozen=True)
class ChampionStressResult:
    stress_episodes: pd.DataFrame
    current_stress: pd.DataFrame
    champion_summary: pd.DataFrame
    family_summary: pd.DataFrame
    trigger_analysis: pd.DataFrame
    relapse_analysis: pd.DataFrame
    summary: pd.DataFrame
    resilience_result: ChampionResilienceResult


def _slope(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) < 2:
        return 0.0
    x = np.arange(len(numeric), dtype=float)
    return float(np.polyfit(x, numeric.to_numpy(dtype=float), 1)[0])


def _stress_outlook(score: float) -> str:
    if score >= 75:
        return "SEVERE"
    if score >= 55:
        return "HIGH"
    if score >= 35:
        return "MODERATE"
    return "LOW"


def build_stress_episodes(
    lifecycle_history: pd.DataFrame,
    recovery_episodes: pd.DataFrame,
    daily_states: pd.DataFrame | None = None,
    *,
    trigger_lookback: int = 2,
    relapse_horizon: int = 3,
) -> pd.DataFrame:
    if lifecycle_history.empty or recovery_episodes.empty:
        return pd.DataFrame()
    if trigger_lookback < 1 or relapse_horizon < 1:
        raise ValueError("trigger_lookback y relapse_horizon deben ser positivos")

    history = lifecycle_history.copy()
    history["origin_date"] = pd.to_datetime(history["origin_date"])
    regimes = pd.DataFrame()
    if daily_states is not None and not daily_states.empty:
        regimes = daily_states.copy()
        regimes["origin_date"] = pd.to_datetime(regimes["origin_date"])

    rows: list[dict[str, object]] = []
    for _, episode in recovery_episodes.iterrows():
        champion = episode["champion"]
        sample = history.loc[history["champion"] == champion].sort_values("origin_date").reset_index(drop=True)
        start_date = pd.to_datetime(episode["adverse_start_date"])
        end_date = pd.to_datetime(episode["adverse_end_date"])
        recovery_date = pd.to_datetime(episode["recovery_date"]) if pd.notna(episode["recovery_date"]) else pd.NaT
        adverse = sample.loc[(sample["origin_date"] >= start_date) & (sample["origin_date"] <= end_date)].copy()
        if adverse.empty:
            continue

        pre = sample.loc[sample["origin_date"] < start_date].tail(1)
        pre_advantage = float(pre["advantage_vs_universe"].iloc[0]) if not pre.empty else float(adverse["advantage_vs_universe"].iloc[0])
        advantage = pd.to_numeric(adverse["advantage_vs_universe"], errors="coerce")
        quality = pd.to_numeric(adverse["quality_score"], errors="coerce")
        worst_index = advantage.idxmin()
        bottom_date = adverse.loc[worst_index, "origin_date"]
        bottom_advantage = float(advantage.loc[worst_index])
        maximum_damage = max(pre_advantage - bottom_advantage, 0.0)
        deterioration_velocity = max(-_slope(advantage), 0.0)
        time_to_bottom = int((adverse.index.get_loc(worst_index)) + 1)
        stress_volatility = float(advantage.std(ddof=0))
        cumulative_damage = float(np.maximum(pre_advantage - advantage, 0.0).sum())

        recovered = bool(episode["recovered"])
        recovery_efficiency = 0.0
        trigger_advantage_delta = np.nan
        trigger_slope_delta = np.nan
        trigger_health_delta = np.nan
        regime_before = None
        regime_at_recovery = None
        relapse = False

        if recovered and pd.notna(recovery_date):
            recovery_row = sample.loc[sample["origin_date"] == recovery_date]
            if not recovery_row.empty:
                recovery_advantage = float(recovery_row["advantage_vs_universe"].iloc[0])
                duration = max(int(episode["recovery_duration_observations"]), 1)
                recovered_damage = max(recovery_advantage - bottom_advantage, 0.0)
                recovery_efficiency = recovered_damage / duration

                position = int(recovery_row.index[0])
                trigger_block = sample.iloc[max(0, position - trigger_lookback): position + 1]
                if len(trigger_block) >= 2:
                    trigger_advantage_delta = float(trigger_block["advantage_vs_universe"].iloc[-1] - trigger_block["advantage_vs_universe"].iloc[0])
                    trigger_slope_delta = float(trigger_block["advantage_slope"].iloc[-1] - trigger_block["advantage_slope"].iloc[0])
                    trigger_health_delta = float(trigger_block["lifecycle_health_score"].iloc[-1] - trigger_block["lifecycle_health_score"].iloc[0])

                future = sample.iloc[position + 1: position + 1 + relapse_horizon]
                relapse = bool(future["lifecycle_state"].isin(["DETERIORATING", "OBSOLETE"]).any())

                if not regimes.empty and "regime" in regimes.columns:
                    before_match = regimes.loc[regimes["origin_date"] < recovery_date].tail(1)
                    at_match = regimes.loc[regimes["origin_date"] == recovery_date]
                    regime_before = before_match["regime"].iloc[0] if not before_match.empty else None
                    regime_at_recovery = at_match["regime"].iloc[0] if not at_match.empty else None

        stress_score = float(np.clip(
            0.30 * min(deterioration_velocity / 10.0, 1.0) * 100
            + 0.30 * min(maximum_damage / 50.0, 1.0) * 100
            + 0.20 * min(cumulative_damage / 100.0, 1.0) * 100
            + 0.10 * min(stress_volatility / 25.0, 1.0) * 100
            + 0.10 * (100.0 if relapse else 0.0),
            0.0,
            100.0,
        ))

        rows.append({
            **episode.to_dict(),
            "bottom_date": bottom_date,
            "time_to_bottom": time_to_bottom,
            "deterioration_velocity": deterioration_velocity,
            "maximum_damage": maximum_damage,
            "cumulative_damage": cumulative_damage,
            "stress_volatility": stress_volatility,
            "recovery_efficiency": recovery_efficiency,
            "trigger_advantage_delta": trigger_advantage_delta,
            "trigger_slope_delta": trigger_slope_delta,
            "trigger_health_delta": trigger_health_delta,
            "regime_before_recovery": regime_before,
            "regime_at_recovery": regime_at_recovery,
            "relapse_within_horizon": relapse,
            "relapse_horizon": relapse_horizon,
            "stress_score": stress_score,
            "stress_outlook": _stress_outlook(stress_score),
        })

    return pd.DataFrame(rows).sort_values(["adverse_start_date", "champion"]).reset_index(drop=True)


def _summarize(group: pd.DataFrame) -> dict[str, object]:
    completed = group.loc[group["episode_completed"]].copy()
    total = int(len(group))
    completed_count = int(len(completed))
    relapse_rate = float(completed["relapse_within_horizon"].mean()) if completed_count else np.nan
    return {
        "stress_episodes": total,
        "completed_stress_episodes": completed_count,
        "mean_stress_score": float(group["stress_score"].mean()) if total else np.nan,
        "median_deterioration_velocity": float(group["deterioration_velocity"].median()) if total else np.nan,
        "median_maximum_damage": float(group["maximum_damage"].median()) if total else np.nan,
        "median_time_to_bottom": float(group["time_to_bottom"].median()) if total else np.nan,
        "mean_recovery_efficiency": float(completed["recovery_efficiency"].mean()) if completed_count else np.nan,
        "relapse_rate": relapse_rate,
        "recovery_rate": float(group["recovered"].mean()) if total else np.nan,
    }


def build_group_summary(episodes: pd.DataFrame, group_column: str) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame()
    rows = [{group_column: key, **_summarize(sample)} for key, sample in episodes.groupby(group_column, sort=True)]
    return pd.DataFrame(rows).sort_values("mean_stress_score").reset_index(drop=True)


def build_current_stress(lifecycle_history: pd.DataFrame, champion_summary: pd.DataFrame) -> pd.DataFrame:
    if lifecycle_history.empty:
        return pd.DataFrame()
    latest = lifecycle_history.sort_values("origin_date").groupby("champion", as_index=False).tail(1).copy()
    current = latest.merge(champion_summary, on="champion", how="left")
    current["current_adverse_state"] = current["lifecycle_state"].isin(["DETERIORATING", "OBSOLETE"])
    current["stress_risk_score"] = (
        pd.to_numeric(current["mean_stress_score"], errors="coerce").fillna(50.0) * 0.45
        + (100.0 - pd.to_numeric(current["lifecycle_health_score"], errors="coerce").fillna(50.0)) * 0.35
        + pd.to_numeric(current["relapse_rate"], errors="coerce").fillna(0.5) * 20.0
    ).clip(0, 100)
    current["stress_risk_outlook"] = current["stress_risk_score"].map(_stress_outlook)
    return current.sort_values("stress_risk_score").reset_index(drop=True)


def run_champion_stress_laboratory(
    selections: pd.DataFrame,
    daily_states: pd.DataFrame | None = None,
    *,
    short_window: int = 3,
    long_window: int = 6,
    minimum_history: int = 4,
    minimum_state_persistence: int = 2,
    minimum_own_episodes: int = 3,
    minimum_family_episodes: int = 5,
    trigger_lookback: int = 2,
    relapse_horizon: int = 3,
) -> ChampionStressResult:
    resilience = run_champion_resilience_laboratory(
        selections,
        daily_states,
        short_window=short_window,
        long_window=long_window,
        minimum_history=minimum_history,
        minimum_state_persistence=minimum_state_persistence,
        minimum_own_episodes=minimum_own_episodes,
        minimum_family_episodes=minimum_family_episodes,
    )
    episodes = build_stress_episodes(
        resilience.lifecycle_result.lifecycle_history,
        resilience.recovery_episodes,
        daily_states,
        trigger_lookback=trigger_lookback,
        relapse_horizon=relapse_horizon,
    )
    champion_summary = build_group_summary(episodes, "champion")
    family_summary = build_group_summary(episodes, "family")
    current = build_current_stress(resilience.lifecycle_result.lifecycle_history, champion_summary)
    trigger_columns = [
        "champion", "family", "recovery_date", "recovery_state",
        "trigger_advantage_delta", "trigger_slope_delta", "trigger_health_delta",
        "regime_before_recovery", "regime_at_recovery", "recovery_efficiency",
    ]
    trigger_analysis = episodes.loc[episodes["recovered"], trigger_columns].copy() if not episodes.empty else pd.DataFrame()
    relapse_columns = [
        "champion", "family", "recovery_date", "post_recovery_persistence",
        "relapse_within_horizon", "relapse_horizon", "stress_score",
    ]
    relapse_analysis = episodes.loc[episodes["recovered"], relapse_columns].copy() if not episodes.empty else pd.DataFrame()

    summary = pd.DataFrame()
    if not current.empty:
        summary = pd.DataFrame([{
            "champions": int(current["champion"].nunique()),
            "stress_episodes": int(len(episodes)),
            "completed_stress_episodes": int(episodes["episode_completed"].sum()) if not episodes.empty else 0,
            "mean_stress_score": float(episodes["stress_score"].mean()) if not episodes.empty else np.nan,
            "mean_relapse_rate": float(champion_summary["relapse_rate"].mean()) if not champion_summary.empty else np.nan,
            "lowest_stress_champion": current.iloc[0]["champion"],
            "lowest_stress_risk_score": float(current.iloc[0]["stress_risk_score"]),
            "highest_stress_champion": current.iloc[-1]["champion"],
            "highest_stress_risk_score": float(current.iloc[-1]["stress_risk_score"]),
        }])

    return ChampionStressResult(
        stress_episodes=episodes,
        current_stress=current,
        champion_summary=champion_summary,
        family_summary=family_summary,
        trigger_analysis=trigger_analysis,
        relapse_analysis=relapse_analysis,
        summary=summary,
        resilience_result=resilience,
    )

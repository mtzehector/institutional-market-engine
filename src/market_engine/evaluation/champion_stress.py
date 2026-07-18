from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from market_engine.evaluation.champion_resilience import (
    ChampionResilienceResult,
    run_champion_resilience_laboratory,
)

ADVERSE_STATES = {"DETERIORATING", "OBSOLETE"}


@dataclass(frozen=True)
class ChampionStressResult:
    stress_episodes: pd.DataFrame
    current_stress: pd.DataFrame
    champion_summary: pd.DataFrame
    family_summary: pd.DataFrame
    trigger_analysis: pd.DataFrame
    relapse_analysis: pd.DataFrame
    active_episodes: pd.DataFrame
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


def _first_weakening_position(sample: pd.DataFrame, confirmed_start: int) -> int:
    position = confirmed_start
    while position > 0:
        previous = sample.iloc[position - 1]
        raw_adverse = str(previous.get("raw_lifecycle_state")) in ADVERSE_STATES
        declining = str(previous.get("performance_direction")) == "DECLINING"
        negative_slope = float(previous.get("advantage_slope", 0.0)) < -1.0
        if not (raw_adverse or declining or negative_slope):
            break
        position -= 1
    return position


def _trigger_coherence(advantage_delta: float, slope_delta: float, health_delta: float) -> tuple[int, int, float]:
    values = [advantage_delta, slope_delta, health_delta]
    observable = [value for value in values if pd.notna(value)]
    positive = sum(float(value) > 0 for value in observable)
    total = len(observable)
    score = 100.0 * positive / total if total else np.nan
    return positive, total, score


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

        confirmed_matches = sample.index[sample["origin_date"] == start_date].tolist()
        if not confirmed_matches:
            continue
        confirmed_start = int(confirmed_matches[0])
        weakening_start = _first_weakening_position(sample, confirmed_start)
        end_matches = sample.index[sample["origin_date"] == end_date].tolist()
        end_position = int(end_matches[-1]) if end_matches else confirmed_start
        adverse = sample.iloc[weakening_start : end_position + 1].copy()
        if adverse.empty:
            continue

        pre = sample.iloc[max(0, weakening_start - 1) : weakening_start]
        pre_advantage = float(pre["advantage_vs_universe"].iloc[0]) if not pre.empty else float(adverse["advantage_vs_universe"].iloc[0])
        advantage = pd.to_numeric(adverse["advantage_vs_universe"], errors="coerce")
        worst_label = advantage.idxmin()
        bottom_date = adverse.loc[worst_label, "origin_date"]
        bottom_advantage = float(advantage.loc[worst_label])
        maximum_damage = max(pre_advantage - bottom_advantage, 0.0)
        time_to_bottom = int(adverse.index.get_loc(worst_label) + 1)
        damage_velocity = maximum_damage / max(time_to_bottom, 1)
        slope_velocity = max(-_slope(advantage.iloc[:time_to_bottom]), 0.0)
        deterioration_velocity = float(0.65 * damage_velocity + 0.35 * slope_velocity)
        stress_volatility = float(advantage.std(ddof=0))
        cumulative_damage = float(np.maximum(pre_advantage - advantage, 0.0).sum())

        recovered = bool(episode["recovered"])
        recovery_efficiency = 0.0
        trigger_advantage_delta = np.nan
        trigger_slope_delta = np.nan
        trigger_health_delta = np.nan
        regime_before = None
        regime_at_recovery = None
        relapse: bool | float = np.nan
        relapse_observable = False
        positive_signals = 0
        total_signals = 0
        trigger_coherence_score = np.nan

        if recovered and pd.notna(recovery_date):
            recovery_row = sample.loc[sample["origin_date"] == recovery_date]
            if not recovery_row.empty:
                recovery_advantage = float(recovery_row["advantage_vs_universe"].iloc[0])
                duration = max(int(episode["recovery_duration_observations"]), 1)
                recovery_efficiency = max(recovery_advantage - bottom_advantage, 0.0) / duration
                position = int(recovery_row.index[0])
                trigger_block = sample.iloc[max(0, position - trigger_lookback) : position + 1]
                if len(trigger_block) >= 2:
                    trigger_advantage_delta = float(trigger_block["advantage_vs_universe"].iloc[-1] - trigger_block["advantage_vs_universe"].iloc[0])
                    trigger_slope_delta = float(trigger_block["advantage_slope"].iloc[-1] - trigger_block["advantage_slope"].iloc[0])
                    trigger_health_delta = float(trigger_block["lifecycle_health_score"].iloc[-1] - trigger_block["lifecycle_health_score"].iloc[0])
                positive_signals, total_signals, trigger_coherence_score = _trigger_coherence(
                    trigger_advantage_delta, trigger_slope_delta, trigger_health_delta
                )
                future = sample.iloc[position + 1 : position + 1 + relapse_horizon]
                relapse_observable = len(future) >= relapse_horizon
                if relapse_observable:
                    relapse = bool(future["lifecycle_state"].isin(ADVERSE_STATES).any())
                if not regimes.empty and "regime" in regimes.columns:
                    before_match = regimes.loc[regimes["origin_date"] < recovery_date].tail(1)
                    at_match = regimes.loc[regimes["origin_date"] == recovery_date]
                    regime_before = before_match["regime"].iloc[0] if not before_match.empty else None
                    regime_at_recovery = at_match["regime"].iloc[0] if not at_match.empty else None

        relapse_penalty = 100.0 if relapse is True else 0.0
        stress_score = float(np.clip(
            0.30 * min(deterioration_velocity / 10.0, 1.0) * 100
            + 0.30 * min(maximum_damage / 50.0, 1.0) * 100
            + 0.20 * min(cumulative_damage / 100.0, 1.0) * 100
            + 0.10 * min(stress_volatility / 25.0, 1.0) * 100
            + 0.10 * relapse_penalty,
            0.0,
            100.0,
        ))
        rows.append({
            **episode.to_dict(),
            "weakening_start_date": adverse.iloc[0]["origin_date"],
            "bottom_date": bottom_date,
            "time_to_bottom": time_to_bottom,
            "deterioration_velocity": deterioration_velocity,
            "damage_velocity": damage_velocity,
            "slope_velocity": slope_velocity,
            "maximum_damage": maximum_damage,
            "cumulative_damage": cumulative_damage,
            "stress_volatility": stress_volatility,
            "recovery_efficiency": recovery_efficiency,
            "trigger_advantage_delta": trigger_advantage_delta,
            "trigger_slope_delta": trigger_slope_delta,
            "trigger_health_delta": trigger_health_delta,
            "trigger_signals_positive": positive_signals,
            "trigger_signals_total": total_signals,
            "trigger_coherence_score": trigger_coherence_score,
            "regime_before_recovery": regime_before,
            "regime_at_recovery": regime_at_recovery,
            "relapse_observable": relapse_observable,
            "relapse_within_horizon": relapse,
            "relapse_horizon": relapse_horizon,
            "stress_score": stress_score,
            "stress_outlook": _stress_outlook(stress_score),
        })
    return pd.DataFrame(rows).sort_values(["adverse_start_date", "champion"]).reset_index(drop=True)


def _summarize(group: pd.DataFrame) -> dict[str, object]:
    completed = group.loc[group["episode_completed"]].copy()
    observable = completed.loc[completed["relapse_observable"]].copy()
    total = int(len(group))
    completed_count = int(len(completed))
    relapse_rate = float(observable["relapse_within_horizon"].astype(float).mean()) if len(observable) else np.nan
    return {
        "stress_episodes": total,
        "completed_stress_episodes": completed_count,
        "observable_relapse_episodes": int(len(observable)),
        "historical_stress_susceptibility": float(group["stress_score"].mean()) if total else np.nan,
        "mean_stress_score": float(group["stress_score"].mean()) if total else np.nan,
        "median_deterioration_velocity": float(group["deterioration_velocity"].median()) if total else np.nan,
        "median_maximum_damage": float(group["maximum_damage"].median()) if total else np.nan,
        "median_time_to_bottom": float(group["time_to_bottom"].median()) if total else np.nan,
        "mean_recovery_efficiency": float(completed["recovery_efficiency"].mean()) if completed_count else np.nan,
        "mean_trigger_coherence": float(completed["trigger_coherence_score"].mean()) if completed_count else np.nan,
        "relapse_rate": relapse_rate,
        "recovery_rate": float(group["recovered"].mean()) if total else np.nan,
    }


def build_group_summary(episodes: pd.DataFrame, group_column: str) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame()
    rows = [{group_column: key, **_summarize(sample)} for key, sample in episodes.groupby(group_column, sort=True)]
    return pd.DataFrame(rows).sort_values("historical_stress_susceptibility").reset_index(drop=True)


def _active_stress_components(sample: pd.DataFrame) -> dict[str, float | bool | str]:
    latest = sample.iloc[-1]
    active = str(latest["lifecycle_state"]) in ADVERSE_STATES or str(latest.get("raw_lifecycle_state")) in ADVERSE_STATES
    recent = sample.tail(6)
    peak_advantage = float(pd.to_numeric(recent["advantage_vs_universe"], errors="coerce").max())
    current_advantage = float(latest["advantage_vs_universe"])
    drawdown = max(peak_advantage - current_advantage, 0.0)
    slope = float(latest.get("advantage_slope", 0.0))
    health = float(latest.get("lifecycle_health_score", 50.0))
    direction = str(latest.get("performance_direction"))
    pending = str(latest.get("pending_lifecycle_state")) in ADVERSE_STATES
    state_penalty = 30.0 if str(latest["lifecycle_state"]) in ADVERSE_STATES else 0.0
    raw_penalty = 15.0 if str(latest.get("raw_lifecycle_state")) in ADVERSE_STATES else 0.0
    pending_penalty = 10.0 if pending else 0.0
    direction_penalty = 15.0 if direction == "DECLINING" else 0.0
    score = float(np.clip(
        state_penalty + raw_penalty + pending_penalty + direction_penalty
        + 20.0 * min(max(-slope, 0.0) / 8.0, 1.0)
        + 15.0 * min(drawdown / 40.0, 1.0)
        + 10.0 * min((100.0 - health) / 100.0, 1.0),
        0.0,
        100.0,
    ))
    return {
        "current_adverse_state": bool(active),
        "current_active_stress_score": score,
        "current_advantage_drawdown": drawdown,
        "current_negative_slope": max(-slope, 0.0),
        "current_health_deficit": 100.0 - health,
        "current_stress_outlook": _stress_outlook(score),
    }


def build_current_stress(lifecycle_history: pd.DataFrame, champion_summary: pd.DataFrame) -> pd.DataFrame:
    if lifecycle_history.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for champion, sample in lifecycle_history.groupby("champion", sort=True):
        sample = sample.sort_values("origin_date").reset_index(drop=True)
        latest = sample.iloc[-1].to_dict()
        rows.append({"champion": champion, **latest, **_active_stress_components(sample)})
    current = pd.DataFrame(rows)
    current = current.merge(champion_summary, on="champion", how="left")
    historical = pd.to_numeric(current["historical_stress_susceptibility"], errors="coerce").fillna(50.0)
    active = pd.to_numeric(current["current_active_stress_score"], errors="coerce").fillna(0.0)
    relapse = pd.to_numeric(current["relapse_rate"], errors="coerce").fillna(0.0) * 100.0
    current["stress_risk_score"] = (0.35 * historical + 0.55 * active + 0.10 * relapse).clip(0, 100)
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
        selections, daily_states,
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
        "trigger_signals_positive", "trigger_signals_total", "trigger_coherence_score",
        "regime_before_recovery", "regime_at_recovery", "recovery_efficiency",
    ]
    trigger_analysis = episodes.loc[episodes["recovered"], trigger_columns].copy() if not episodes.empty else pd.DataFrame()
    relapse_columns = [
        "champion", "family", "recovery_date", "post_recovery_persistence",
        "relapse_observable", "relapse_within_horizon", "relapse_horizon", "stress_score",
    ]
    relapse_analysis = episodes.loc[episodes["recovered"], relapse_columns].copy() if not episodes.empty else pd.DataFrame()
    active_episodes = current.loc[current["current_adverse_state"]].copy()

    summary = pd.DataFrame()
    if not current.empty:
        summary = pd.DataFrame([{
            "champions": int(current["champion"].nunique()),
            "stress_episodes": int(len(episodes)),
            "completed_stress_episodes": int(episodes["episode_completed"].sum()) if not episodes.empty else 0,
            "active_stress_champions": int(current["current_adverse_state"].sum()),
            "mean_historical_stress_susceptibility": float(current["historical_stress_susceptibility"].mean()),
            "mean_current_active_stress": float(current["current_active_stress_score"].mean()),
            "highest_active_stress_champion": current.sort_values("current_active_stress_score").iloc[-1]["champion"],
            "highest_active_stress_score": float(current["current_active_stress_score"].max()),
            "highest_total_risk_champion": current.iloc[-1]["champion"],
            "highest_total_risk_score": float(current.iloc[-1]["stress_risk_score"]),
        }])

    return ChampionStressResult(
        stress_episodes=episodes,
        current_stress=current,
        champion_summary=champion_summary,
        family_summary=family_summary,
        trigger_analysis=trigger_analysis,
        relapse_analysis=relapse_analysis,
        active_episodes=active_episodes,
        summary=summary,
        resilience_result=resilience,
    )

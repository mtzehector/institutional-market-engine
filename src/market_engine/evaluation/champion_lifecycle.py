from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from market_engine.evaluation.market_memory import build_date_champion_quality

LIFECYCLE_STATES = (
    "DISCOVERY",
    "EMERGING",
    "MATURE",
    "DETERIORATING",
    "OBSOLETE",
    "RECOVERING",
)

PERFORMANCE_LEVELS = ("STRONG", "MEDIUM", "WEAK")
PERFORMANCE_DIRECTIONS = ("IMPROVING", "STABLE", "DECLINING")


@dataclass(frozen=True)
class ChampionLifecycleResult:
    lifecycle_history: pd.DataFrame
    current_status: pd.DataFrame
    transitions: pd.DataFrame
    regime_performance: pd.DataFrame
    summary: pd.DataFrame


def _slope(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) < 2:
        return 0.0
    x = np.arange(len(numeric), dtype=float)
    return float(np.polyfit(x, numeric.to_numpy(dtype=float), 1)[0])


def _safe_ratio(numerator: float, denominator: float, default: float = 1.0) -> float:
    if not np.isfinite(denominator) or abs(denominator) <= 1e-12:
        return default
    return float(numerator / denominator)


def _performance_level(
    *,
    short_advantage: float,
    positive_rate: float,
    relative_quality_ratio: float,
) -> str:
    if (
        short_advantage >= 10
        and positive_rate >= 0.67
        and relative_quality_ratio >= 0.95
    ) or relative_quality_ratio >= 1.10:
        return "STRONG"
    if short_advantage < 0 or positive_rate < 0.40 or relative_quality_ratio < 0.82:
        return "WEAK"
    return "MEDIUM"


def _performance_direction(
    *,
    short_advantage: float,
    long_advantage: float,
    advantage_slope: float,
) -> str:
    acceleration = short_advantage - long_advantage
    if advantage_slope > 1.0 or acceleration > 3.0:
        return "IMPROVING"
    if advantage_slope < -1.0 or acceleration < -3.0:
        return "DECLINING"
    return "STABLE"


def _candidate_state(
    *,
    observations: int,
    minimum_history: int,
    level: str,
    direction: str,
    long_advantage: float,
    consecutive_underperformance: int,
    previous_state: str | None,
) -> str:
    if observations < minimum_history:
        return "DISCOVERY"
    if level == "WEAK" and direction == "DECLINING" and consecutive_underperformance >= 3:
        return "OBSOLETE"
    if level == "STRONG" and direction in {"IMPROVING", "STABLE"}:
        return "MATURE"
    if direction == "IMPROVING":
        if previous_state in {"DETERIORATING", "OBSOLETE"}:
            return "RECOVERING"
        return "EMERGING"
    if direction == "DECLINING":
        return "DETERIORATING"
    if level == "WEAK" or long_advantage < 0:
        return "DETERIORATING"
    return "EMERGING"


def _apply_hysteresis(
    *,
    confirmed_state: str | None,
    candidate_state: str,
    health_score: float,
    advantage_slope: float,
    short_advantage: float,
) -> str:
    if confirmed_state is None or confirmed_state == "DISCOVERY":
        return candidate_state

    if confirmed_state in {"DETERIORATING", "OBSOLETE"} and candidate_state in {
        "RECOVERING",
        "EMERGING",
        "MATURE",
    }:
        if advantage_slope <= 1.0 or health_score < 50 or short_advantage <= 0:
            return confirmed_state

    if confirmed_state in {"MATURE", "EMERGING", "RECOVERING"} and candidate_state in {
        "DETERIORATING",
        "OBSOLETE",
    }:
        if advantage_slope > -2.0 and short_advantage >= 0 and health_score >= 40:
            return confirmed_state

    return candidate_state


def build_champion_lifecycle_history(
    selections: pd.DataFrame,
    *,
    short_window: int = 3,
    long_window: int = 6,
    minimum_history: int = 4,
    minimum_state_persistence: int = 2,
) -> pd.DataFrame:
    if short_window < 2:
        raise ValueError("short_window debe ser al menos 2")
    if long_window < short_window:
        raise ValueError("long_window debe ser mayor o igual que short_window")
    if minimum_history < 2:
        raise ValueError("minimum_history debe ser al menos 2")
    if minimum_state_persistence < 1:
        raise ValueError("minimum_state_persistence debe ser positivo")

    quality = build_date_champion_quality(selections)
    if quality.empty:
        return pd.DataFrame()

    quality["origin_date"] = pd.to_datetime(quality["origin_date"])
    universe = quality.loc[
        quality["champion"] == "UNIVERSE", ["origin_date", "quality_score"]
    ].rename(columns={"quality_score": "universe_quality"})
    champions = quality.loc[quality["champion"] != "UNIVERSE"].copy()
    champions = champions.merge(universe, on="origin_date", how="left")
    champions["advantage_vs_universe"] = (
        champions["quality_score"] - champions["universe_quality"]
    )

    rows: list[dict[str, object]] = []
    for champion, sample in champions.groupby("champion", sort=True):
        sample = sample.sort_values("origin_date").reset_index(drop=True)
        confirmed_state: str | None = None
        pending_state: str | None = None
        pending_count = 0
        underperformance_streak = 0
        running_peak = -np.inf

        for index, row in sample.iterrows():
            history = sample.iloc[: index + 1].copy()
            advantage = pd.to_numeric(history["advantage_vs_universe"], errors="coerce")
            quality_values = pd.to_numeric(history["quality_score"], errors="coerce")

            current_advantage = float(advantage.iloc[-1]) if pd.notna(advantage.iloc[-1]) else 0.0
            underperformance_streak = underperformance_streak + 1 if current_advantage < 0 else 0

            current_quality = float(quality_values.iloc[-1])
            running_peak = max(running_peak, current_quality)
            drawdown = max(running_peak - current_quality, 0.0)

            short_advantage = float(advantage.tail(short_window).mean())
            long_advantage = float(advantage.tail(long_window).mean())
            advantage_slope = _slope(advantage.tail(long_window))
            quality_volatility = float(quality_values.tail(long_window).std(ddof=0))
            positive_rate = float((advantage.tail(long_window) > 0).mean())

            historical_median = float(quality_values.median())
            recent_median = float(quality_values.tail(long_window).median())
            relative_quality_ratio = _safe_ratio(current_quality, historical_median)
            recent_quality_ratio = _safe_ratio(current_quality, recent_median)
            advantage_vs_own_mean = current_advantage - float(advantage.mean())

            level = _performance_level(
                short_advantage=short_advantage,
                positive_rate=positive_rate,
                relative_quality_ratio=relative_quality_ratio,
            )
            direction = _performance_direction(
                short_advantage=short_advantage,
                long_advantage=long_advantage,
                advantage_slope=advantage_slope,
            )

            health_score = float(
                np.clip(
                    45
                    + 1.0 * short_advantage
                    + 1.4 * advantage_slope
                    + 18 * (positive_rate - 0.5)
                    + 22 * (relative_quality_ratio - 1.0)
                    + 12 * (recent_quality_ratio - 1.0)
                    - 0.20 * drawdown
                    - 0.35 * quality_volatility,
                    0,
                    100,
                )
            )

            raw_state = _candidate_state(
                observations=index + 1,
                minimum_history=minimum_history,
                level=level,
                direction=direction,
                long_advantage=long_advantage,
                consecutive_underperformance=underperformance_streak,
                previous_state=confirmed_state,
            )
            hysteresis_state = _apply_hysteresis(
                confirmed_state=confirmed_state,
                candidate_state=raw_state,
                health_score=health_score,
                advantage_slope=advantage_slope,
                short_advantage=short_advantage,
            )

            previous_confirmed = confirmed_state
            if confirmed_state is None:
                confirmed_state = hysteresis_state
                pending_state = None
                pending_count = 0
            elif hysteresis_state == confirmed_state:
                pending_state = None
                pending_count = 0
            else:
                if pending_state == hysteresis_state:
                    pending_count += 1
                else:
                    pending_state = hysteresis_state
                    pending_count = 1
                if pending_count >= minimum_state_persistence:
                    confirmed_state = hysteresis_state
                    pending_state = None
                    pending_count = 0

            state = confirmed_state or hysteresis_state
            rows.append(
                {
                    "origin_date": row["origin_date"],
                    "champion": champion,
                    "raw_lifecycle_state": raw_state,
                    "lifecycle_state": state,
                    "previous_lifecycle_state": previous_confirmed,
                    "pending_lifecycle_state": pending_state,
                    "pending_state_count": pending_count,
                    "is_transition": bool(
                        previous_confirmed is not None and state != previous_confirmed
                    ),
                    "observations_seen": index + 1,
                    "performance_level": level,
                    "performance_direction": direction,
                    "quality_score": current_quality,
                    "universe_quality": row["universe_quality"],
                    "advantage_vs_universe": current_advantage,
                    "short_advantage": short_advantage,
                    "long_advantage": long_advantage,
                    "advantage_slope": advantage_slope,
                    "advantage_vs_own_mean": advantage_vs_own_mean,
                    "quality_volatility": quality_volatility,
                    "positive_advantage_rate": positive_rate,
                    "quality_drawdown_from_peak": drawdown,
                    "historical_quality_median": historical_median,
                    "recent_quality_median": recent_median,
                    "relative_quality_ratio": relative_quality_ratio,
                    "recent_quality_ratio": recent_quality_ratio,
                    "consecutive_underperformance": underperformance_streak,
                    "lifecycle_health_score": health_score,
                }
            )

    return pd.DataFrame(rows).sort_values(["origin_date", "champion"]).reset_index(drop=True)


def build_current_lifecycle_status(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    latest = history.sort_values("origin_date").groupby("champion", as_index=False).tail(1).copy()
    state_priority = {
        "MATURE": 5,
        "RECOVERING": 4,
        "EMERGING": 3,
        "DISCOVERY": 2,
        "DETERIORATING": 1,
        "OBSOLETE": 0,
    }
    latest["state_priority"] = latest["lifecycle_state"].map(state_priority).fillna(0)
    latest["deployment_score"] = (
        0.60 * latest["lifecycle_health_score"]
        + 0.20 * latest["positive_advantage_rate"] * 100
        + 0.10 * latest["relative_quality_ratio"].clip(0, 1.5) / 1.5 * 100
        + 0.10 * latest["state_priority"] / 5 * 100
    ).clip(0, 100)
    latest["recommended_action"] = latest["lifecycle_state"].map(
        {
            "MATURE": "ACTIVE",
            "RECOVERING": "INCREASE_GRADUALLY",
            "EMERGING": "MONITOR_AND_TEST",
            "DISCOVERY": "OBSERVE",
            "DETERIORATING": "REDUCE_AUTHORITY",
            "OBSOLETE": "SUSPEND",
        }
    )
    return latest.sort_values("deployment_score", ascending=False).reset_index(drop=True)


def build_lifecycle_transitions(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    transitions = history.loc[history["is_transition"]].copy()
    if transitions.empty:
        return transitions
    transitions["transition"] = (
        transitions["previous_lifecycle_state"].astype(str)
        + " -> "
        + transitions["lifecycle_state"].astype(str)
    )
    return transitions.sort_values(["origin_date", "champion"]).reset_index(drop=True)


def build_regime_lifecycle_performance(
    history: pd.DataFrame,
    daily_states: pd.DataFrame | None,
) -> pd.DataFrame:
    if history.empty or daily_states is None or daily_states.empty or "regime" not in daily_states:
        return pd.DataFrame()
    regimes = daily_states[["origin_date", "regime"]].copy()
    regimes["origin_date"] = pd.to_datetime(regimes["origin_date"])
    merged = history.merge(regimes, on="origin_date", how="left")
    return (
        merged.groupby(["champion", "regime"], dropna=False)
        .agg(
            dates=("origin_date", "nunique"),
            mean_quality=("quality_score", "mean"),
            mean_advantage_vs_universe=("advantage_vs_universe", "mean"),
            positive_advantage_rate=("advantage_vs_universe", lambda s: float((s > 0).mean())),
            mean_health=("lifecycle_health_score", "mean"),
            mean_relative_quality=("relative_quality_ratio", "mean"),
        )
        .reset_index()
        .sort_values(["regime", "mean_health"], ascending=[True, False])
        .reset_index(drop=True)
    )


def run_champion_lifecycle_laboratory(
    selections: pd.DataFrame,
    daily_states: pd.DataFrame | None = None,
    *,
    short_window: int = 3,
    long_window: int = 6,
    minimum_history: int = 4,
    minimum_state_persistence: int = 2,
) -> ChampionLifecycleResult:
    history = build_champion_lifecycle_history(
        selections,
        short_window=short_window,
        long_window=long_window,
        minimum_history=minimum_history,
        minimum_state_persistence=minimum_state_persistence,
    )
    current = build_current_lifecycle_status(history)
    transitions = build_lifecycle_transitions(history)
    regime_performance = build_regime_lifecycle_performance(history, daily_states)

    summary = pd.DataFrame()
    if not current.empty:
        summary = pd.DataFrame(
            [
                {
                    "champions": int(current["champion"].nunique()),
                    "active_or_recovering": int(
                        current["lifecycle_state"].isin(["MATURE", "RECOVERING"]).sum()
                    ),
                    "deteriorating_or_obsolete": int(
                        current["lifecycle_state"].isin(["DETERIORATING", "OBSOLETE"]).sum()
                    ),
                    "mean_health": float(current["lifecycle_health_score"].mean()),
                    "top_champion": current.iloc[0]["champion"],
                    "top_lifecycle_state": current.iloc[0]["lifecycle_state"],
                    "top_deployment_score": float(current.iloc[0]["deployment_score"]),
                    "transitions_detected": int(len(transitions)),
                    "transition_rate": float(len(transitions) / max(len(history), 1)),
                }
            ]
        )

    return ChampionLifecycleResult(
        lifecycle_history=history,
        current_status=current,
        transitions=transitions,
        regime_performance=regime_performance,
        summary=summary,
    )

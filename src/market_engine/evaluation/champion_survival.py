from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from market_engine.evaluation.champion_lifecycle import (
    ChampionLifecycleResult,
    run_champion_lifecycle_laboratory,
)


@dataclass(frozen=True)
class ChampionSurvivalResult:
    survival_history: pd.DataFrame
    current_survival: pd.DataFrame
    state_duration_spells: pd.DataFrame
    survival_curves: pd.DataFrame
    summary: pd.DataFrame
    lifecycle_result: ChampionLifecycleResult


def _build_state_spells(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for champion, sample in history.groupby("champion", sort=True):
        sample = sample.sort_values("origin_date").reset_index(drop=True)
        spell_id = 0
        start = 0
        for index in range(1, len(sample) + 1):
            boundary = index == len(sample) or (
                sample.loc[index, "lifecycle_state"] != sample.loc[index - 1, "lifecycle_state"]
            )
            if not boundary:
                continue
            block = sample.iloc[start:index]
            rows.append(
                {
                    "champion": champion,
                    "spell_id": spell_id,
                    "lifecycle_state": block.iloc[0]["lifecycle_state"],
                    "start_date": block.iloc[0]["origin_date"],
                    "end_date": block.iloc[-1]["origin_date"],
                    "duration_observations": int(len(block)),
                    "ended_with_transition": bool(index < len(sample)),
                    "next_state": sample.loc[index, "lifecycle_state"] if index < len(sample) else None,
                }
            )
            spell_id += 1
            start = index
    return pd.DataFrame(rows)


def _empirical_survival_probability(
    durations: pd.Series,
    *,
    current_age: int,
    horizon: int,
    prior_strength: float = 2.0,
    prior_probability: float = 0.5,
) -> tuple[float, int]:
    numeric = pd.to_numeric(durations, errors="coerce").dropna().astype(int)
    at_risk = numeric.loc[numeric >= current_age]
    support = int(len(at_risk))
    successes = int((at_risk >= current_age + horizon).sum())
    probability = (successes + prior_strength * prior_probability) / (support + prior_strength)
    return float(np.clip(probability, 0.0, 1.0)), support


def _state_age(sample: pd.DataFrame, index: int) -> int:
    current_state = sample.loc[index, "lifecycle_state"]
    age = 1
    position = index - 1
    while position >= 0 and sample.loc[position, "lifecycle_state"] == current_state:
        age += 1
        position -= 1
    return age


def build_survival_history(
    lifecycle_history: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = (1, 3, 5),
    minimum_completed_spells: int = 3,
) -> pd.DataFrame:
    if lifecycle_history.empty:
        return pd.DataFrame()
    if not horizons or min(horizons) < 1:
        raise ValueError("horizons debe contener enteros positivos")
    if minimum_completed_spells < 1:
        raise ValueError("minimum_completed_spells debe ser positivo")

    history = lifecycle_history.copy()
    history["origin_date"] = pd.to_datetime(history["origin_date"])
    spells = _build_state_spells(history)
    rows: list[dict[str, object]] = []

    for champion, sample in history.groupby("champion", sort=True):
        sample = sample.sort_values("origin_date").reset_index(drop=True)
        champion_spells = spells.loc[spells["champion"] == champion].copy()
        for index, row in sample.iterrows():
            current_date = row["origin_date"]
            state = str(row["lifecycle_state"])
            age = _state_age(sample, index)

            completed_own = champion_spells.loc[
                (champion_spells["lifecycle_state"] == state)
                & champion_spells["ended_with_transition"]
                & (pd.to_datetime(champion_spells["end_date"]) < current_date)
            ]
            completed_pool = spells.loc[
                (spells["lifecycle_state"] == state)
                & spells["ended_with_transition"]
                & (pd.to_datetime(spells["end_date"]) < current_date)
            ]
            reference = completed_own
            reference_scope = "CHAMPION"
            if len(reference) < minimum_completed_spells:
                reference = completed_pool
                reference_scope = "POOLED_STATE"

            record: dict[str, object] = {
                "origin_date": current_date,
                "champion": champion,
                "lifecycle_state": state,
                "state_age": age,
                "completed_spells_available": int(len(reference)),
                "survival_reference_scope": reference_scope,
                "lifecycle_health_score": row.get("lifecycle_health_score"),
                "performance_level": row.get("performance_level"),
                "performance_direction": row.get("performance_direction"),
                "short_advantage": row.get("short_advantage"),
                "advantage_slope": row.get("advantage_slope"),
            }
            probabilities: list[float] = []
            for horizon in horizons:
                probability, support = _empirical_survival_probability(
                    reference["duration_observations"] if not reference.empty else pd.Series(dtype=float),
                    current_age=age,
                    horizon=horizon,
                )
                record[f"survival_probability_{horizon}"] = probability
                record[f"at_risk_support_{horizon}"] = support
                probabilities.append(probability)

            direction_bonus = {
                "IMPROVING": 10.0,
                "STABLE": 3.0,
                "DECLINING": -12.0,
            }.get(str(row.get("performance_direction")), 0.0)
            health = float(pd.to_numeric(pd.Series([row.get("lifecycle_health_score")]), errors="coerce").fillna(50).iloc[0])
            record["survival_confidence_score"] = float(
                np.clip(55.0 * float(np.mean(probabilities)) + 0.35 * health + direction_bonus, 0.0, 100.0)
            )
            rows.append(record)

    return pd.DataFrame(rows).sort_values(["origin_date", "champion"]).reset_index(drop=True)


def build_survival_curves(spells: pd.DataFrame, *, max_horizon: int = 10) -> pd.DataFrame:
    if spells.empty:
        return pd.DataFrame()
    completed = spells.loc[spells["ended_with_transition"]].copy()
    rows: list[dict[str, object]] = []
    for state, sample in completed.groupby("lifecycle_state", sort=True):
        durations = pd.to_numeric(sample["duration_observations"], errors="coerce").dropna()
        for horizon in range(1, max_horizon + 1):
            rows.append(
                {
                    "lifecycle_state": state,
                    "horizon": horizon,
                    "completed_spells": int(len(durations)),
                    "survival_probability": float((durations >= horizon).mean()) if len(durations) else np.nan,
                    "median_duration": float(durations.median()) if len(durations) else np.nan,
                    "mean_duration": float(durations.mean()) if len(durations) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_current_survival_status(survival_history: pd.DataFrame) -> pd.DataFrame:
    if survival_history.empty:
        return pd.DataFrame()
    latest = (
        survival_history.sort_values("origin_date")
        .groupby("champion", as_index=False)
        .tail(1)
        .copy()
    )
    probability_columns = sorted(
        [column for column in latest.columns if column.startswith("survival_probability_")],
        key=lambda value: int(value.rsplit("_", 1)[-1]),
    )
    longest = probability_columns[-1]
    latest["survival_outlook"] = pd.cut(
        latest[longest],
        bins=[-np.inf, 0.30, 0.55, 0.75, np.inf],
        labels=["FRAGILE", "UNCERTAIN", "PERSISTENT", "HIGH_PERSISTENCE"],
    ).astype(str)
    latest["survival_adjusted_authority"] = (
        pd.to_numeric(latest[longest], errors="coerce").fillna(0.5)
        * pd.to_numeric(latest["survival_confidence_score"], errors="coerce").fillna(50.0)
    ).clip(0, 100)
    return latest.sort_values("survival_adjusted_authority", ascending=False).reset_index(drop=True)


def run_champion_survival_laboratory(
    selections: pd.DataFrame,
    daily_states: pd.DataFrame | None = None,
    *,
    short_window: int = 3,
    long_window: int = 6,
    minimum_history: int = 4,
    minimum_state_persistence: int = 2,
    horizons: tuple[int, ...] = (1, 3, 5),
    minimum_completed_spells: int = 3,
) -> ChampionSurvivalResult:
    lifecycle = run_champion_lifecycle_laboratory(
        selections,
        daily_states,
        short_window=short_window,
        long_window=long_window,
        minimum_history=minimum_history,
        minimum_state_persistence=minimum_state_persistence,
    )
    spells = _build_state_spells(lifecycle.lifecycle_history)
    survival_history = build_survival_history(
        lifecycle.lifecycle_history,
        horizons=horizons,
        minimum_completed_spells=minimum_completed_spells,
    )
    current = build_current_survival_status(survival_history)
    curves = build_survival_curves(spells, max_horizon=max(horizons))

    summary = pd.DataFrame()
    if not current.empty:
        longest = f"survival_probability_{max(horizons)}"
        summary = pd.DataFrame(
            [
                {
                    "champions": int(current["champion"].nunique()),
                    "mean_survival_probability": float(current[longest].mean()),
                    "high_persistence_champions": int(
                        current["survival_outlook"].isin(["PERSISTENT", "HIGH_PERSISTENCE"]).sum()
                    ),
                    "top_champion": current.iloc[0]["champion"],
                    "top_lifecycle_state": current.iloc[0]["lifecycle_state"],
                    "top_survival_outlook": current.iloc[0]["survival_outlook"],
                    "top_survival_adjusted_authority": float(
                        current.iloc[0]["survival_adjusted_authority"]
                    ),
                    "completed_state_spells": int(spells["ended_with_transition"].sum()) if not spells.empty else 0,
                }
            ]
        )

    return ChampionSurvivalResult(
        survival_history=survival_history,
        current_survival=current,
        state_duration_spells=spells,
        survival_curves=curves,
        summary=summary,
        lifecycle_result=lifecycle,
    )

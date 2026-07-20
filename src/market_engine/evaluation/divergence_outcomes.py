from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from market_engine.evaluation.sentinel_divergence import (
    SentinelDivergenceResult,
    run_sentinel_divergence_laboratory,
)


@dataclass(frozen=True)
class DivergenceOutcomeResult:
    daily_outcomes: pd.DataFrame
    state_outcomes: pd.DataFrame
    transition_matrix: pd.DataFrame
    episode_outcomes: pd.DataFrame
    summary: pd.DataFrame
    divergence_result: SentinelDivergenceResult


def _validate_horizons(horizons: Iterable[int]) -> tuple[int, ...]:
    values = tuple(sorted(set(int(value) for value in horizons)))
    if not values or any(value < 1 for value in values):
        raise ValueError("horizons debe contener enteros positivos")
    return values


def _forward_minimum_return(series: pd.Series, horizon: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(values), np.nan)
    for position in range(len(values) - horizon):
        future = values[position + 1 : position + horizon + 1]
        if np.isfinite(values[position]) and np.isfinite(future).all():
            result[position] = float(np.min(future / values[position] - 1.0))
    return pd.Series(result, index=series.index)


def _add_forward_outcomes(daily: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    data = daily.sort_values("date").reset_index(drop=True).copy()
    for horizon in horizons:
        sample_future = data["sample_market_cap_index"].shift(-horizon)
        reference_future = data["reference_market_cap_index"].shift(-horizon)
        future_spread = data["leadership_spread"].shift(-horizon)

        data[f"sample_return_forward_{horizon}"] = (
            sample_future / data["sample_market_cap_index"] - 1.0
        )
        data[f"reference_return_forward_{horizon}"] = (
            reference_future / data["reference_market_cap_index"] - 1.0
        )
        data[f"sample_future_drawdown_{horizon}"] = _forward_minimum_return(
            data["sample_market_cap_index"], horizon
        )
        data[f"reference_future_drawdown_{horizon}"] = _forward_minimum_return(
            data["reference_market_cap_index"], horizon
        )
        data[f"spread_change_forward_{horizon}"] = future_spread - data["leadership_spread"]
        data[f"state_forward_{horizon}"] = data["divergence_state"].shift(-horizon)
        data[f"outcome_completed_{horizon}"] = reference_future.notna()
    return data


def _summarize_states(daily: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon in horizons:
        completed = daily.loc[daily[f"outcome_completed_{horizon}"]].copy()
        for state, group in completed.groupby("divergence_state", sort=True):
            reference_returns = group[f"reference_return_forward_{horizon}"]
            sample_returns = group[f"sample_return_forward_{horizon}"]
            rows.append(
                {
                    "divergence_state": state,
                    "horizon": horizon,
                    "observations": int(len(group)),
                    "mean_reference_return": float(reference_returns.mean()),
                    "median_reference_return": float(reference_returns.median()),
                    "reference_positive_rate": float((reference_returns > 0).mean()),
                    "mean_sample_return": float(sample_returns.mean()),
                    "sample_positive_rate": float((sample_returns > 0).mean()),
                    "mean_reference_future_drawdown": float(
                        group[f"reference_future_drawdown_{horizon}"].mean()
                    ),
                    "mean_sample_future_drawdown": float(
                        group[f"sample_future_drawdown_{horizon}"].mean()
                    ),
                    "mean_spread_change": float(
                        group[f"spread_change_forward_{horizon}"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _build_transition_matrix(daily: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon in horizons:
        pairs = daily[["divergence_state", f"state_forward_{horizon}"]].dropna()
        if pairs.empty:
            continue
        counts = (
            pairs.groupby(["divergence_state", f"state_forward_{horizon}"])
            .size()
            .rename("transitions")
            .reset_index()
        )
        totals = counts.groupby("divergence_state")["transitions"].transform("sum")
        counts["transition_probability"] = counts["transitions"] / totals
        counts["horizon"] = horizon
        counts = counts.rename(
            columns={
                "divergence_state": "from_state",
                f"state_forward_{horizon}": "to_state",
            }
        )
        rows.extend(counts.to_dict("records"))
    return pd.DataFrame(rows)


def _episode_outcomes(
    daily: pd.DataFrame,
    episodes: pd.DataFrame,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame()
    by_date = daily.set_index("date")
    rows: list[dict[str, object]] = []
    for _, episode in episodes.iterrows():
        anchor_date = episode["end_date"] if pd.notna(episode["end_date"]) else episode["peak_divergence_date"]
        if anchor_date not in by_date.index:
            continue
        row = by_date.loc[anchor_date]
        record = episode.to_dict()
        record["outcome_anchor_date"] = anchor_date
        for horizon in horizons:
            for prefix in (
                "sample_return_forward",
                "reference_return_forward",
                "sample_future_drawdown",
                "reference_future_drawdown",
                "spread_change_forward",
                "state_forward",
                "outcome_completed",
            ):
                column = f"{prefix}_{horizon}"
                record[column] = row.get(column, np.nan)
        rows.append(record)
    return pd.DataFrame(rows)


def run_divergence_outcome_laboratory(
    sample: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    horizons: Iterable[int] = (1, 5, 10, 20),
    smoothing_window: int = 5,
    spread_window: int = 20,
    extreme_zscore: float = 2.0,
    minimum_episode_observations: int = 3,
) -> DivergenceOutcomeResult:
    validated_horizons = _validate_horizons(horizons)
    divergence = run_sentinel_divergence_laboratory(
        sample,
        reference,
        smoothing_window=smoothing_window,
        spread_window=spread_window,
        extreme_zscore=extreme_zscore,
        minimum_episode_observations=minimum_episode_observations,
    )
    daily = _add_forward_outcomes(divergence.daily, validated_horizons)
    states = _summarize_states(daily, validated_horizons)
    transitions = _build_transition_matrix(daily, validated_horizons)
    episodes = _episode_outcomes(daily, divergence.episodes, validated_horizons)

    latest = daily.iloc[-1]
    summary = pd.DataFrame(
        [
            {
                "start_date": daily.iloc[0]["date"],
                "end_date": latest["date"],
                "observations": int(len(daily)),
                "horizons": ",".join(str(value) for value in validated_horizons),
                "states_evaluated": int(daily["divergence_state"].nunique()),
                "episodes_evaluated": int(len(episodes)),
                "latest_divergence_state": latest["divergence_state"],
                "latest_leadership_spread": float(latest["leadership_spread"]),
                "maximum_complete_horizon": max(validated_horizons),
            }
        ]
    )
    return DivergenceOutcomeResult(
        daily_outcomes=daily,
        state_outcomes=states,
        transition_matrix=transitions,
        episode_outcomes=episodes,
        summary=summary,
        divergence_result=divergence,
    )

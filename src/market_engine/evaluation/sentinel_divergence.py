from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from market_engine.evaluation.sample_morphology_fidelity import compare_sample_to_reference


@dataclass(frozen=True)
class SentinelDivergenceResult:
    daily: pd.DataFrame
    episodes: pd.DataFrame
    state_summary: pd.DataFrame
    extremes: pd.DataFrame
    summary: pd.DataFrame


def _safe_ratio(numerator: pd.Series, denominator: pd.Series, epsilon: float = 1e-9) -> pd.Series:
    denominator_safe = denominator.where(denominator.abs() > epsilon)
    result = numerator / denominator_safe
    return result.replace([np.inf, -np.inf], np.nan)


def _classify_state(row: pd.Series, extreme_zscore: float) -> str:
    sample_return = row.get("sample_daily_return")
    reference_return = row.get("reference_daily_return")
    spread = row.get("leadership_spread")
    velocity = row.get("spread_velocity")
    acceleration = row.get("spread_acceleration")
    zscore = row.get("spread_zscore")
    drawdown_amp = row.get("sample_drawdown_amplification")

    if pd.notna(sample_return) and pd.notna(reference_return):
        if np.sign(sample_return) != np.sign(reference_return) and not np.isclose(sample_return, 0.0) and not np.isclose(reference_return, 0.0):
            return "DIRECTIONAL_BREAK"

    if pd.notna(drawdown_amp) and drawdown_amp >= 1.15 and sample_return < 0 and reference_return < 0:
        return "STRESS_AMPLIFICATION"

    if pd.notna(zscore) and zscore >= extreme_zscore and spread > 0:
        return "EXTREME_CONCENTRATION"

    if spread > 0 and velocity < 0 and acceleration < 0:
        return "SENTINEL_WEAKENING"

    if spread > 0 and velocity > 0:
        return "SENTINEL_LEADERSHIP"

    if velocity < 0 and reference_return > sample_return:
        return "REFERENCE_CATCH_UP"

    return "BROAD_ALIGNMENT"


def _detect_episodes(daily: pd.DataFrame, minimum_episode_observations: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    active_states = {
        "EXTREME_CONCENTRATION",
        "SENTINEL_WEAKENING",
        "STRESS_AMPLIFICATION",
        "DIRECTIONAL_BREAK",
    }
    start: int | None = None
    episode_number = 0

    for position, row in daily.iterrows():
        is_active = row["divergence_state"] in active_states
        if start is None and is_active:
            start = position
        elif start is not None and not is_active:
            segment = daily.iloc[start:position]
            if len(segment) >= minimum_episode_observations:
                episode_number += 1
                rows.append(_episode_row(segment, episode_number, completed=True))
            start = None

    if start is not None:
        segment = daily.iloc[start:]
        if len(segment) >= minimum_episode_observations:
            episode_number += 1
            rows.append(_episode_row(segment, episode_number, completed=False))

    return pd.DataFrame(rows)


def _episode_row(segment: pd.DataFrame, number: int, *, completed: bool) -> dict[str, object]:
    dominant_state = segment["divergence_state"].mode().iloc[0]
    peak_position = segment["leadership_spread"].abs().idxmax()
    peak_row = segment.loc[peak_position]
    return {
        "episode_id": f"DIVERGENCE_{number:03d}",
        "start_date": segment.iloc[0]["date"],
        "peak_divergence_date": peak_row["date"],
        "end_date": segment.iloc[-1]["date"] if completed else pd.NaT,
        "initial_spread": float(segment.iloc[0]["leadership_spread"]),
        "maximum_spread": float(segment["leadership_spread"].max()),
        "minimum_spread": float(segment["leadership_spread"].min()),
        "maximum_abs_zscore": float(segment["spread_zscore"].abs().max()),
        "duration_observations": int(len(segment)),
        "dominant_state": dominant_state,
        "sample_return_during_episode": float(segment["sample_market_cap_index"].iloc[-1] / segment["sample_market_cap_index"].iloc[0] - 1.0),
        "reference_return_during_episode": float(segment["reference_market_cap_index"].iloc[-1] / segment["reference_market_cap_index"].iloc[0] - 1.0),
        "spread_change": float(segment["leadership_spread"].iloc[-1] - segment["leadership_spread"].iloc[0]),
        "episode_completed": completed,
    }


def run_sentinel_divergence_laboratory(
    sample: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    smoothing_window: int = 5,
    spread_window: int = 20,
    extreme_zscore: float = 2.0,
    minimum_episode_observations: int = 3,
) -> SentinelDivergenceResult:
    if spread_window < 3:
        raise ValueError("spread_window debe ser mayor o igual a 3")
    if extreme_zscore <= 0:
        raise ValueError("extreme_zscore debe ser positivo")
    if minimum_episode_observations < 1:
        raise ValueError("minimum_episode_observations debe ser mayor o igual a 1")

    fidelity = compare_sample_to_reference(
        sample,
        reference,
        smoothing_window=smoothing_window,
        rolling_window=spread_window,
    )
    daily = fidelity.aligned_geometry.copy()
    daily["leadership_spread"] = daily["sample_market_cap_index"] - daily["reference_market_cap_index"]
    daily["spread_velocity"] = daily["leadership_spread"].diff()
    daily["spread_acceleration"] = daily["spread_velocity"].diff()
    daily["rolling_spread_mean"] = daily["leadership_spread"].rolling(spread_window, min_periods=3).mean()
    daily["rolling_spread_std"] = daily["leadership_spread"].rolling(spread_window, min_periods=3).std(ddof=0)
    daily["spread_zscore"] = _safe_ratio(
        daily["leadership_spread"] - daily["rolling_spread_mean"],
        daily["rolling_spread_std"],
    ).fillna(0.0)
    daily["sample_drawdown_amplification"] = _safe_ratio(
        daily["sample_drawdown_pct"].abs(),
        daily["reference_drawdown_pct"].abs(),
    )
    daily["sample_return_amplification"] = _safe_ratio(
        daily["sample_daily_return"],
        daily["reference_daily_return"],
    )
    daily["directional_disagreement"] = (
        np.sign(daily["sample_daily_return"].fillna(0.0))
        != np.sign(daily["reference_daily_return"].fillna(0.0))
    )
    daily["divergence_state"] = daily.apply(_classify_state, axis=1, extreme_zscore=extreme_zscore)

    episodes = _detect_episodes(daily, minimum_episode_observations)
    state_summary = (
        daily.groupby("divergence_state", as_index=False)
        .agg(
            observations=("date", "size"),
            mean_spread=("leadership_spread", "mean"),
            mean_abs_zscore=("spread_zscore", lambda s: s.abs().mean()),
            mean_drawdown_amplification=("sample_drawdown_amplification", "mean"),
        )
        .sort_values("observations", ascending=False)
        .reset_index(drop=True)
    )
    extremes = pd.concat(
        [
            daily.nlargest(10, "leadership_spread").assign(extreme_type="HIGHEST_SPREAD"),
            daily.nsmallest(10, "leadership_spread").assign(extreme_type="LOWEST_SPREAD"),
            daily.nlargest(10, "spread_zscore").assign(extreme_type="HIGHEST_ZSCORE"),
            daily.nsmallest(10, "spread_zscore").assign(extreme_type="LOWEST_ZSCORE"),
        ],
        ignore_index=True,
    )
    latest = daily.iloc[-1]
    summary = pd.DataFrame([
        {
            "start_date": daily.iloc[0]["date"],
            "end_date": latest["date"],
            "observations": int(len(daily)),
            "latest_spread": float(latest["leadership_spread"]),
            "latest_spread_velocity": float(latest["spread_velocity"]) if pd.notna(latest["spread_velocity"]) else np.nan,
            "latest_spread_acceleration": float(latest["spread_acceleration"]) if pd.notna(latest["spread_acceleration"]) else np.nan,
            "latest_spread_zscore": float(latest["spread_zscore"]),
            "latest_divergence_state": latest["divergence_state"],
            "maximum_spread": float(daily["leadership_spread"].max()),
            "minimum_spread": float(daily["leadership_spread"].min()),
            "directional_breaks": int((daily["divergence_state"] == "DIRECTIONAL_BREAK").sum()),
            "stress_amplification_observations": int((daily["divergence_state"] == "STRESS_AMPLIFICATION").sum()),
            "detected_divergence_episodes": int(len(episodes)),
        }
    ])
    return SentinelDivergenceResult(
        daily=daily,
        episodes=episodes,
        state_summary=state_summary,
        extremes=extremes,
        summary=summary,
    )

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable

import numpy as np
import pandas as pd

from market_engine.evaluation.historical_outcome_distributions import (
    HistoricalOutcomeDistributionResult,
    run_historical_outcome_distribution_laboratory,
)


@dataclass(frozen=True)
class DistributionReliabilityCalibrationResult:
    calibrated_distributions: pd.DataFrame
    calibrated_probabilities: pd.DataFrame
    calibrated_drawdowns: pd.DataFrame
    calibrated_ranking: pd.DataFrame
    summary: pd.DataFrame
    distribution_result: HistoricalOutcomeDistributionResult


def _non_overlapping_count(positions: np.ndarray, horizon: int) -> int:
    if len(positions) == 0:
        return 0
    count = 1
    last = int(positions[0])
    for value in positions[1:]:
        current = int(value)
        if current - last >= horizon:
            count += 1
            last = current
    return count


def _effective_sample_size(values: pd.Series, horizon: int) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna().reset_index(drop=True)
    n = len(clean)
    if n <= 2:
        return float(n)
    maximum_lag = min(max(1, horizon - 1), n - 1)
    correlations: list[float] = []
    for lag in range(1, maximum_lag + 1):
        correlation = clean.autocorr(lag=lag)
        if pd.notna(correlation) and correlation > 0:
            correlations.append(float(correlation))
    denominator = 1.0 + 2.0 * sum(correlations)
    return float(np.clip(n / denominator, 1.0, n))


def _wilson_interval(successes: int, sample_size: float, z: float = 1.959963984540054) -> tuple[float, float]:
    if sample_size <= 0:
        return np.nan, np.nan
    proportion = float(successes) / max(float(successes), sample_size) if successes > sample_size else float(successes) / sample_size
    proportion = float(np.clip(proportion, 0.0, 1.0))
    denominator = 1.0 + z * z / sample_size
    center = (proportion + z * z / (2.0 * sample_size)) / denominator
    margin = z * sqrt((proportion * (1.0 - proportion) + z * z / (4.0 * sample_size)) / sample_size) / denominator
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def _calibrated_stability(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if values.empty:
        return 0.0
    robust_dispersion = float(values.quantile(0.75) - values.quantile(0.25))
    standard_deviation = float(values.std(ddof=0))
    downside = max(0.0, -float(values.quantile(0.10)))
    risk_scale = 0.45 * robust_dispersion + 0.35 * standard_deviation + 0.20 * downside
    return float(np.clip(100.0 / (1.0 + 25.0 * risk_scale), 0.0, 100.0))


def run_distribution_reliability_calibration_laboratory(
    sample: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    horizons: Iterable[int] = (1, 5, 10, 20),
    smoothing_window: int = 5,
    spread_window: int = 20,
    extreme_zscore: float = 2.0,
    minimum_episode_observations: int = 3,
    confidence_prior: float = 30.0,
) -> DistributionReliabilityCalibrationResult:
    if confidence_prior <= 0:
        raise ValueError("confidence_prior debe ser positivo")

    base = run_historical_outcome_distribution_laboratory(
        sample,
        reference,
        horizons=horizons,
        smoothing_window=smoothing_window,
        spread_window=spread_window,
        extreme_zscore=extreme_zscore,
        minimum_episode_observations=minimum_episode_observations,
    )
    daily = base.outcome_result.daily_outcomes.copy().reset_index(drop=True)
    validated_horizons = tuple(sorted(set(int(value) for value in horizons)))

    distribution_rows: list[dict[str, object]] = []
    probability_rows: list[dict[str, object]] = []
    drawdown_rows: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []

    base_ranking = base.ranking.set_index(["divergence_state", "horizon"])

    for horizon in validated_horizons:
        completed = daily.loc[daily[f"outcome_completed_{horizon}"]].copy()
        completed["_position"] = completed.index
        for state, group in completed.groupby("divergence_state", sort=True):
            returns = pd.to_numeric(group[f"reference_return_forward_{horizon}"], errors="coerce").dropna()
            drawdowns = pd.to_numeric(group[f"reference_future_drawdown_{horizon}"], errors="coerce").dropna().clip(upper=0.0)
            raw_n = int(len(returns))
            effective_n = _effective_sample_size(returns, horizon)
            independent_n = _non_overlapping_count(group.loc[returns.index, "_position"].to_numpy(), horizon)
            effective_n = float(min(effective_n, max(1, independent_n)))
            confidence_weight = float(effective_n / (effective_n + confidence_prior))
            positive_rate = float((returns > 0).mean()) if raw_n else np.nan
            effective_successes = int(round(positive_rate * effective_n)) if raw_n else 0
            lower, upper = _wilson_interval(effective_successes, effective_n)

            distribution_rows.append({
                "divergence_state": state,
                "horizon": horizon,
                "raw_observations": raw_n,
                "independent_observations": independent_n,
                "effective_sample_size": effective_n,
                "confidence_weight": confidence_weight,
                "mean_reference_return": float(returns.mean()),
                "median_reference_return": float(returns.median()),
                "standard_deviation": float(returns.std(ddof=0)),
                "p10": float(returns.quantile(0.10)),
                "p25": float(returns.quantile(0.25)),
                "p50": float(returns.quantile(0.50)),
                "p75": float(returns.quantile(0.75)),
                "p90": float(returns.quantile(0.90)),
            })
            probability_rows.append({
                "divergence_state": state,
                "horizon": horizon,
                "raw_observations": raw_n,
                "effective_sample_size": effective_n,
                "probability_positive": positive_rate,
                "positive_rate_lower_95": lower,
                "positive_rate_upper_95": upper,
                "probability_loss_lt_minus_5pct": float((returns < -0.05).mean()),
                "probability_gain_gt_5pct": float((returns > 0.05).mean()),
            })
            drawdown_rows.append({
                "divergence_state": state,
                "horizon": horizon,
                "raw_observations": int(len(drawdowns)),
                "mean_future_drawdown": float(drawdowns.mean()),
                "p10_future_drawdown": float(drawdowns.quantile(0.10)),
                "worst_future_drawdown": float(drawdowns.min()),
                "probability_drawdown_gt_5pct": float((drawdowns < -0.05).mean()),
                "probability_drawdown_gt_10pct": float((drawdowns < -0.10).mean()),
            })

            key = (state, horizon)
            raw_opportunity = float(base_ranking.loc[key, "opportunity_index"]) if key in base_ranking.index else 0.0
            raw_tail_risk = float(base_ranking.loc[key, "tail_risk_index"]) if key in base_ranking.index else 100.0
            stability = _calibrated_stability(returns)
            raw_balanced = float(np.clip(0.40 * stability + 0.40 * raw_opportunity + 0.20 * (100.0 - raw_tail_risk), 0.0, 100.0))
            calibrated_balanced = float(50.0 + confidence_weight * (raw_balanced - 50.0))
            ranking_rows.append({
                "divergence_state": state,
                "horizon": horizon,
                "raw_observations": raw_n,
                "independent_observations": independent_n,
                "effective_sample_size": effective_n,
                "confidence_weight": confidence_weight,
                "calibrated_stability_score": stability,
                "tail_risk_index": raw_tail_risk,
                "opportunity_index": raw_opportunity,
                "raw_balanced_score": raw_balanced,
                "calibrated_balanced_score": calibrated_balanced,
            })

    distributions = pd.DataFrame(distribution_rows)
    probabilities = pd.DataFrame(probability_rows)
    drawdowns = pd.DataFrame(drawdown_rows)
    ranking = pd.DataFrame(ranking_rows).sort_values(
        ["horizon", "calibrated_balanced_score"], ascending=[True, False]
    ).reset_index(drop=True)

    latest = daily.iloc[-1]
    current_state = latest["divergence_state"]
    current = ranking.loc[ranking["divergence_state"] == current_state]
    longest = current.sort_values("horizon").iloc[-1] if not current.empty else None
    summary = pd.DataFrame([{
        "start_date": daily.iloc[0]["date"],
        "end_date": latest["date"],
        "latest_divergence_state": current_state,
        "latest_leadership_spread": float(latest["leadership_spread"]),
        "states_evaluated": int(daily["divergence_state"].nunique()),
        "horizons": ",".join(str(value) for value in validated_horizons),
        "confidence_prior": confidence_prior,
        "current_state_longest_horizon": int(longest["horizon"]) if longest is not None else np.nan,
        "current_state_effective_sample_size": float(longest["effective_sample_size"]) if longest is not None else np.nan,
        "current_state_confidence_weight": float(longest["confidence_weight"]) if longest is not None else np.nan,
        "current_state_calibrated_score": float(longest["calibrated_balanced_score"]) if longest is not None else np.nan,
    }])

    return DistributionReliabilityCalibrationResult(
        calibrated_distributions=distributions,
        calibrated_probabilities=probabilities,
        calibrated_drawdowns=drawdowns,
        calibrated_ranking=ranking,
        summary=summary,
        distribution_result=base,
    )

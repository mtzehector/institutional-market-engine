from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from market_engine.evaluation.divergence_outcomes import (
    DivergenceOutcomeResult,
    run_divergence_outcome_laboratory,
)


@dataclass(frozen=True)
class HistoricalOutcomeDistributionResult:
    distributions: pd.DataFrame
    probabilities: pd.DataFrame
    drawdown_probabilities: pd.DataFrame
    ranking: pd.DataFrame
    summary: pd.DataFrame
    outcome_result: DivergenceOutcomeResult


def _validate_thresholds(values: Iterable[float], *, name: str) -> tuple[float, ...]:
    parsed = tuple(sorted(set(float(value) for value in values)))
    if not parsed or any(value <= 0 for value in parsed):
        raise ValueError(f"{name} debe contener valores positivos")
    return parsed


def _percentiles(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {
            "minimum": np.nan,
            "p05": np.nan,
            "p10": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "maximum": np.nan,
        }
    quantiles = values.quantile([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    return {
        "minimum": float(values.min()),
        "p05": float(quantiles.loc[0.05]),
        "p10": float(quantiles.loc[0.10]),
        "p25": float(quantiles.loc[0.25]),
        "p50": float(quantiles.loc[0.50]),
        "p75": float(quantiles.loc[0.75]),
        "p90": float(quantiles.loc[0.90]),
        "p95": float(quantiles.loc[0.95]),
        "maximum": float(values.max()),
    }


def _stability_score(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if values.empty:
        return 0.0
    dispersion = float(values.quantile(0.75) - values.quantile(0.25))
    downside = abs(float(values.quantile(0.10)))
    penalty = 12.0 * dispersion + 6.0 * downside
    return float(np.clip(100.0 - penalty, 0.0, 100.0))


def _tail_risk_index(returns: pd.Series, drawdowns: pd.Series) -> float:
    return_values = pd.to_numeric(returns, errors="coerce").dropna()
    drawdown_values = pd.to_numeric(drawdowns, errors="coerce").dropna()
    if return_values.empty:
        return 100.0
    p05_loss = max(0.0, -float(return_values.quantile(0.05)))
    drawdown_p10 = max(0.0, -float(drawdown_values.quantile(0.10))) if not drawdown_values.empty else 0.0
    loss_probability = float((return_values < 0).mean())
    score = 450.0 * p05_loss + 300.0 * drawdown_p10 + 25.0 * loss_probability
    return float(np.clip(score, 0.0, 100.0))


def _opportunity_index(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if values.empty:
        return 0.0
    positive_rate = float((values > 0).mean())
    gain_05 = float((values > 0.05).mean())
    gain_10 = float((values > 0.10).mean())
    median_gain = max(0.0, float(values.median()))
    score = 45.0 * positive_rate + 25.0 * gain_05 + 20.0 * gain_10 + min(10.0, 200.0 * median_gain)
    return float(np.clip(score, 0.0, 100.0))


def _distribution_rows(
    daily: pd.DataFrame,
    horizons: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    distributions: list[dict[str, object]] = []
    probabilities: list[dict[str, object]] = []
    drawdown_rows: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []

    for horizon in horizons:
        completed = daily.loc[daily[f"outcome_completed_{horizon}"]].copy()
        for state, group in completed.groupby("divergence_state", sort=True):
            reference_returns = pd.to_numeric(
                group[f"reference_return_forward_{horizon}"], errors="coerce"
            ).dropna()
            reference_drawdowns = pd.to_numeric(
                group[f"reference_future_drawdown_{horizon}"], errors="coerce"
            ).dropna()
            sample_returns = pd.to_numeric(
                group[f"sample_return_forward_{horizon}"], errors="coerce"
            ).dropna()

            distribution = {
                "divergence_state": state,
                "horizon": horizon,
                "observations": int(len(reference_returns)),
                "mean_reference_return": float(reference_returns.mean()),
                "median_reference_return": float(reference_returns.median()),
                "standard_deviation": float(reference_returns.std(ddof=0)),
                **_percentiles(reference_returns),
                "mean_sample_return": float(sample_returns.mean()),
            }
            distributions.append(distribution)

            probabilities.append(
                {
                    "divergence_state": state,
                    "horizon": horizon,
                    "observations": int(len(reference_returns)),
                    "probability_positive": float((reference_returns > 0).mean()),
                    "probability_gain_gt_2pct": float((reference_returns > 0.02).mean()),
                    "probability_gain_gt_5pct": float((reference_returns > 0.05).mean()),
                    "probability_gain_gt_10pct": float((reference_returns > 0.10).mean()),
                    "probability_loss_lt_minus_2pct": float((reference_returns < -0.02).mean()),
                    "probability_loss_lt_minus_5pct": float((reference_returns < -0.05).mean()),
                    "probability_loss_lt_minus_10pct": float((reference_returns < -0.10).mean()),
                }
            )

            drawdown_rows.append(
                {
                    "divergence_state": state,
                    "horizon": horizon,
                    "observations": int(len(reference_drawdowns)),
                    "probability_drawdown_gt_2pct": float((reference_drawdowns < -0.02).mean()),
                    "probability_drawdown_gt_5pct": float((reference_drawdowns < -0.05).mean()),
                    "probability_drawdown_gt_10pct": float((reference_drawdowns < -0.10).mean()),
                    "probability_drawdown_gt_20pct": float((reference_drawdowns < -0.20).mean()),
                    "mean_future_drawdown": float(reference_drawdowns.mean()),
                    "p10_future_drawdown": float(reference_drawdowns.quantile(0.10)),
                    "worst_future_drawdown": float(reference_drawdowns.min()),
                }
            )

            stability = _stability_score(reference_returns)
            tail_risk = _tail_risk_index(reference_returns, reference_drawdowns)
            opportunity = _opportunity_index(reference_returns)
            balanced = float(
                np.clip(0.40 * stability + 0.40 * opportunity + 0.20 * (100.0 - tail_risk), 0.0, 100.0)
            )
            ranking_rows.append(
                {
                    "divergence_state": state,
                    "horizon": horizon,
                    "observations": int(len(reference_returns)),
                    "outcome_stability_score": stability,
                    "tail_risk_index": tail_risk,
                    "opportunity_index": opportunity,
                    "balanced_outcome_score": balanced,
                }
            )

    distributions_frame = pd.DataFrame(distributions)
    probabilities_frame = pd.DataFrame(probabilities)
    drawdowns_frame = pd.DataFrame(drawdown_rows)
    ranking_frame = pd.DataFrame(ranking_rows)
    if not ranking_frame.empty:
        ranking_frame = ranking_frame.sort_values(
            ["horizon", "balanced_outcome_score"], ascending=[True, False]
        ).reset_index(drop=True)
    return distributions_frame, probabilities_frame, drawdowns_frame, ranking_frame


def run_historical_outcome_distribution_laboratory(
    sample: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    horizons: Iterable[int] = (1, 5, 10, 20),
    smoothing_window: int = 5,
    spread_window: int = 20,
    extreme_zscore: float = 2.0,
    minimum_episode_observations: int = 3,
) -> HistoricalOutcomeDistributionResult:
    outcome = run_divergence_outcome_laboratory(
        sample,
        reference,
        horizons=horizons,
        smoothing_window=smoothing_window,
        spread_window=spread_window,
        extreme_zscore=extreme_zscore,
        minimum_episode_observations=minimum_episode_observations,
    )
    validated_horizons = tuple(
        sorted(
            int(value)
            for value in str(outcome.summary.iloc[0]["horizons"]).split(",")
            if str(value).strip()
        )
    )
    distributions, probabilities, drawdowns, ranking = _distribution_rows(
        outcome.daily_outcomes, validated_horizons
    )

    latest = outcome.daily_outcomes.iloc[-1]
    current_state = latest["divergence_state"]
    current_ranking = ranking.loc[ranking["divergence_state"] == current_state]
    best_current = (
        current_ranking.sort_values("horizon").iloc[-1]
        if not current_ranking.empty
        else None
    )
    summary = pd.DataFrame(
        [
            {
                "start_date": outcome.daily_outcomes.iloc[0]["date"],
                "end_date": latest["date"],
                "observations": int(len(outcome.daily_outcomes)),
                "horizons": ",".join(str(value) for value in validated_horizons),
                "states_evaluated": int(outcome.daily_outcomes["divergence_state"].nunique()),
                "latest_divergence_state": current_state,
                "latest_leadership_spread": float(latest["leadership_spread"]),
                "current_state_longest_horizon": int(best_current["horizon"]) if best_current is not None else np.nan,
                "current_state_balanced_score": float(best_current["balanced_outcome_score"]) if best_current is not None else np.nan,
                "current_state_tail_risk_index": float(best_current["tail_risk_index"]) if best_current is not None else np.nan,
                "current_state_opportunity_index": float(best_current["opportunity_index"]) if best_current is not None else np.nan,
            }
        ]
    )
    return HistoricalOutcomeDistributionResult(
        distributions=distributions,
        probabilities=probabilities,
        drawdown_probabilities=drawdowns,
        ranking=ranking,
        summary=summary,
        outcome_result=outcome,
    )

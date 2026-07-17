from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from market_engine.evaluation.champion_validation import (
    add_champion_scores,
    build_champion_selections,
)
from market_engine.evaluation.model_evaluation import evaluate_predictions


REGIMES = (
    "RISK_ON_EUPHORIA",
    "RISK_ON_BROAD",
    "QUIET_ACCUMULATION",
    "ROTATION_VOLATILE",
    "RISK_OFF_STRESS",
    "NEUTRAL",
)


@dataclass(frozen=True)
class RegimeTransitionResult:
    daily_states: pd.DataFrame
    transitions: pd.DataFrame
    champion_regime_metrics: pd.DataFrame
    regime_ranking: pd.DataFrame
    transition_impact: pd.DataFrame
    selections: pd.DataFrame


def _causal_reference(series: pd.Series, lookback: int) -> tuple[pd.Series, pd.Series]:
    numeric = pd.to_numeric(series, errors="coerce")
    prior = numeric.shift(1)
    median = prior.rolling(lookback, min_periods=max(4, lookback // 3)).median()
    dispersion = prior.rolling(lookback, min_periods=max(4, lookback // 3)).std(ddof=0)
    dispersion = dispersion.replace(0.0, np.nan)
    return median, dispersion


def build_daily_market_states(frame: pd.DataFrame, lookback: int = 12) -> pd.DataFrame:
    required = {
        "origin_date",
        "ticker",
        "return_1d_pct",
        "return_5d_pct",
        "relative_volume",
        "atr_pct",
        "smart_money_pct",
        "smart_money_slope_5d",
        "qqq_return_1d_pct",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas para detectar régimen: {sorted(missing)}")
    if lookback < 6:
        raise ValueError("lookback debe ser al menos 6")

    data = frame.copy()
    data["origin_date"] = pd.to_datetime(data["origin_date"])
    data["positive_1d"] = pd.to_numeric(data["return_1d_pct"], errors="coerce") > 0
    data["positive_5d"] = pd.to_numeric(data["return_5d_pct"], errors="coerce") > 0
    data["institutional_positive"] = (
        (pd.to_numeric(data["smart_money_pct"], errors="coerce") > 50.0)
        & (pd.to_numeric(data["smart_money_slope_5d"], errors="coerce") > 0.0)
    )

    daily = (
        data.groupby("origin_date")
        .agg(
            universe_size=("ticker", "nunique"),
            qqq_return_1d_pct=("qqq_return_1d_pct", "median"),
            breadth_positive_1d=("positive_1d", "mean"),
            breadth_positive_5d=("positive_5d", "mean"),
            median_relative_volume=("relative_volume", "median"),
            median_atr_pct=("atr_pct", "median"),
            institutional_breadth=("institutional_positive", "mean"),
        )
        .reset_index()
        .sort_values("origin_date")
        .reset_index(drop=True)
    )

    reference_columns = [
        "breadth_positive_1d",
        "breadth_positive_5d",
        "median_relative_volume",
        "median_atr_pct",
        "institutional_breadth",
    ]
    for column in reference_columns:
        median, dispersion = _causal_reference(daily[column], lookback)
        daily[f"reference_{column}"] = median
        daily[f"z_{column}"] = (
            pd.to_numeric(daily[column], errors="coerce") - median
        ) / dispersion

    labels: list[str] = []
    for _, row in daily.iterrows():
        qqq = float(row["qqq_return_1d_pct"]) if pd.notna(row["qqq_return_1d_pct"]) else 0.0
        breadth_1d = float(row["breadth_positive_1d"])
        breadth_5d = float(row["breadth_positive_5d"])
        volume_z = float(row["z_median_relative_volume"]) if pd.notna(row["z_median_relative_volume"]) else 0.0
        atr_z = float(row["z_median_atr_pct"]) if pd.notna(row["z_median_atr_pct"]) else 0.0
        institutional_z = float(row["z_institutional_breadth"]) if pd.notna(row["z_institutional_breadth"]) else 0.0

        if qqq > 0 and breadth_1d >= 0.65 and breadth_5d >= 0.60 and volume_z >= 0.25:
            regime = "RISK_ON_EUPHORIA"
        elif qqq < 0 and breadth_1d <= 0.40 and breadth_5d <= 0.45 and atr_z >= 0.25:
            regime = "RISK_OFF_STRESS"
        elif atr_z >= 0.75 and 0.35 <= breadth_1d <= 0.65:
            regime = "ROTATION_VOLATILE"
        elif institutional_z >= 0.50 and atr_z <= 0.0 and qqq >= -0.25:
            regime = "QUIET_ACCUMULATION"
        elif qqq > 0 and breadth_1d >= 0.55 and breadth_5d >= 0.55:
            regime = "RISK_ON_BROAD"
        else:
            regime = "NEUTRAL"
        labels.append(regime)

    daily["regime"] = labels
    daily["previous_regime"] = daily["regime"].shift(1)
    daily["is_transition"] = daily["regime"] != daily["previous_regime"]
    daily.loc[daily.index[0], "is_transition"] = False

    state_columns = [
        "breadth_positive_1d",
        "breadth_positive_5d",
        "median_relative_volume",
        "median_atr_pct",
        "institutional_breadth",
    ]
    normalized_changes = []
    for column in state_columns:
        median, dispersion = _causal_reference(daily[column], lookback)
        change = pd.to_numeric(daily[column], errors="coerce").diff().abs()
        normalized_changes.append((change / dispersion).replace([np.inf, -np.inf], np.nan))
    daily["transition_strength"] = pd.concat(normalized_changes, axis=1).mean(axis=1).fillna(0.0)
    return daily


def build_transitions(daily_states: pd.DataFrame) -> pd.DataFrame:
    if daily_states.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    previous_transition_index = 0
    for index, row in daily_states.iterrows():
        if not bool(row["is_transition"]):
            continue
        rows.append(
            {
                "transition_date": row["origin_date"],
                "from_regime": row["previous_regime"],
                "to_regime": row["regime"],
                "prior_regime_origin_dates": index - previous_transition_index,
                "transition_strength": row["transition_strength"],
                "qqq_return_1d_pct": row["qqq_return_1d_pct"],
                "breadth_positive_1d": row["breadth_positive_1d"],
                "breadth_positive_5d": row["breadth_positive_5d"],
                "median_relative_volume": row["median_relative_volume"],
                "median_atr_pct": row["median_atr_pct"],
                "institutional_breadth": row["institutional_breadth"],
            }
        )
        previous_transition_index = index
    return pd.DataFrame(rows)


def _quality(summary: pd.Series) -> float:
    rare = float(summary.get("rare_event_f1", 0.0) or 0.0)
    balanced = float(summary.get("balanced_accuracy", 0.0) or 0.0)
    macro = float(summary.get("macro_f1", 0.0) or 0.0)
    brier_value = summary.get("mean_brier_skill", 0.0)
    brier = float(brier_value) if pd.notna(brier_value) else 0.0
    calibration_value = summary.get("mean_calibration_error", 1.0)
    calibration = float(calibration_value) if pd.notna(calibration_value) else 1.0
    return 100.0 * (
        0.35 * rare + 0.25 * balanced + 0.20 * macro + 0.10 * brier + 0.10 * (1.0 - calibration)
    )


def evaluate_champions_by_regime(
    selections: pd.DataFrame,
    daily_states: pd.DataFrame,
    minimum_observations: int = 10,
) -> pd.DataFrame:
    state_map = daily_states[["origin_date", "regime"]].copy()
    data = selections.copy()
    data["origin_date"] = pd.to_datetime(data["origin_date"])
    data = data.merge(state_map, on="origin_date", how="left")
    rows: list[pd.DataFrame] = []
    for (regime, champion), sample in data.groupby(["regime", "champion"], sort=True):
        if len(sample) < minimum_observations:
            continue
        summary = evaluate_predictions(sample, ticker=str(champion)).summary.copy()
        summary = summary.rename(columns={"ticker": "champion"})
        summary["regime"] = regime
        summary["unique_origin_dates"] = int(sample["origin_date"].nunique())
        summary["unique_tickers"] = int(sample["ticker"].nunique())
        summary["regime_quality_score"] = _quality(summary.iloc[0])
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_regime_ranking(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics.copy()
    ranking = metrics.copy()
    ranking["regime_rank"] = ranking.groupby("regime")["regime_quality_score"].rank(
        ascending=False, method="first"
    )
    return ranking.sort_values(["regime", "regime_rank"]).reset_index(drop=True)


def build_transition_impact(
    selections: pd.DataFrame,
    transitions: pd.DataFrame,
    radius: int = 3,
) -> pd.DataFrame:
    if transitions.empty:
        return pd.DataFrame()
    dates = sorted(pd.to_datetime(selections["origin_date"]).dropna().unique())
    date_positions = {pd.Timestamp(value): index for index, value in enumerate(dates)}
    rows: list[dict[str, object]] = []

    for _, transition in transitions.iterrows():
        date = pd.Timestamp(transition["transition_date"])
        if date not in date_positions:
            continue
        position = date_positions[date]
        pre_dates = dates[max(0, position - radius):position]
        post_dates = dates[position:min(len(dates), position + radius)]
        for champion, sample in selections.groupby("champion", sort=False):
            pre = sample.loc[pd.to_datetime(sample["origin_date"]).isin(pre_dates)]
            post = sample.loc[pd.to_datetime(sample["origin_date"]).isin(post_dates)]
            if pre.empty or post.empty:
                continue
            pre_summary = evaluate_predictions(pre, ticker=str(champion)).summary.iloc[0]
            post_summary = evaluate_predictions(post, ticker=str(champion)).summary.iloc[0]
            rows.append(
                {
                    "transition_date": date,
                    "from_regime": transition["from_regime"],
                    "to_regime": transition["to_regime"],
                    "champion": champion,
                    "pre_observations": len(pre),
                    "post_observations": len(post),
                    "pre_quality": _quality(pre_summary),
                    "post_quality": _quality(post_summary),
                    "delta_quality": _quality(post_summary) - _quality(pre_summary),
                    "delta_rare_event_f1": post_summary["rare_event_f1"] - pre_summary["rare_event_f1"],
                    "delta_balanced_accuracy": post_summary["balanced_accuracy"] - pre_summary["balanced_accuracy"],
                    "delta_mean_brier_skill": post_summary["mean_brier_skill"] - pre_summary["mean_brier_skill"],
                    "delta_calibration_error": post_summary["mean_calibration_error"] - pre_summary["mean_calibration_error"],
                }
            )
    return pd.DataFrame(rows)


def run_regime_transition_laboratory(
    prediction_features: pd.DataFrame,
    lookback: int = 12,
    minimum_observations: int = 10,
    transition_radius: int = 3,
) -> RegimeTransitionResult:
    scored = add_champion_scores(prediction_features)
    selections = build_champion_selections(scored)
    daily_states = build_daily_market_states(scored, lookback=lookback)
    transitions = build_transitions(daily_states)
    metrics = evaluate_champions_by_regime(
        selections,
        daily_states,
        minimum_observations=minimum_observations,
    )
    ranking = build_regime_ranking(metrics)
    impact = build_transition_impact(selections, transitions, radius=transition_radius)
    return RegimeTransitionResult(
        daily_states=daily_states,
        transitions=transitions,
        champion_regime_metrics=metrics,
        regime_ranking=ranking,
        transition_impact=impact,
        selections=selections,
    )

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
    coverage_audit: pd.DataFrame


def _causal_reference(series: pd.Series, lookback: int) -> tuple[pd.Series, pd.Series]:
    numeric = pd.to_numeric(series, errors="coerce")
    prior = numeric.shift(1)
    minimum = max(4, lookback // 3)
    median = prior.rolling(lookback, min_periods=minimum).median()
    dispersion = prior.rolling(lookback, min_periods=minimum).std(ddof=0)
    return median, dispersion.replace(0.0, np.nan)


def build_coverage_audit(
    frame: pd.DataFrame,
    minimum_universe_size: int = 80,
    minimum_coverage_ratio: float = 0.70,
) -> pd.DataFrame:
    if not 0 < minimum_coverage_ratio <= 1:
        raise ValueError("minimum_coverage_ratio debe estar entre 0 y 1")
    if minimum_universe_size < 1:
        raise ValueError("minimum_universe_size debe ser positivo")

    data = frame[["origin_date", "ticker"]].copy()
    data["origin_date"] = pd.to_datetime(data["origin_date"])
    counts = (
        data.groupby("origin_date")["ticker"]
        .nunique()
        .rename("universe_size")
        .reset_index()
        .sort_values("origin_date")
        .reset_index(drop=True)
    )
    reference_universe_size = int(counts["universe_size"].max()) if not counts.empty else 0
    ratio_threshold = int(np.ceil(reference_universe_size * minimum_coverage_ratio))
    effective_threshold = max(minimum_universe_size, ratio_threshold)
    counts["reference_universe_size"] = reference_universe_size
    counts["coverage_ratio"] = np.where(
        reference_universe_size > 0,
        counts["universe_size"] / reference_universe_size,
        0.0,
    )
    counts["minimum_universe_size"] = minimum_universe_size
    counts["minimum_coverage_ratio"] = minimum_coverage_ratio
    counts["effective_universe_threshold"] = effective_threshold
    counts["is_synchronized_date"] = counts["universe_size"] >= effective_threshold
    counts["exclusion_reason"] = np.where(
        counts["is_synchronized_date"],
        "",
        "INSUFFICIENT_TRANSVERSAL_COVERAGE",
    )
    return counts


def _confirm_persistent_regimes(raw_labels: pd.Series, persistence: int) -> pd.Series:
    if persistence < 1:
        raise ValueError("minimum_regime_persistence debe ser al menos 1")
    labels = raw_labels.astype(str).tolist()
    if not labels:
        return pd.Series(dtype="object", index=raw_labels.index)

    confirmed: list[str] = []
    current = labels[0]
    candidate: str | None = None
    candidate_count = 0
    for raw in labels:
        if raw == current:
            candidate = None
            candidate_count = 0
        elif raw == candidate:
            candidate_count += 1
        else:
            candidate = raw
            candidate_count = 1

        if candidate is not None and candidate_count >= persistence:
            current = candidate
            candidate = None
            candidate_count = 0
        confirmed.append(current)
    return pd.Series(confirmed, index=raw_labels.index, dtype="object")


def build_daily_market_states(
    frame: pd.DataFrame,
    lookback: int = 12,
    minimum_universe_size: int = 80,
    minimum_coverage_ratio: float = 0.70,
    minimum_regime_persistence: int = 2,
) -> pd.DataFrame:
    required = {
        "origin_date", "ticker", "return_1d_pct", "return_5d_pct",
        "relative_volume", "atr_pct", "smart_money_pct",
        "smart_money_slope_5d", "qqq_return_1d_pct",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas para detectar régimen: {sorted(missing)}")
    if lookback < 6:
        raise ValueError("lookback debe ser al menos 6")

    coverage = build_coverage_audit(
        frame,
        minimum_universe_size=minimum_universe_size,
        minimum_coverage_ratio=minimum_coverage_ratio,
    )
    valid_dates = set(
        pd.to_datetime(
            coverage.loc[coverage["is_synchronized_date"], "origin_date"]
        )
    )
    if not valid_dates:
        threshold = int(coverage["effective_universe_threshold"].max()) if not coverage.empty else 0
        raise ValueError(
            "No existen fechas con cobertura transversal suficiente. "
            f"Umbral efectivo: {threshold} tickers."
        )

    data = frame.copy()
    data["origin_date"] = pd.to_datetime(data["origin_date"])
    data = data.loc[data["origin_date"].isin(valid_dates)].copy()
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
    daily = daily.merge(
        coverage[["origin_date", "reference_universe_size", "coverage_ratio"]],
        on="origin_date",
        how="left",
    )

    reference_columns = [
        "breadth_positive_1d", "breadth_positive_5d",
        "median_relative_volume", "median_atr_pct", "institutional_breadth",
    ]
    for column in reference_columns:
        median, dispersion = _causal_reference(daily[column], lookback)
        daily[f"reference_{column}"] = median
        daily[f"z_{column}"] = (
            pd.to_numeric(daily[column], errors="coerce") - median
        ) / dispersion

    raw_labels: list[str] = []
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
        raw_labels.append(regime)

    daily["raw_regime"] = raw_labels
    daily["regime"] = _confirm_persistent_regimes(
        daily["raw_regime"], minimum_regime_persistence
    )
    daily["regime_confirmation_delay"] = np.where(
        daily["raw_regime"] == daily["regime"], 0, 1
    )
    daily["previous_regime"] = daily["regime"].shift(1)
    daily["is_transition"] = daily["regime"] != daily["previous_regime"]
    daily.loc[daily.index[0], "is_transition"] = False

    normalized_changes = []
    for column in reference_columns:
        _, dispersion = _causal_reference(daily[column], lookback)
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
        rows.append({
            "transition_date": row["origin_date"],
            "from_regime": row["previous_regime"],
            "to_regime": row["regime"],
            "raw_regime": row.get("raw_regime"),
            "prior_regime_origin_dates": index - previous_transition_index,
            "transition_strength": row["transition_strength"],
            "universe_size": row["universe_size"],
            "coverage_ratio": row.get("coverage_ratio"),
            "qqq_return_1d_pct": row["qqq_return_1d_pct"],
            "breadth_positive_1d": row["breadth_positive_1d"],
            "breadth_positive_5d": row["breadth_positive_5d"],
            "median_relative_volume": row["median_relative_volume"],
            "median_atr_pct": row["median_atr_pct"],
            "institutional_breadth": row["institutional_breadth"],
        })
        previous_transition_index = index
    return pd.DataFrame(rows)


def _safe_float(value: object, default: float = 0.0) -> float:
    return float(value) if pd.notna(value) else default


def _quality(summary: pd.Series) -> float:
    rare = _safe_float(summary.get("rare_event_f1"), 0.0)
    balanced = _safe_float(summary.get("balanced_accuracy"), 0.0)
    macro = _safe_float(summary.get("macro_f1"), 0.0)
    brier = _safe_float(summary.get("mean_brier_skill"), 0.0)
    calibration = _safe_float(summary.get("mean_calibration_error"), 1.0)
    return 100.0 * (
        0.35 * rare + 0.25 * balanced + 0.20 * macro
        + 0.10 * brier + 0.10 * (1.0 - calibration)
    )


def evaluate_champions_by_regime(
    selections: pd.DataFrame,
    daily_states: pd.DataFrame,
    minimum_observations: int = 10,
) -> pd.DataFrame:
    state_map = daily_states[["origin_date", "regime"]].copy()
    data = selections.copy()
    data["origin_date"] = pd.to_datetime(data["origin_date"])
    data = data.merge(state_map, on="origin_date", how="inner")
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
    daily_states: pd.DataFrame,
    radius: int = 3,
) -> pd.DataFrame:
    if transitions.empty:
        return pd.DataFrame()
    dates = sorted(pd.to_datetime(daily_states["origin_date"]).dropna().unique())
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
            sample_dates = pd.to_datetime(sample["origin_date"])
            pre = sample.loc[sample_dates.isin(pre_dates)]
            post = sample.loc[sample_dates.isin(post_dates)]
            if pre.empty or post.empty:
                continue
            pre_summary = evaluate_predictions(pre, ticker=str(champion)).summary.iloc[0]
            post_summary = evaluate_predictions(post, ticker=str(champion)).summary.iloc[0]
            rows.append({
                "transition_date": date,
                "from_regime": transition["from_regime"],
                "to_regime": transition["to_regime"],
                "champion": champion,
                "pre_origin_dates": len(pre_dates),
                "post_origin_dates": len(post_dates),
                "pre_observations": len(pre),
                "post_observations": len(post),
                "pre_quality": _quality(pre_summary),
                "post_quality": _quality(post_summary),
                "delta_quality": _quality(post_summary) - _quality(pre_summary),
                "delta_rare_event_f1": _safe_float(post_summary.get("rare_event_f1")) - _safe_float(pre_summary.get("rare_event_f1")),
                "delta_balanced_accuracy": _safe_float(post_summary.get("balanced_accuracy")) - _safe_float(pre_summary.get("balanced_accuracy")),
                "delta_mean_brier_skill": _safe_float(post_summary.get("mean_brier_skill")) - _safe_float(pre_summary.get("mean_brier_skill")),
                "delta_calibration_error": _safe_float(post_summary.get("mean_calibration_error"), 1.0) - _safe_float(pre_summary.get("mean_calibration_error"), 1.0),
            })
    return pd.DataFrame(rows)


def run_regime_transition_laboratory(
    prediction_features: pd.DataFrame,
    lookback: int = 12,
    minimum_observations: int = 10,
    transition_radius: int = 3,
    minimum_universe_size: int = 80,
    minimum_coverage_ratio: float = 0.70,
    minimum_regime_persistence: int = 2,
) -> RegimeTransitionResult:
    scored = add_champion_scores(prediction_features)
    coverage = build_coverage_audit(
        scored,
        minimum_universe_size=minimum_universe_size,
        minimum_coverage_ratio=minimum_coverage_ratio,
    )
    daily_states = build_daily_market_states(
        scored,
        lookback=lookback,
        minimum_universe_size=minimum_universe_size,
        minimum_coverage_ratio=minimum_coverage_ratio,
        minimum_regime_persistence=minimum_regime_persistence,
    )
    valid_dates = set(pd.to_datetime(daily_states["origin_date"]))
    synchronized_scored = scored.loc[
        pd.to_datetime(scored["origin_date"]).isin(valid_dates)
    ].copy()
    selections = build_champion_selections(synchronized_scored)
    transitions = build_transitions(daily_states)
    metrics = evaluate_champions_by_regime(
        selections, daily_states, minimum_observations=minimum_observations
    )
    ranking = build_regime_ranking(metrics)
    impact = build_transition_impact(
        selections, transitions, daily_states, radius=transition_radius
    )
    return RegimeTransitionResult(
        daily_states=daily_states,
        transitions=transitions,
        champion_regime_metrics=metrics,
        regime_ranking=ranking,
        transition_impact=impact,
        selections=selections,
        coverage_audit=coverage,
    )

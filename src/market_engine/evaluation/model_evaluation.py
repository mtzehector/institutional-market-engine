from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


@dataclass(frozen=True)
class EvaluationResult:
    summary: pd.DataFrame
    confidence_bands: pd.DataFrame
    recent_performance: pd.DataFrame


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0 or pd.isna(denominator):
        return np.nan
    return float(numerator / denominator)


def _expected_calibration_error(
    probabilities: pd.Series,
    targets: pd.Series,
    bins: int = 10,
) -> float:
    frame = pd.DataFrame(
        {"probability": probabilities.astype(float), "target": targets.astype(int)}
    ).dropna()
    if frame.empty:
        return np.nan

    edges = np.linspace(0.0, 1.0, bins + 1)
    frame["bin"] = pd.cut(
        frame["probability"],
        bins=edges,
        include_lowest=True,
        duplicates="drop",
    )
    grouped = frame.groupby("bin", observed=True).agg(
        observations=("target", "size"),
        predicted=("probability", "mean"),
        observed=("target", "mean"),
    )
    weights = grouped["observations"] / grouped["observations"].sum()
    return float((weights * (grouped["predicted"] - grouped["observed"]).abs()).sum())


def _confidence_weighted_accuracy(predictions: pd.DataFrame) -> float:
    confidence = predictions[
        ["probability_up", "probability_down", "probability_no_gap"]
    ].max(axis=1)
    correct = predictions["correct_direction"].astype(float)
    if confidence.sum() == 0:
        return np.nan
    return float((confidence * correct).sum() / confidence.sum())


def _high_confidence_accuracy(
    predictions: pd.DataFrame,
    minimum_confidence: float,
) -> tuple[float, int]:
    confidence = predictions[
        ["probability_up", "probability_down", "probability_no_gap"]
    ].max(axis=1)
    selected = predictions.loc[confidence >= minimum_confidence]
    if selected.empty:
        return np.nan, 0
    return float(selected["correct_direction"].mean()), int(len(selected))


def _brier_skill(target: pd.Series, probability: pd.Series) -> tuple[float, float, float]:
    y_true = target.astype(int)
    y_probability = probability.astype(float)
    model_brier = float(brier_score_loss(y_true, y_probability))
    base_rate = float(y_true.mean())
    baseline_probability = np.full(len(y_true), base_rate)
    baseline_brier = float(brier_score_loss(y_true, baseline_probability))
    skill = 1.0 - model_brier / baseline_brier if baseline_brier > 0 else np.nan
    return model_brier, baseline_brier, float(skill) if pd.notna(skill) else np.nan


def _score_component(value: float, lower: float, upper: float) -> float:
    if pd.isna(value):
        return 0.0
    if upper <= lower:
        raise ValueError("upper debe ser mayor que lower")
    return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))


def _predictability_grade(score: float, observations: int) -> str:
    if observations < 40:
        return "MUESTRA_INSUFICIENTE"
    if score >= 80:
        return "A_EXCELENTE"
    if score >= 65:
        return "B_CONFIABLE"
    if score >= 50:
        return "C_EN_OBSERVACION"
    if score >= 35:
        return "D_DEBIL"
    return "E_NO_CONFIABLE"


def _confidence_band_table(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["dominant_probability"] = frame[
        ["probability_up", "probability_down", "probability_no_gap"]
    ].max(axis=1)
    frame["confidence_band"] = pd.cut(
        frame["dominant_probability"],
        bins=[0.0, 0.40, 0.50, 0.60, 0.70, 0.80, 1.0],
        labels=["<=40%", "40-50%", "50-60%", "60-70%", "70-80%", ">80%"],
        include_lowest=True,
    )
    result = (
        frame.groupby("confidence_band", observed=True)
        .agg(
            observations=("correct_direction", "size"),
            accuracy=("correct_direction", "mean"),
            average_confidence=("dominant_probability", "mean"),
            average_abs_gap=("actual_gap_pct", lambda values: values.abs().mean()),
        )
        .reset_index()
    )
    result["calibration_difference"] = (
        result["average_confidence"] - result["accuracy"]
    )
    return result


def _recent_windows(predictions: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ordered = predictions.sort_values("target_date")
    for window in windows:
        sample = ordered.tail(window)
        if sample.empty:
            continue
        rows.append(
            {
                "window_sessions": min(window, len(sample)),
                "from_date": sample.iloc[0]["target_date"],
                "to_date": sample.iloc[-1]["target_date"],
                "directional_accuracy": float(sample["correct_direction"].mean()),
                "confidence_weighted_accuracy": _confidence_weighted_accuracy(sample),
                "gap_up_accuracy": float(
                    (
                        sample.loc[sample["actual_direction"] == "GAP_UP", "predicted_direction"]
                        == "GAP_UP"
                    ).mean()
                )
                if (sample["actual_direction"] == "GAP_UP").any()
                else np.nan,
                "gap_down_accuracy": float(
                    (
                        sample.loc[
                            sample["actual_direction"] == "GAP_DOWN", "predicted_direction"
                        ]
                        == "GAP_DOWN"
                    ).mean()
                )
                if (sample["actual_direction"] == "GAP_DOWN").any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def evaluate_predictions(
    predictions: pd.DataFrame,
    ticker: str,
    high_confidence_threshold: float = 0.60,
) -> EvaluationResult:
    required = {
        "target_date",
        "actual_gap_pct",
        "actual_gap_up",
        "actual_gap_down",
        "actual_direction",
        "probability_up",
        "probability_down",
        "probability_no_gap",
        "predicted_direction",
        "correct_direction",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Faltan columnas para evaluar: {sorted(missing)}")
    if predictions.empty:
        raise ValueError("No hay predicciones para evaluar")

    frame = predictions.sort_values("target_date").reset_index(drop=True).copy()
    observations = len(frame)
    directional_accuracy = float(frame["correct_direction"].mean())
    confidence_weighted_accuracy = _confidence_weighted_accuracy(frame)
    high_conf_accuracy, high_conf_observations = _high_confidence_accuracy(
        frame, high_confidence_threshold
    )

    up_brier, up_baseline_brier, up_skill = _brier_skill(
        frame["actual_gap_up"], frame["probability_up"]
    )
    down_brier, down_baseline_brier, down_skill = _brier_skill(
        frame["actual_gap_down"], frame["probability_down"]
    )
    mean_brier_skill = float(np.nanmean([up_skill, down_skill]))

    ece_up = _expected_calibration_error(frame["probability_up"], frame["actual_gap_up"])
    ece_down = _expected_calibration_error(
        frame["probability_down"], frame["actual_gap_down"]
    )
    mean_ece = float(np.nanmean([ece_up, ece_down]))

    up_actual = frame["actual_direction"] == "GAP_UP"
    down_actual = frame["actual_direction"] == "GAP_DOWN"
    up_recall = float((frame.loc[up_actual, "predicted_direction"] == "GAP_UP").mean()) if up_actual.any() else np.nan
    down_recall = float((frame.loc[down_actual, "predicted_direction"] == "GAP_DOWN").mean()) if down_actual.any() else np.nan

    predicted_up = frame["predicted_direction"] == "GAP_UP"
    predicted_down = frame["predicted_direction"] == "GAP_DOWN"
    up_precision = float((frame.loc[predicted_up, "actual_direction"] == "GAP_UP").mean()) if predicted_up.any() else np.nan
    down_precision = float((frame.loc[predicted_down, "actual_direction"] == "GAP_DOWN").mean()) if predicted_down.any() else np.nan

    sample_reliability = min(1.0, observations / 250.0)
    score = 100.0 * (
        0.35 * _score_component(directional_accuracy, 0.33, 0.70)
        + 0.20 * _score_component(confidence_weighted_accuracy, 0.33, 0.75)
        + 0.20 * _score_component(mean_brier_skill, 0.0, 0.35)
        + 0.15 * _score_component(1.0 - mean_ece, 0.80, 0.98)
        + 0.10 * sample_reliability
    )

    if high_conf_observations >= 10 and pd.notna(high_conf_accuracy):
        score = 0.90 * score + 10.0 * _score_component(high_conf_accuracy, 0.40, 0.80)

    score = float(np.clip(score, 0.0, 100.0))
    summary = pd.DataFrame(
        [
            {
                "ticker": ticker.upper(),
                "from_date": frame.iloc[0]["target_date"],
                "to_date": frame.iloc[-1]["target_date"],
                "observations": observations,
                "predictability_score": score,
                "predictability_grade": _predictability_grade(score, observations),
                "directional_accuracy": directional_accuracy,
                "confidence_weighted_accuracy": confidence_weighted_accuracy,
                "high_confidence_threshold": high_confidence_threshold,
                "high_confidence_accuracy": high_conf_accuracy,
                "high_confidence_observations": high_conf_observations,
                "gap_up_cases": int(up_actual.sum()),
                "gap_down_cases": int(down_actual.sum()),
                "gap_up_precision": up_precision,
                "gap_up_recall": up_recall,
                "gap_down_precision": down_precision,
                "gap_down_recall": down_recall,
                "brier_up": up_brier,
                "baseline_brier_up": up_baseline_brier,
                "brier_skill_up": up_skill,
                "brier_down": down_brier,
                "baseline_brier_down": down_baseline_brier,
                "brier_skill_down": down_skill,
                "mean_brier_skill": mean_brier_skill,
                "calibration_error_up": ece_up,
                "calibration_error_down": ece_down,
                "mean_calibration_error": mean_ece,
                "sample_reliability": sample_reliability,
                "average_absolute_gap_pct": float(frame["actual_gap_pct"].abs().mean()),
            }
        ]
    )

    return EvaluationResult(
        summary=summary,
        confidence_bands=_confidence_band_table(frame),
        recent_performance=_recent_windows(frame, (20, 60, 120, 250)),
    )


def rank_ticker_evaluations(summaries: pd.DataFrame) -> pd.DataFrame:
    if summaries.empty:
        return summaries.copy()
    required = {"ticker", "predictability_score", "observations"}
    missing = required - set(summaries.columns)
    if missing:
        raise ValueError(f"Faltan columnas para ranking: {sorted(missing)}")

    ranking = summaries.sort_values(
        ["predictability_score", "observations", "directional_accuracy"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking

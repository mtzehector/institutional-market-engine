from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


CLASSES = ("GAP_UP", "GAP_DOWN", "SIN_GAP")


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
        frame["probability"], bins=edges, include_lowest=True, duplicates="drop"
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


def _classification_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    actual = frame["actual_direction"]
    predicted = frame["predicted_direction"]
    recalls: list[float] = []
    f1_values: list[float] = []
    result: dict[str, float | int] = {}

    for label, prefix in (("GAP_UP", "gap_up"), ("GAP_DOWN", "gap_down"), ("SIN_GAP", "no_gap")):
        actual_mask = actual == label
        predicted_mask = predicted == label
        tp = int((actual_mask & predicted_mask).sum())
        actual_count = int(actual_mask.sum())
        predicted_count = int(predicted_mask.sum())
        precision = _safe_divide(tp, predicted_count)
        recall = _safe_divide(tp, actual_count)
        f1 = (
            _safe_divide(2.0 * precision * recall, precision + recall)
            if pd.notna(precision) and pd.notna(recall)
            else np.nan
        )
        result[f"actual_{prefix}_count"] = actual_count
        result[f"predicted_{prefix}_count"] = predicted_count
        result[f"{prefix}_precision"] = precision
        result[f"{prefix}_recall"] = recall
        result[f"{prefix}_f1"] = f1
        if pd.notna(recall):
            recalls.append(float(recall))
        if pd.notna(f1):
            f1_values.append(float(f1))

    result["balanced_accuracy"] = float(np.mean(recalls)) if recalls else np.nan
    result["macro_f1"] = float(np.mean(f1_values)) if f1_values else np.nan

    no_gap_baseline_accuracy = float((actual == "SIN_GAP").mean())
    directional_accuracy = float((actual == predicted).mean())
    result["no_gap_baseline_accuracy"] = no_gap_baseline_accuracy
    result["incremental_accuracy_vs_no_gap"] = directional_accuracy - no_gap_baseline_accuracy

    actual_rare = actual != "SIN_GAP"
    predicted_rare = predicted != "SIN_GAP"
    rare_tp = int((actual_rare & predicted_rare).sum())
    rare_precision = _safe_divide(rare_tp, int(predicted_rare.sum()))
    rare_recall = _safe_divide(rare_tp, int(actual_rare.sum()))
    rare_f1 = (
        _safe_divide(2.0 * rare_precision * rare_recall, rare_precision + rare_recall)
        if pd.notna(rare_precision) and pd.notna(rare_recall)
        else np.nan
    )
    result["actual_rare_event_count"] = int(actual_rare.sum())
    result["predicted_rare_event_count"] = int(predicted_rare.sum())
    result["rare_event_precision"] = rare_precision
    result["rare_event_recall"] = rare_recall
    result["rare_event_f1"] = rare_f1
    return result


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
            actual_rare_events=("actual_direction", lambda values: (values != "SIN_GAP").sum()),
            predicted_rare_events=("predicted_direction", lambda values: (values != "SIN_GAP").sum()),
        )
        .reset_index()
    )
    result["calibration_difference"] = result["average_confidence"] - result["accuracy"]
    return result


def _recent_windows(predictions: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ordered = predictions.sort_values("target_date")
    for window in windows:
        sample = ordered.tail(window)
        if sample.empty:
            continue
        metrics = _classification_metrics(sample)
        rows.append(
            {
                "window_sessions": min(window, len(sample)),
                "from_date": sample.iloc[0]["target_date"],
                "to_date": sample.iloc[-1]["target_date"],
                "directional_accuracy": float(sample["correct_direction"].mean()),
                "confidence_weighted_accuracy": _confidence_weighted_accuracy(sample),
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "rare_event_precision": metrics["rare_event_precision"],
                "rare_event_recall": metrics["rare_event_recall"],
                "rare_event_f1": metrics["rare_event_f1"],
                "no_gap_baseline_accuracy": metrics["no_gap_baseline_accuracy"],
                "incremental_accuracy_vs_no_gap": metrics["incremental_accuracy_vs_no_gap"],
            }
        )
    return pd.DataFrame(rows)


def evaluate_predictions(
    predictions: pd.DataFrame,
    ticker: str,
    high_confidence_threshold: float = 0.60,
) -> EvaluationResult:
    required = {
        "target_date", "actual_gap_pct", "actual_gap_up", "actual_gap_down",
        "actual_direction", "probability_up", "probability_down", "probability_no_gap",
        "predicted_direction", "correct_direction",
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
    classification = _classification_metrics(frame)

    up_brier, up_baseline_brier, up_skill = _brier_skill(
        frame["actual_gap_up"], frame["probability_up"]
    )
    down_brier, down_baseline_brier, down_skill = _brier_skill(
        frame["actual_gap_down"], frame["probability_down"]
    )
    valid_skills = [value for value in (up_skill, down_skill) if pd.notna(value)]
    mean_brier_skill = float(np.mean(valid_skills)) if valid_skills else np.nan

    ece_up = _expected_calibration_error(frame["probability_up"], frame["actual_gap_up"])
    ece_down = _expected_calibration_error(frame["probability_down"], frame["actual_gap_down"])
    mean_ece = float(np.nanmean([ece_up, ece_down]))
    sample_reliability = min(1.0, observations / 250.0)

    balanced_component = _score_component(float(classification["balanced_accuracy"]), 1.0 / 3.0, 1.0)
    macro_f1_component = _score_component(float(classification["macro_f1"]), 0.0, 1.0)
    rare_f1_component = _score_component(float(classification["rare_event_f1"]), 0.0, 1.0)
    incremental_component = _score_component(
        float(classification["incremental_accuracy_vs_no_gap"]), 0.0, 0.25
    )
    brier_component = _score_component(mean_brier_skill, 0.0, 0.35)
    calibration_component = _score_component(1.0 - mean_ece, 0.80, 0.98)

    raw_score = 100.0 * (
        0.25 * balanced_component
        + 0.20 * macro_f1_component
        + 0.20 * rare_f1_component
        + 0.10 * incremental_component
        + 0.10 * brier_component
        + 0.10 * calibration_component
        + 0.05 * sample_reliability
    )

    rare_events = int(classification["actual_rare_event_count"])
    rare_sample_reliability = min(1.0, rare_events / 20.0)
    rare_event_gate = 0.60 + 0.40 * rare_sample_reliability
    score = float(np.clip(raw_score * rare_event_gate, 0.0, 100.0))

    summary_values: dict[str, Any] = {
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
        **classification,
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
        "rare_sample_reliability": rare_sample_reliability,
        "balanced_component": balanced_component,
        "macro_f1_component": macro_f1_component,
        "rare_f1_component": rare_f1_component,
        "incremental_component": incremental_component,
        "brier_component": brier_component,
        "calibration_component": calibration_component,
        "rare_event_gate": rare_event_gate,
        "raw_predictability_score": raw_score,
        "average_absolute_gap_pct": float(frame["actual_gap_pct"].abs().mean()),
    }
    return EvaluationResult(
        summary=pd.DataFrame([summary_values]),
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

    sort_columns = ["predictability_score", "observations"]
    if "rare_event_f1" in summaries.columns:
        sort_columns.append("rare_event_f1")
    else:
        sort_columns.append("directional_accuracy")
    ranking = summaries.sort_values(sort_columns, ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, precision_score, recall_score, roc_auc_score

from market_engine.gaps.predictor import predict_next_gap


@dataclass(frozen=True)
class WalkForwardResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    calibration: pd.DataFrame


def _safe_metric(function: Any, *args: Any, **kwargs: Any) -> float:
    try:
        return float(function(*args, **kwargs))
    except Exception:
        return np.nan


def _direction(probability_up: float, probability_down: float, threshold: float) -> str:
    if probability_up >= threshold and probability_up > probability_down:
        return "GAP_UP"
    if probability_down >= threshold and probability_down > probability_up:
        return "GAP_DOWN"
    return "SIN_GAP"


def _actual_direction(up: int, down: int) -> str:
    if up == 1:
        return "GAP_UP"
    if down == 1:
        return "GAP_DOWN"
    return "SIN_GAP"


def _calibration_table(
    probabilities: pd.Series,
    targets: pd.Series,
    label: str,
    bins: int = 10,
) -> pd.DataFrame:
    frame = pd.DataFrame({"probability": probabilities, "target": targets}).dropna()
    if frame.empty:
        return pd.DataFrame()

    edges = np.linspace(0.0, 1.0, bins + 1)
    frame["bin"] = pd.cut(
        frame["probability"],
        bins=edges,
        include_lowest=True,
        duplicates="drop",
    )
    result = (
        frame.groupby("bin", observed=True)
        .agg(
            observations=("target", "size"),
            mean_predicted_probability=("probability", "mean"),
            observed_frequency=("target", "mean"),
        )
        .reset_index()
    )
    result.insert(0, "target", label)
    result["absolute_calibration_error"] = (
        result["mean_predicted_probability"] - result["observed_frequency"]
    ).abs()
    return result


def _metrics(predictions: pd.DataFrame, decision_threshold: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for label, probability_column, target_column in [
        ("GAP_UP", "probability_up", "actual_gap_up"),
        ("GAP_DOWN", "probability_down", "actual_gap_down"),
    ]:
        y_true = predictions[target_column].astype(int)
        y_probability = predictions[probability_column].astype(float)
        y_predicted = (y_probability >= decision_threshold).astype(int)

        rows.append(
            {
                "target": label,
                "observations": len(predictions),
                "positive_cases": int(y_true.sum()),
                "base_rate": float(y_true.mean()),
                "mean_probability": float(y_probability.mean()),
                "brier_score": _safe_metric(brier_score_loss, y_true, y_probability),
                "roc_auc": (
                    _safe_metric(roc_auc_score, y_true, y_probability)
                    if y_true.nunique() == 2
                    else np.nan
                ),
                "precision_at_threshold": _safe_metric(
                    precision_score, y_true, y_predicted, zero_division=0
                ),
                "recall_at_threshold": _safe_metric(
                    recall_score, y_true, y_predicted, zero_division=0
                ),
                "decision_threshold": decision_threshold,
            }
        )

    predicted_direction = predictions.apply(
        lambda row: _direction(
            float(row["probability_up"]),
            float(row["probability_down"]),
            decision_threshold,
        ),
        axis=1,
    )
    actual_direction = predictions.apply(
        lambda row: _actual_direction(
            int(row["actual_gap_up"]), int(row["actual_gap_down"])
        ),
        axis=1,
    )

    rows.append(
        {
            "target": "DIRECCION_3_CLASES",
            "observations": len(predictions),
            "positive_cases": np.nan,
            "base_rate": np.nan,
            "mean_probability": np.nan,
            "brier_score": np.nan,
            "roc_auc": np.nan,
            "precision_at_threshold": float((predicted_direction == actual_direction).mean()),
            "recall_at_threshold": np.nan,
            "decision_threshold": decision_threshold,
        }
    )

    return pd.DataFrame(rows)


def walk_forward_gap(
    features: pd.DataFrame,
    from_date: date,
    to_date: date,
    min_history_rows: int = 180,
    step: int = 1,
    decision_threshold: float = 0.50,
    max_history_rows: int | None = None,
) -> WalkForwardResult:
    """Run walk-forward predictions without future leakage.

    By default the training window expands from the beginning of the available
    history. When ``max_history_rows`` is supplied, only the most recent N rows
    available at each origin are used. This makes it possible to compare the
    useful memory or half-life of each ticker without changing the prediction
    model itself.
    """
    if step < 1:
        raise ValueError("step debe ser mayor o igual que 1")
    if not 0 < decision_threshold < 1:
        raise ValueError("decision_threshold debe estar entre 0 y 1")
    if max_history_rows is not None and max_history_rows < min_history_rows:
        raise ValueError("max_history_rows no puede ser menor que min_history_rows")

    ordered = features.sort_values("date").reset_index(drop=True).copy()
    records: list[dict[str, Any]] = []

    for origin_index in range(min_history_rows, len(ordered) - 1, step):
        origin = ordered.iloc[origin_index]
        target = ordered.iloc[origin_index + 1]
        target_date = pd.Timestamp(target["date"]).date()

        if target_date < from_date or target_date > to_date:
            continue
        if pd.isna(origin.get("target_gap_up_next")) or pd.isna(
            origin.get("target_gap_down_next")
        ):
            continue

        start_index = 0
        if max_history_rows is not None:
            start_index = max(0, origin_index + 1 - max_history_rows)
        historical_slice = ordered.iloc[start_index : origin_index + 1].copy()

        try:
            prediction = predict_next_gap(historical_slice)
        except ValueError:
            continue

        actual_up = int(origin["target_gap_up_next"])
        actual_down = int(origin["target_gap_down_next"])
        probability_up = float(prediction.probability_up)
        probability_down = float(prediction.probability_down)

        records.append(
            {
                "origin_date": pd.Timestamp(origin["date"]),
                "target_date": pd.Timestamp(target["date"]),
                "gap_threshold_pct": float(origin["gap_threshold_pct"]),
                "actual_gap_pct": float(origin["gap_next_pct"]),
                "actual_gap_up": actual_up,
                "actual_gap_down": actual_down,
                "actual_direction": _actual_direction(actual_up, actual_down),
                "probability_up": probability_up,
                "probability_down": probability_down,
                "probability_no_gap": float(prediction.probability_no_gap),
                "predicted_direction": _direction(
                    probability_up, probability_down, decision_threshold
                ),
                "correct_direction": _direction(
                    probability_up, probability_down, decision_threshold
                )
                == _actual_direction(actual_up, actual_down),
                "base_up": float(prediction.base_up),
                "base_down": float(prediction.base_down),
                "lift_up": float(prediction.lift_up),
                "lift_down": float(prediction.lift_down),
                "training_rows": int(prediction.training_rows),
                "memory_rows_requested": max_history_rows,
                "memory_rows_available": int(len(historical_slice)),
                "brier_up_internal": float(prediction.brier_up),
                "brier_down_internal": float(prediction.brier_down),
                "roc_auc_up_internal": float(prediction.roc_auc_up),
                "roc_auc_down_internal": float(prediction.roc_auc_down),
                "mode_up": prediction.mode_up,
                "mode_down": prediction.mode_down,
            }
        )

    predictions = pd.DataFrame(records)
    if predictions.empty:
        raise ValueError(
            "No se generaron predicciones. Amplía el histórico o revisa las fechas."
        )

    metrics = _metrics(predictions, decision_threshold)
    calibration = pd.concat(
        [
            _calibration_table(
                predictions["probability_up"], predictions["actual_gap_up"], "GAP_UP"
            ),
            _calibration_table(
                predictions["probability_down"],
                predictions["actual_gap_down"],
                "GAP_DOWN",
            ),
        ],
        ignore_index=True,
    )

    return WalkForwardResult(
        predictions=predictions,
        metrics=metrics,
        calibration=calibration,
    )

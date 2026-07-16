from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd

from market_engine.backtesting.walk_forward import walk_forward_gap
from market_engine.evaluation.model_evaluation import evaluate_predictions


@dataclass(frozen=True)
class AdaptiveMemoryResult:
    ranking: pd.DataFrame
    best_by_ticker: pd.DataFrame
    predictions: pd.DataFrame


def _memory_label(rows: int | None) -> str:
    return "EXPANDING" if rows is None else f"{rows}_ROWS"


def evaluate_memory_windows(
    features: pd.DataFrame,
    ticker: str,
    from_date: date,
    to_date: date,
    memory_windows: Iterable[int | None] = (180, 252, 378, 504, 756, None),
    min_history_rows: int = 180,
    step: int = 5,
    decision_threshold: float = 0.50,
    high_confidence_threshold: float = 0.60,
    include_predictions: bool = False,
) -> AdaptiveMemoryResult:
    rows: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []

    normalized_windows: list[int | None] = []
    seen: set[str] = set()
    for window in memory_windows:
        if window is not None and window < min_history_rows:
            continue
        key = _memory_label(window)
        if key not in seen:
            seen.add(key)
            normalized_windows.append(window)

    if not normalized_windows:
        raise ValueError("No hay ventanas de memoria válidas para evaluar")

    for window in normalized_windows:
        result = walk_forward_gap(
            features,
            from_date=from_date,
            to_date=to_date,
            min_history_rows=min_history_rows,
            step=step,
            decision_threshold=decision_threshold,
            max_history_rows=window,
        )
        evaluation = evaluate_predictions(
            result.predictions,
            ticker=ticker,
            high_confidence_threshold=high_confidence_threshold,
        )
        summary = evaluation.summary.copy()
        summary["memory_rows"] = window
        summary["memory_label"] = _memory_label(window)
        summary["evaluation_from"] = pd.Timestamp(from_date)
        summary["evaluation_to"] = pd.Timestamp(to_date)
        rows.append(summary)

        if include_predictions:
            predictions = result.predictions.copy()
            predictions.insert(0, "ticker", ticker.upper())
            predictions.insert(1, "memory_label", _memory_label(window))
            predictions.insert(2, "memory_rows", window)
            prediction_frames.append(predictions)

    ranking = pd.concat(rows, ignore_index=True)
    ranking = ranking.sort_values(
        ["predictability_score", "observations", "directional_accuracy"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    ranking.insert(0, "memory_rank", np.arange(1, len(ranking) + 1))

    best = ranking.head(1).copy()
    best.insert(0, "selection", "BEST_MEMORY")

    predictions_frame = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )

    return AdaptiveMemoryResult(
        ranking=ranking,
        best_by_ticker=best,
        predictions=predictions_frame,
    )


def rank_best_memories(best_rows: pd.DataFrame) -> pd.DataFrame:
    if best_rows.empty:
        return best_rows.copy()
    required = {"ticker", "predictability_score", "memory_label", "observations"}
    missing = required - set(best_rows.columns)
    if missing:
        raise ValueError(f"Faltan columnas para ranking adaptativo: {sorted(missing)}")

    ranking = best_rows.sort_values(
        ["predictability_score", "observations", "directional_accuracy"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    ranking.insert(0, "adaptive_rank", np.arange(1, len(ranking) + 1))
    return ranking

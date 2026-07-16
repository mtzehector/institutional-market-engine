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
    errors: pd.DataFrame


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
    """Evaluate several rolling memories without letting one invalid window abort all.

    A finite ``memory_window`` represents effective training rows. The walk-forward
    slice also needs the current origin row, so one extra row is retained internally.
    This avoids the former 180 -> 179 off-by-one failure.
    """
    rows: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    error_rows: list[dict[str, object]] = []

    normalized_windows: list[int | None] = []
    seen: set[str] = set()
    for window in memory_windows:
        if window is not None and window < min_history_rows:
            error_rows.append(
                {
                    "ticker": ticker.upper(),
                    "memory_label": _memory_label(window),
                    "memory_rows": window,
                    "error": (
                        f"Ventana menor que min_history_rows={min_history_rows}; "
                        "se omitió."
                    ),
                }
            )
            continue
        key = _memory_label(window)
        if key not in seen:
            seen.add(key)
            normalized_windows.append(window)

    if not normalized_windows:
        raise ValueError("No hay ventanas de memoria válidas para evaluar")

    for window in normalized_windows:
        label = _memory_label(window)
        try:
            # `predict_next_gap` reserves the last row as the current observation;
            # keep one additional row so the requested number remains trainable.
            retained_rows = None if window is None else window + 1
            result = walk_forward_gap(
                features,
                from_date=from_date,
                to_date=to_date,
                min_history_rows=min_history_rows,
                step=step,
                decision_threshold=decision_threshold,
                max_history_rows=retained_rows,
            )
            evaluation = evaluate_predictions(
                result.predictions,
                ticker=ticker,
                high_confidence_threshold=high_confidence_threshold,
            )
            summary = evaluation.summary.copy()
            summary["memory_rows"] = window
            summary["memory_label"] = label
            summary["evaluation_from"] = pd.Timestamp(from_date)
            summary["evaluation_to"] = pd.Timestamp(to_date)
            rows.append(summary)

            if include_predictions:
                predictions = result.predictions.copy()
                predictions.insert(0, "ticker", ticker.upper())
                predictions.insert(1, "memory_label", label)
                predictions.insert(2, "memory_rows", window)
                prediction_frames.append(predictions)
        except Exception as exc:
            error_rows.append(
                {
                    "ticker": ticker.upper(),
                    "memory_label": label,
                    "memory_rows": window,
                    "error": str(exc),
                }
            )

    if not rows:
        detail = "; ".join(
            f"{row['memory_label']}: {row['error']}" for row in error_rows[:6]
        )
        raise ValueError(
            "Ninguna ventana produjo predicciones. "
            f"Histórico disponible={len(features)} filas. Detalle: {detail}"
        )

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
        errors=pd.DataFrame(error_rows),
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

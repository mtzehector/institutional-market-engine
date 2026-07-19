from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from market_engine.evaluation.market_morphology import build_market_geometry


@dataclass(frozen=True)
class SampleMorphologyFidelityResult:
    aligned_geometry: pd.DataFrame
    metrics: pd.DataFrame
    rolling_fidelity: pd.DataFrame
    summary: pd.DataFrame


def _correlation(left: pd.Series, right: pd.Series) -> float:
    pair = pd.concat([left, right], axis=1).dropna()
    if len(pair) < 2 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))


def _bounded_score(value: float) -> float:
    if pd.isna(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0) * 100.0)


def compare_sample_to_reference(
    sample: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    smoothing_window: int = 5,
    rolling_window: int = 20,
) -> SampleMorphologyFidelityResult:
    """Compara cuánta geometría de una referencia conserva una muestra.

    Ambas entradas usan las columnas ``date``, ``ticker`` y ``market_cap``.
    La comparación se realiza sobre índices normalizados, por lo que el tamaño
    absoluto distinto de los universos no invalida el experimento.
    """
    if rolling_window < 3:
        raise ValueError("rolling_window debe ser mayor o igual a 3")

    sample_geometry = build_market_geometry(sample, smoothing_window=smoothing_window)
    reference_geometry = build_market_geometry(reference, smoothing_window=smoothing_window)

    sample_columns = {
        column: f"sample_{column}"
        for column in sample_geometry.columns
        if column != "date"
    }
    reference_columns = {
        column: f"reference_{column}"
        for column in reference_geometry.columns
        if column != "date"
    }
    aligned = sample_geometry.rename(columns=sample_columns).merge(
        reference_geometry.rename(columns=reference_columns),
        on="date",
        how="inner",
        validate="one_to_one",
    )
    if len(aligned) < 3:
        raise ValueError("No existen suficientes fechas comunes entre muestra y referencia")

    aligned["index_error"] = (
        aligned["sample_market_cap_index"] - aligned["reference_market_cap_index"]
    )
    aligned["absolute_index_error"] = aligned["index_error"].abs()
    aligned["drawdown_error"] = (
        aligned["sample_drawdown_pct"] - aligned["reference_drawdown_pct"]
    )
    aligned["absolute_drawdown_error"] = aligned["drawdown_error"].abs()

    index_corr = _correlation(aligned["sample_market_cap_index"], aligned["reference_market_cap_index"])
    returns_corr = _correlation(aligned["sample_daily_return"], aligned["reference_daily_return"])
    slope_corr = _correlation(aligned["sample_smoothed_slope"], aligned["reference_smoothed_slope"])
    acceleration_corr = _correlation(aligned["sample_acceleration"], aligned["reference_acceleration"])
    drawdown_corr = _correlation(aligned["sample_drawdown_pct"], aligned["reference_drawdown_pct"])

    mae_index = float(aligned["absolute_index_error"].mean())
    mae_drawdown = float(aligned["absolute_drawdown_error"].mean())
    maximum_index_error = float(aligned["absolute_index_error"].max())
    directional_agreement = float(
        (
            np.sign(aligned["sample_daily_return"].fillna(0.0))
            == np.sign(aligned["reference_daily_return"].fillna(0.0))
        ).mean()
    )

    fidelity_score = float(np.clip(
        0.25 * _bounded_score((index_corr + 1.0) / 2.0)
        + 0.20 * _bounded_score((returns_corr + 1.0) / 2.0)
        + 0.15 * _bounded_score((slope_corr + 1.0) / 2.0)
        + 0.10 * _bounded_score((acceleration_corr + 1.0) / 2.0)
        + 0.15 * _bounded_score((drawdown_corr + 1.0) / 2.0)
        + 0.15 * directional_agreement * 100.0
        - min(mae_index, 25.0)
        - min(mae_drawdown, 20.0),
        0.0,
        100.0,
    ))

    metrics = pd.DataFrame([
        {"metric": "index_correlation", "value": index_corr},
        {"metric": "returns_correlation", "value": returns_corr},
        {"metric": "slope_correlation", "value": slope_corr},
        {"metric": "acceleration_correlation", "value": acceleration_corr},
        {"metric": "drawdown_correlation", "value": drawdown_corr},
        {"metric": "directional_agreement", "value": directional_agreement},
        {"metric": "mean_absolute_index_error", "value": mae_index},
        {"metric": "mean_absolute_drawdown_error", "value": mae_drawdown},
        {"metric": "maximum_absolute_index_error", "value": maximum_index_error},
        {"metric": "morphology_fidelity_score", "value": fidelity_score},
    ])

    rolling = aligned[["date"]].copy()
    rolling["rolling_returns_correlation"] = aligned["sample_daily_return"].rolling(
        rolling_window, min_periods=max(3, rolling_window // 2)
    ).corr(aligned["reference_daily_return"])
    rolling["rolling_drawdown_error"] = aligned["absolute_drawdown_error"].rolling(
        rolling_window, min_periods=1
    ).mean()
    rolling["rolling_index_error"] = aligned["absolute_index_error"].rolling(
        rolling_window, min_periods=1
    ).mean()

    latest = aligned.iloc[-1]
    summary = pd.DataFrame([{
        "start_date": aligned.iloc[0]["date"],
        "end_date": latest["date"],
        "common_observations": int(len(aligned)),
        "sample_companies_latest": int(latest["sample_observed_companies"]),
        "reference_companies_latest": int(latest["reference_observed_companies"]),
        "morphology_fidelity_score": fidelity_score,
        "index_correlation": index_corr,
        "returns_correlation": returns_corr,
        "drawdown_correlation": drawdown_corr,
        "directional_agreement": directional_agreement,
        "mean_absolute_index_error": mae_index,
        "mean_absolute_drawdown_error": mae_drawdown,
    }])

    return SampleMorphologyFidelityResult(
        aligned_geometry=aligned,
        metrics=metrics,
        rolling_fidelity=rolling,
        summary=summary,
    )

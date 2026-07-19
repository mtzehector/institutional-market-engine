from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketMorphologyResult:
    geometry: pd.DataFrame
    episodes: pd.DataFrame
    summary: pd.DataFrame


def _validate_input(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "ticker", "market_cap"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    data = frame.loc[:, ["date", "ticker", "market_cap"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["market_cap"] = pd.to_numeric(data["market_cap"], errors="coerce")
    data = data.dropna(subset=["date", "ticker", "market_cap"])
    data = data.loc[data["market_cap"] > 0]
    if data.empty:
        raise ValueError("No existen observaciones válidas de capitalización de mercado")
    return data.sort_values(["date", "ticker"]).reset_index(drop=True)


def build_market_geometry(
    frame: pd.DataFrame,
    *,
    smoothing_window: int = 5,
) -> pd.DataFrame:
    """Construye la geometría agregada sin atribuir causas ni narrativas.

    La entrada contiene una observación por fecha y ticker. La capitalización
    agregada se normaliza a 100 en la primera fecha para permitir comparar
    muestras de diferente tamaño.
    """
    if smoothing_window < 1:
        raise ValueError("smoothing_window debe ser mayor o igual a 1")

    data = _validate_input(frame)
    daily = (
        data.groupby("date", as_index=False)
        .agg(
            aggregate_market_cap=("market_cap", "sum"),
            observed_companies=("ticker", "nunique"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    base = float(daily.iloc[0]["aggregate_market_cap"])
    daily["market_cap_index"] = 100.0 * daily["aggregate_market_cap"] / base
    daily["daily_return"] = daily["aggregate_market_cap"].pct_change()
    daily["log_market_cap"] = np.log(daily["aggregate_market_cap"])
    daily["slope"] = daily["log_market_cap"].diff()
    daily["smoothed_slope"] = daily["slope"].rolling(
        smoothing_window, min_periods=1
    ).mean()
    daily["acceleration"] = daily["smoothed_slope"].diff()
    daily["running_peak_market_cap"] = daily["aggregate_market_cap"].cummax()
    daily["drawdown"] = (
        daily["aggregate_market_cap"] / daily["running_peak_market_cap"] - 1.0
    )
    daily["drawdown_pct"] = 100.0 * daily["drawdown"]
    daily["at_peak"] = np.isclose(daily["drawdown"], 0.0)
    daily["days_since_peak"] = (
        daily.groupby(daily["at_peak"].cumsum()).cumcount()
    )
    return daily


def detect_drawdown_episodes(
    geometry: pd.DataFrame,
    *,
    minimum_drawdown_pct: float = 10.0,
) -> pd.DataFrame:
    """Reconstruye episodios pico-caída-recuperación sin asignarles nombres."""
    if geometry.empty:
        return pd.DataFrame()
    if minimum_drawdown_pct <= 0:
        raise ValueError("minimum_drawdown_pct debe ser positivo")

    ordered = geometry.sort_values("date").reset_index(drop=True)
    rows: list[dict[str, object]] = []
    peak_position = 0
    trough_position = 0
    in_episode = False
    episode_number = 0

    for position, row in ordered.iterrows():
        drawdown_pct = float(row["drawdown_pct"])
        if not in_episode and drawdown_pct <= -minimum_drawdown_pct:
            in_episode = True
            episode_number += 1
            peak_position = int(ordered.loc[:position, "aggregate_market_cap"].idxmax())
            trough_position = position
        elif in_episode and row["aggregate_market_cap"] < ordered.iloc[trough_position]["aggregate_market_cap"]:
            trough_position = position

        if in_episode and bool(row["at_peak"]):
            peak = ordered.iloc[peak_position]
            trough = ordered.iloc[trough_position]
            rows.append(
                {
                    "episode_id": f"EPISODE_{episode_number:03d}",
                    "peak_date": peak["date"],
                    "trough_date": trough["date"],
                    "recovery_date": row["date"],
                    "maximum_drawdown_pct": float(trough["drawdown_pct"]),
                    "decline_observations": int(trough_position - peak_position),
                    "recovery_observations": int(position - trough_position),
                    "total_episode_observations": int(position - peak_position),
                    "episode_completed": True,
                }
            )
            in_episode = False

    if in_episode:
        peak = ordered.iloc[peak_position]
        trough = ordered.iloc[trough_position]
        rows.append(
            {
                "episode_id": f"EPISODE_{episode_number:03d}",
                "peak_date": peak["date"],
                "trough_date": trough["date"],
                "recovery_date": pd.NaT,
                "maximum_drawdown_pct": float(trough["drawdown_pct"]),
                "decline_observations": int(trough_position - peak_position),
                "recovery_observations": np.nan,
                "total_episode_observations": int(len(ordered) - 1 - peak_position),
                "episode_completed": False,
            }
        )

    return pd.DataFrame(rows)


def run_market_morphology_laboratory(
    frame: pd.DataFrame,
    *,
    smoothing_window: int = 5,
    minimum_drawdown_pct: float = 10.0,
) -> MarketMorphologyResult:
    geometry = build_market_geometry(frame, smoothing_window=smoothing_window)
    episodes = detect_drawdown_episodes(
        geometry,
        minimum_drawdown_pct=minimum_drawdown_pct,
    )

    latest = geometry.iloc[-1]
    summary = pd.DataFrame(
        [
            {
                "start_date": geometry.iloc[0]["date"],
                "end_date": latest["date"],
                "observations": int(len(geometry)),
                "latest_aggregate_market_cap": float(latest["aggregate_market_cap"]),
                "latest_market_cap_index": float(latest["market_cap_index"]),
                "latest_drawdown_pct": float(latest["drawdown_pct"]),
                "latest_smoothed_slope": float(latest["smoothed_slope"]),
                "latest_acceleration": float(latest["acceleration"])
                if pd.notna(latest["acceleration"])
                else np.nan,
                "detected_episodes": int(len(episodes)),
                "active_episode": bool(
                    not episodes.empty and not bool(episodes.iloc[-1]["episode_completed"])
                ),
            }
        ]
    )
    return MarketMorphologyResult(
        geometry=geometry,
        episodes=episodes,
        summary=summary,
    )

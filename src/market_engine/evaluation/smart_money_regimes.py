from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SmartMoneyRegimeResult:
    summary: pd.DataFrame
    regimes: pd.DataFrame
    crossings: pd.DataFrame


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0 or pd.isna(denominator):
        return np.nan
    return float(numerator / denominator)


def _initial_state(value: float, lower: float, upper: float) -> str:
    if pd.isna(value):
        return "UNKNOWN"
    if value < lower:
        return "RETAIL_DOMINANT"
    if value > upper:
        return "SMART_MONEY_DOMINANT"
    return "EQUILIBRIUM"


def _confirmed_states(
    smart_money: pd.Series,
    lower: float,
    upper: float,
    minimum_persistence: int,
) -> pd.Series:
    raw = smart_money.map(lambda value: _initial_state(value, lower, upper))
    confirmed = pd.Series(index=raw.index, dtype="object")
    current = "UNKNOWN"
    candidate = "UNKNOWN"
    candidate_count = 0

    for index, state in raw.items():
        if state in {"UNKNOWN", "EQUILIBRIUM"}:
            confirmed.loc[index] = current if current != "UNKNOWN" else state
            continue

        if current == "UNKNOWN":
            if state == candidate:
                candidate_count += 1
            else:
                candidate = state
                candidate_count = 1
            if candidate_count >= minimum_persistence:
                current = state
            confirmed.loc[index] = current if current != "UNKNOWN" else "EQUILIBRIUM"
            continue

        if state == current:
            candidate = "UNKNOWN"
            candidate_count = 0
            confirmed.loc[index] = current
            continue

        if state == candidate:
            candidate_count += 1
        else:
            candidate = state
            candidate_count = 1

        if candidate_count >= minimum_persistence:
            current = candidate
            candidate = "UNKNOWN"
            candidate_count = 0
        confirmed.loc[index] = current

    return confirmed


def _regime_rows(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    data = frame.loc[frame["confirmed_regime"].isin(["RETAIL_DOMINANT", "SMART_MONEY_DOMINANT"])].copy()
    if data.empty:
        return pd.DataFrame()

    data["regime_group"] = data["confirmed_regime"].ne(data["confirmed_regime"].shift()).cumsum()
    rows: list[dict[str, object]] = []
    for regime_id, sample in data.groupby("regime_group", sort=True):
        duration = int(len(sample))
        median_relative_volume = float(sample["relative_volume"].median())
        mean_relative_volume = float(sample["relative_volume"].mean())
        persistence_factor = min(1.0, duration / 10.0)
        strength = float(np.log1p(duration) * median_relative_volume * persistence_factor)
        rows.append(
            {
                "ticker": ticker.upper(),
                "regime_id": int(regime_id),
                "regime": sample.iloc[0]["confirmed_regime"],
                "start_date": sample.iloc[0]["date"],
                "end_date": sample.iloc[-1]["date"],
                "duration_sessions": duration,
                "median_smart_money_pct": float(sample["smart_money_pct"].median()),
                "mean_smart_money_pct": float(sample["smart_money_pct"].mean()),
                "median_relative_volume": median_relative_volume,
                "mean_relative_volume": mean_relative_volume,
                "persistence_factor": persistence_factor,
                "institutional_regime_strength": strength,
            }
        )
    return pd.DataFrame(rows)


def _crossing_rows(
    frame: pd.DataFrame,
    regimes: pd.DataFrame,
    ticker: str,
    lookaround: int,
    false_regime_days: int,
) -> pd.DataFrame:
    if regimes.empty or len(regimes) < 2:
        return pd.DataFrame()

    data = frame.reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for position in range(1, len(regimes)):
        previous = regimes.iloc[position - 1]
        current = regimes.iloc[position]
        cross_date = current["start_date"]
        matches = data.index[data["date"] == cross_date]
        if len(matches) == 0:
            continue
        index = int(matches[0])
        before = data.iloc[max(0, index - lookaround):index]
        after = data.iloc[index + 1:index + 1 + lookaround]
        current_row = data.iloc[index]
        velocity = float(current_row["smart_money_pct"] - data.iloc[max(0, index - 1)]["smart_money_pct"])
        persistence = min(1.0, float(current["duration_sessions"]) / max(1, false_regime_days))
        cross_volume = float(current_row["relative_volume"])
        weight = float(abs(velocity) * max(cross_volume, 0.0) * persistence)
        rows.append(
            {
                "ticker": ticker.upper(),
                "cross_date": cross_date,
                "from_regime": previous["regime"],
                "to_regime": current["regime"],
                "smart_money_pct": float(current_row["smart_money_pct"]),
                "crossing_velocity": velocity,
                "pre_cross_relative_volume": float(before["relative_volume"].median()) if not before.empty else np.nan,
                "cross_relative_volume": cross_volume,
                "post_cross_relative_volume": float(after["relative_volume"].median()) if not after.empty else np.nan,
                "subsequent_regime_duration": int(current["duration_sessions"]),
                "false_regime": bool(int(current["duration_sessions"]) <= false_regime_days),
                "persistence_factor": persistence,
                "volume_weighted_cross_load": weight,
            }
        )
    return pd.DataFrame(rows)


def analyze_smart_money_regimes(
    frame: pd.DataFrame,
    ticker: str,
    memory_rows: int | None = None,
    lower_equilibrium: float = 49.0,
    upper_equilibrium: float = 51.0,
    minimum_persistence: int = 2,
    lookaround: int = 3,
    false_regime_days: int = 3,
    high_volume_threshold: float = 1.25,
) -> SmartMoneyRegimeResult:
    required = {"date", "smart_money_pct", "volume", "relative_volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas para analizar regímenes: {sorted(missing)}")
    if lower_equilibrium >= upper_equilibrium:
        raise ValueError("lower_equilibrium debe ser menor que upper_equilibrium")
    if minimum_persistence < 1:
        raise ValueError("minimum_persistence debe ser al menos 1")

    data = frame.sort_values("date").dropna(subset=["smart_money_pct", "relative_volume"]).copy()
    if memory_rows is not None:
        data = data.tail(memory_rows)
    if data.empty:
        raise ValueError("No hay datos utilizables para analizar Smart Money")

    data["confirmed_regime"] = _confirmed_states(
        data["smart_money_pct"], lower_equilibrium, upper_equilibrium, minimum_persistence
    )
    regimes = _regime_rows(data, ticker)
    crossings = _crossing_rows(data, regimes, ticker, lookaround, false_regime_days)

    sessions = int(len(data))
    crossing_count = int(len(crossings))
    regime_durations = regimes["duration_sessions"] if not regimes.empty else pd.Series(dtype=float)
    median_duration = float(regime_durations.median()) if not regime_durations.empty else np.nan
    mean_duration = float(regime_durations.mean()) if not regime_durations.empty else np.nan
    duration_cv = _safe_divide(float(regime_durations.std(ddof=0)), mean_duration)
    false_rate = float(crossings["false_regime"].mean()) if not crossings.empty else np.nan
    confirmed_cross_rate = float((crossings["cross_relative_volume"] >= high_volume_threshold).mean()) if not crossings.empty else np.nan
    median_cross_velocity = float(crossings["crossing_velocity"].abs().median()) if not crossings.empty else np.nan
    mean_cross_velocity = float(crossings["crossing_velocity"].abs().mean()) if not crossings.empty else np.nan
    crossings_per_session = _safe_divide(crossing_count, sessions)
    memory_regime_load = float(crossing_count) if memory_rows is not None else np.nan
    weighted_load = float(crossings["volume_weighted_cross_load"].sum()) if not crossings.empty else 0.0
    coherence = float(1.0 / (1.0 + crossing_count))
    inertia_index = _safe_divide(median_duration, float(memory_rows or sessions))

    current_regime = regimes.iloc[-1] if not regimes.empty else None
    current_regime_relative_volume = float(current_regime["median_relative_volume"]) if current_regime is not None else np.nan
    current_regime_strength = float(current_regime["institutional_regime_strength"]) if current_regime is not None else np.nan
    median_regime_strength = float(regimes["institutional_regime_strength"].median()) if not regimes.empty else np.nan

    summary = pd.DataFrame(
        [{
            "ticker": ticker.upper(),
            "from_date": data.iloc[0]["date"],
            "to_date": data.iloc[-1]["date"],
            "sessions": sessions,
            "memory_rows": memory_rows,
            "equilibrium_lower": lower_equilibrium,
            "equilibrium_upper": upper_equilibrium,
            "minimum_persistence": minimum_persistence,
            "equilibrium_cross_count": crossing_count,
            "crossings_per_100_sessions": 100.0 * crossings_per_session if pd.notna(crossings_per_session) else np.nan,
            "median_regime_duration": median_duration,
            "mean_regime_duration": mean_duration,
            "current_regime": current_regime["regime"] if current_regime is not None else "UNKNOWN",
            "current_regime_duration": int(current_regime["duration_sessions"]) if current_regime is not None else 0,
            "regime_duration_cv": duration_cv,
            "false_regime_rate": false_rate,
            "mean_crossing_velocity": mean_cross_velocity,
            "median_crossing_velocity": median_cross_velocity,
            "median_relative_volume": float(data["relative_volume"].median()),
            "current_regime_relative_volume": current_regime_relative_volume,
            "high_volume_cross_rate": confirmed_cross_rate,
            "median_institutional_regime_strength": median_regime_strength,
            "current_institutional_regime_strength": current_regime_strength,
            "memory_regime_load": memory_regime_load,
            "volume_weighted_memory_regime_load": weighted_load,
            "smart_money_inertia_index": inertia_index,
            "smart_money_regime_coherence": coherence,
        }]
    )
    return SmartMoneyRegimeResult(summary=summary, regimes=regimes, crossings=crossings)


def correlation_report(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    numeric = frame.select_dtypes(include=[np.number]).copy()
    targets = [
        column for column in [
            "predictability_score",
            "rare_event_f1",
            "balanced_accuracy",
            "macro_f1",
            "mean_brier_skill",
            "memory_rows",
        ] if column in numeric.columns
    ]
    drivers = [
        column for column in [
            "crossings_per_100_sessions",
            "median_regime_duration",
            "regime_duration_cv",
            "false_regime_rate",
            "median_crossing_velocity",
            "median_relative_volume",
            "high_volume_cross_rate",
            "median_institutional_regime_strength",
            "memory_regime_load",
            "volume_weighted_memory_regime_load",
            "smart_money_inertia_index",
            "smart_money_regime_coherence",
        ] if column in numeric.columns
    ]
    rows: list[dict[str, object]] = []
    for target in targets:
        for driver in drivers:
            sample = numeric[[driver, target]].dropna()
            if len(sample) < 3:
                correlation = np.nan
            else:
                correlation = float(sample[driver].corr(sample[target], method="spearman"))
            rows.append({
                "driver_metric": driver,
                "target_metric": target,
                "observations": int(len(sample)),
                "spearman_correlation": correlation,
                "absolute_correlation": abs(correlation) if pd.notna(correlation) else np.nan,
            })
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["absolute_correlation", "observations"], ascending=[False, False])
    return result.reset_index(drop=True)

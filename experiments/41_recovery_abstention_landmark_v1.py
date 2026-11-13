#!/usr/bin/env python3
"""Experimento 41: valor de abstenerse usando un landmark común.

Cada observación recibe dos sesiones completas para confirmar. La decisión y
el desenlace empiezan después del mismo cierre landmark, tanto si confirmó como
si no. No simula cartera, recompra ni salida adaptativa.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def load_exp39(path):
    spec = importlib.util.spec_from_file_location("exp39", Path(path).resolve())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def observation_episodes(ticker, bars, low_zone, confirmation, landmark_sessions, outcome_sessions):
    rows = []
    sessions = list(dict.fromkeys(bars["Session"]))
    position = {day: i for i, day in enumerate(sessions)}
    unavailable_until = -1
    for i, row in bars.iterrows():
        start_pos = position[row["Session"]]
        if start_pos <= unavailable_until:
            continue
        observed = (
            bool(row["Weekly_Ascending"])
            and pd.notna(row["Prior_Daily_SMI"])
            and row["Prior_Daily_SMI"] <= -40
            and pd.notna(row["Range_Position_Pct"])
            and row["Range_Position_Pct"] <= low_zone
        )
        if not observed:
            continue
        landmark_pos = start_pos + landmark_sessions - 1
        entry_pos = landmark_pos + 1
        exit_pos = entry_pos + outcome_sessions - 1
        if exit_pos >= len(sessions):
            break
        landmark_day = sessions[landmark_pos]
        window = bars.loc[(bars.index >= i) & (bars["Session"] <= landmark_day)].copy()
        frozen_high = float(row["Range_High"])
        episode_low = float(row["Range_Low"])
        confirmation_time = pd.NaT
        for _, point in window.iterrows():
            episode_low = min(episode_low, float(point["Low"]))
            width = frozen_high - episode_low
            recovery = ((point["Close"] - episode_low) / width * 100 - low_zone) if width > 0 else np.nan
            if (
                bool(point["Weekly_Ascending"])
                and recovery >= confirmation
                and pd.notna(point["Velocity_1H"]) and point["Velocity_1H"] > 0
                and pd.notna(point["Velocity_4H"]) and point["Velocity_4H"] > 0
            ):
                confirmation_time = point["Date"]
                break
        entry_day, exit_day = sessions[entry_pos], sessions[exit_pos]
        entry_row = bars.loc[bars["Session"] == entry_day].iloc[0]
        future = bars.loc[(bars["Session"] >= entry_day) & (bars["Session"] <= exit_day)]
        entry = float(entry_row["Open"])
        exit_close = float(future.loc[future["Session"] == exit_day, "Close"].iloc[-1])
        rows.append({
            "Ticker": ticker, "Observation_Time": row["Date"],
            "Landmark_Time": bars.loc[bars["Session"] == landmark_day, "Date"].max(),
            "Confirmed_By_Landmark": pd.notna(confirmation_time),
            "Confirmation_Time": confirmation_time, "Entry_Time": entry_row["Date"],
            "Exit_Time": future.loc[future["Session"] == exit_day, "Date"].max(),
            "Forward_Return_Pct": (exit_close / entry - 1) * 100,
            "Max_Favorable_Pct": (future["High"].max() / entry - 1) * 100,
            "Max_Adverse_Pct": (future["Low"].min() / entry - 1) * 100,
        })
        unavailable_until = landmark_pos
    return rows


def bootstrap_date_difference(events, reps=5000, seed=4101):
    by_date = []
    for _, day in events.groupby("Landmark_Date"):
        yes = day.loc[day["Confirmed_By_Landmark"], "Forward_Return_Pct"]
        no = day.loc[~day["Confirmed_By_Landmark"], "Forward_Return_Pct"]
        if len(yes) and len(no):
            by_date.append(float(yes.mean() - no.mean()))
    if not by_date:
        return np.nan, np.nan, 0
    values = np.asarray(by_date)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, (reps, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return low, high, len(values)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel-36", required=True)
    p.add_argument("--audit-36", required=True)
    p.add_argument("--intraday-cache", required=True)
    p.add_argument("--experiment-39", default="experiments/39_recovery_monitor_confirmed_walkforward_v1.py")
    p.add_argument("--output", required=True)
    p.add_argument("--landmark-sessions", type=int, default=2)
    p.add_argument("--outcome-sessions", type=int, default=5)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Output is not empty; create a versioned directory")
    out.mkdir(parents=True, exist_ok=True)
    exp39 = load_exp39(a.experiment_39)
    panel = pd.read_csv(a.panel_36, parse_dates=["date"])
    accepted = pd.read_csv(a.audit_36)
    tickers = accepted.loc[accepted["Status"] == "ACCEPTED", "Ticker"].astype(str)
    records, audit = [], []
    for n, ticker in enumerate(tickers, 1):
        try:
            context = panel.loc[panel["Ticker"] == ticker]
            bars = exp39.prepare_bars(Path(a.intraday_cache) / f"{ticker}.csv", context, 40)
            found = observation_episodes(ticker, bars, 3.0, 10.0,
                                         a.landmark_sessions, a.outcome_sessions)
            records.extend(found)
            audit.append({"Ticker": ticker, "Status": "ACCEPTED", "Episodes": len(found)})
        except Exception as exc:
            audit.append({"Ticker": ticker, "Status": "EXCLUDED", "Reason": str(exc)})
        if n % 50 == 0 or n == len(tickers):
            print(f"[{n}/{len(tickers)}] {ticker}", flush=True)
    pd.DataFrame(audit).to_csv(out / "audit.csv", index=False)
    events = pd.DataFrame(records)
    if events.empty:
        events.to_csv(out / "episodes.csv", index=False)
        print("No episodes")
        return
    events["Landmark_Date"] = pd.to_datetime(events["Landmark_Time"]).dt.date.astype(str)
    events.to_csv(out / "episodes.csv", index=False)
    summary = events.groupby("Confirmed_By_Landmark").agg(
        Episodes=("Ticker", "size"), Tickers=("Ticker", "nunique"),
        Dates=("Landmark_Date", "nunique"), Mean_Return_Pct=("Forward_Return_Pct", "mean"),
        Median_Return_Pct=("Forward_Return_Pct", "median"),
        Positive_Pct=("Forward_Return_Pct", lambda s: 100 * (s > 0).mean()),
        Mean_MFE_Pct=("Max_Favorable_Pct", "mean"), Mean_MAE_Pct=("Max_Adverse_Pct", "mean"),
    ).reset_index()
    low, high, matched = bootstrap_date_difference(events)
    summary["Matched_Dates_For_Difference"] = matched
    summary["Confirmed_Minus_Unconfirmed_CI95_Low"] = low
    summary["Confirmed_Minus_Unconfirmed_CI95_High"] = high
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Output: {out.resolve()}")


if __name__ == "__main__":
    main()

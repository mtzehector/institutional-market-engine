#!/usr/bin/env python3
"""Experimento 39: recuperación intradía confirmada dentro de tendencia semanal.

Usa únicamente barras cerradas. El contexto diario procede del cierre anterior;
el semanal ya estaba disponible al abrir la sesión. Es una réplica conservadora,
no idéntica, del Pine Recovery Monitor v0.3.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--panel-36", required=True)
    p.add_argument("--intraday-cache", required=True)
    p.add_argument("--audit-36", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--daily-window", type=int, default=40)
    p.add_argument("--low-zone-pct", type=float, default=3.0)
    p.add_argument("--confirmation-pct", type=float, default=10.0)
    p.add_argument("--expiry-sessions", type=int, default=10)
    p.add_argument("--outcome-sessions", type=int, default=5)
    return p.parse_args()


def block_dynamics(bars):
    """Completed 1H/4H-like RTH blocks, forward-filled only after block close."""
    endpoints = {}
    for size, length, name in ((4, 4, "Velocity_1H"), (16, 3, "Velocity_4H")):
        points = []
        for _, session in bars.groupby("Session", sort=True):
            session = session.sort_values("Date")
            for start in range(0, len(session), size):
                chunk = session.iloc[start : start + size]
                if len(chunk):
                    points.append((chunk["Date"].iloc[-1], chunk["Close"].iloc[-1]))
        series = pd.Series(dict(points), dtype=float).sort_index()
        velocity = (series - series.shift(length - 1)) / (length - 1)
        endpoints[name] = velocity.reindex(bars["Date"], method="ffill").to_numpy()
    return endpoints


def prepare_bars(path, context, window):
    x = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
    clock = x["Date"].dt.strftime("%H:%M")
    x = x.loc[(clock >= "09:30") & (clock <= "15:45")].copy()
    x["Session"] = x["Date"].dt.normalize()
    counts = x.groupby("Session")["Date"].transform("size")
    x = x.loc[counts == 26].copy()
    daily = x.groupby("Session", sort=True).agg(Session_High=("High", "max"), Session_Low=("Low", "min"))
    prior_high = daily["Session_High"].shift(1).rolling(window - 1, min_periods=window - 1).max()
    prior_low = daily["Session_Low"].shift(1).rolling(window - 1, min_periods=window - 1).min()
    x = x.join(prior_high.rename("Prior_High"), on="Session")
    x = x.join(prior_low.rename("Prior_Low"), on="Session")
    x["Current_High"] = x.groupby("Session")["High"].cummax()
    x["Current_Low"] = x.groupby("Session")["Low"].cummin()
    x["Range_High"] = x[["Prior_High", "Current_High"]].max(axis=1)
    x["Range_Low"] = x[["Prior_Low", "Current_Low"]].min(axis=1)
    x["Range_Position_Pct"] = (x["Close"] - x["Range_Low"]) / (x["Range_High"] - x["Range_Low"]) * 100
    ctx = context.sort_values("date").copy()
    ctx["Session"] = ctx["date"].dt.normalize()
    ctx["Prior_Daily_SMI"] = ctx["Daily_SMI"].shift(1)
    ctx["Prior_Daily_Signal"] = ctx["Daily_Signal"].shift(1)
    x = x.merge(ctx[["Session", "Prior_Daily_SMI", "Prior_Daily_Signal", "Weekly_Ascending"]], on="Session", how="left")
    dynamics = block_dynamics(x)
    for column, values in dynamics.items():
        x[column] = values
    return x.reset_index(drop=True)


def episodes(ticker, x, low_zone, confirmation, expiry, outcome_sessions):
    rows = []
    active = None
    sessions = list(dict.fromkeys(x["Session"]))
    session_pos = {day: i for i, day in enumerate(sessions)}
    for i, row in x.iterrows():
        day_i = session_pos[row["Session"]]
        if active is not None and (not bool(row["Weekly_Ascending"]) or day_i - active["Start_Pos"] > expiry):
            active = None
        observation = (
            bool(row["Weekly_Ascending"])
            and pd.notna(row["Prior_Daily_SMI"])
            and row["Prior_Daily_SMI"] <= -40
            and pd.notna(row["Range_Position_Pct"])
            and row["Range_Position_Pct"] <= low_zone
        )
        if active is None and observation:
            active = {"Start_Pos": day_i, "Start_Time": row["Date"],
                      "Frozen_High": row["Range_High"], "Low": row["Range_Low"]}
        if active is None:
            continue
        active["Low"] = min(active["Low"], row["Low"])
        width = active["Frozen_High"] - active["Low"]
        recovery = (row["Close"] - active["Low"]) / width * 100 - low_zone if width > 0 else np.nan
        confirmed = (
            recovery >= confirmation
            and pd.notna(row["Velocity_1H"]) and row["Velocity_1H"] > 0
            and pd.notna(row["Velocity_4H"]) and row["Velocity_4H"] > 0
        )
        if not confirmed or i + 1 >= len(x):
            continue
        future_days = sessions[day_i + 1 : day_i + outcome_sessions + 1]
        future = x.loc[x["Session"].isin(future_days)]
        if len(future_days) < outcome_sessions or future.empty:
            active = None
            continue
        entry = float(x.iloc[i + 1]["Open"])
        exit_close = float(future.loc[future["Session"] == future_days[-1], "Close"].iloc[-1])
        rows.append({
            "Ticker": ticker, "Observation_Time": active["Start_Time"], "Confirmation_Time": row["Date"],
            "Entry_Time": x.iloc[i + 1]["Date"], "Entry_Price": entry,
            "Weekly_Ascending": True, "Prior_Daily_SMI": row["Prior_Daily_SMI"],
            "Prior_Daily_Above_Signal": row["Prior_Daily_SMI"] > row["Prior_Daily_Signal"],
            "Recovery_After_Zone_Pct": recovery, "Velocity_1H": row["Velocity_1H"],
            "Velocity_4H": row["Velocity_4H"], "Outcome_Sessions": outcome_sessions,
            "Forward_Return_Pct": (exit_close / entry - 1) * 100,
            "Max_Favorable_Pct": (future["High"].max() / entry - 1) * 100,
            "Max_Adverse_Pct": (future["Low"].min() / entry - 1) * 100,
        })
        active = None
    return rows


def main():
    a = arguments()
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Output is not empty; create a versioned directory")
    out.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(a.panel_36, parse_dates=["date"])
    accepted = pd.read_csv(a.audit_36)
    tickers = accepted.loc[accepted["Status"] == "ACCEPTED", "Ticker"].astype(str).tolist()
    all_rows, audit = [], []
    for n, ticker in enumerate(tickers, 1):
        try:
            ctx = panel.loc[panel["Ticker"] == ticker]
            bars = prepare_bars(Path(a.intraday_cache) / f"{ticker}.csv", ctx, a.daily_window)
            found = episodes(ticker, bars, a.low_zone_pct, a.confirmation_pct,
                             a.expiry_sessions, a.outcome_sessions)
            all_rows.extend(found)
            audit.append({"Ticker": ticker, "Status": "ACCEPTED", "Bars": len(bars), "Events": len(found)})
        except Exception as exc:
            audit.append({"Ticker": ticker, "Status": "EXCLUDED", "Reason": str(exc)})
        if n % 50 == 0 or n == len(tickers):
            print(f"[{n}/{len(tickers)}] {ticker}", flush=True)
    pd.DataFrame(audit).to_csv(out / "audit.csv", index=False)
    events = pd.DataFrame(all_rows)
    events.to_csv(out / "events.csv", index=False)
    if events.empty:
        print("No confirmed events")
        return
    events["Year"] = pd.to_datetime(events["Confirmation_Time"]).dt.year
    summary = events.groupby("Year").agg(
        Events=("Ticker", "size"), Tickers=("Ticker", "nunique"),
        Mean_Return_Pct=("Forward_Return_Pct", "mean"), Median_Return_Pct=("Forward_Return_Pct", "median"),
        Positive_Pct=("Forward_Return_Pct", lambda s: 100 * (s > 0).mean()),
        Mean_MFE_Pct=("Max_Favorable_Pct", "mean"), Mean_MAE_Pct=("Max_Adverse_Pct", "mean"),
    ).reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Output: {out.resolve()}")


if __name__ == "__main__":
    main()

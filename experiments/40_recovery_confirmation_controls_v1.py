#!/usr/bin/env python3
"""Controles para eventos del experimento 39: mercado y entrada en observación."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def regular(path):
    x = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
    clock = x["Date"].dt.strftime("%H:%M")
    return x.loc[(clock >= "09:30") & (clock <= "15:45")].reset_index(drop=True)


def price_after(frame, time):
    rows = frame.index[frame["Date"] > time]
    return float(frame.loc[rows[0], "Open"]) if len(rows) else np.nan


def close_at_or_before(frame, time):
    rows = frame.index[frame["Date"] <= time]
    return float(frame.loc[rows[-1], "Close"]) if len(rows) else np.nan


def bootstrap_by_date(frame, column, repetitions=5000, seed=4001):
    values = [g[column].to_numpy(float) for _, g in frame.groupby("Confirmation_Date")]
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions)
    for i in range(repetitions):
        sampled = rng.integers(0, len(values), len(values))
        draws[i] = np.concatenate([values[j] for j in sampled]).mean()
    return np.quantile(draws, [0.025, 0.975])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--events", required=True)
    p.add_argument("--intraday-cache", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Output is not empty; create a versioned directory")
    out.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(a.events, parse_dates=["Observation_Time", "Confirmation_Time", "Entry_Time"])
    cache = {}
    qqq = regular(Path(a.intraday_cache) / "QQQ.csv")
    rows = []
    for _, event in events.iterrows():
        ticker = event["Ticker"]
        if ticker not in cache:
            cache[ticker] = regular(Path(a.intraday_cache) / f"{ticker}.csv")
        bars = cache[ticker]
        entry_index = bars.index[bars["Date"] == event["Entry_Time"]]
        if not len(entry_index):
            continue
        start_i = int(entry_index[0])
        entry_day = bars.loc[start_i, "Date"].normalize()
        days = list(dict.fromkeys(bars.loc[start_i:, "Date"].dt.normalize()))
        if len(days) < int(event["Outcome_Sessions"]):
            continue
        exit_day = days[int(event["Outcome_Sessions"]) - 1]
        exit_time = bars.loc[bars["Date"].dt.normalize() == exit_day, "Date"].max()
        exit_price = close_at_or_before(bars, exit_time)
        observation_entry = price_after(bars, event["Observation_Time"])
        qqq_entry = price_after(qqq, event["Confirmation_Time"])
        qqq_exit = close_at_or_before(qqq, exit_time)
        raw = float(event["Forward_Return_Pct"])
        observation_return = (exit_price / observation_entry - 1) * 100
        market_return = (qqq_exit / qqq_entry - 1) * 100
        rows.append({**event.to_dict(), "Exit_Time": exit_time,
                     "Observation_Entry_Price": observation_entry,
                     "Observation_Entry_Return_Pct": observation_return,
                     "Confirmation_Minus_Observation_Pp": raw - observation_return,
                     "QQQ_Return_Pct": market_return, "Excess_vs_QQQ_Pp": raw - market_return})
    controlled = pd.DataFrame(rows).dropna(subset=["QQQ_Return_Pct", "Observation_Entry_Return_Pct"])
    controlled["Confirmation_Date"] = controlled["Confirmation_Time"].dt.date.astype(str)
    controlled.to_csv(out / "controlled_events.csv", index=False)
    metrics = []
    for column in ("Forward_Return_Pct", "Excess_vs_QQQ_Pp", "Confirmation_Minus_Observation_Pp"):
        ci = bootstrap_by_date(controlled, column)
        metrics.append({"Metric": column, "Events": len(controlled),
                        "Independent_Dates": controlled["Confirmation_Date"].nunique(),
                        "Mean": controlled[column].mean(), "Median": controlled[column].median(),
                        "CI95_Date_Block_Low": ci[0], "CI95_Date_Block_High": ci[1],
                        "Positive_Pct": 100 * (controlled[column] > 0).mean()})
    summary = pd.DataFrame(metrics)
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Output: {out.resolve()}")


if __name__ == "__main__":
    main()

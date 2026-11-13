#!/usr/bin/env python3
"""Experimento 42: política transversal de recuperación con exposición comparable.

Cada fecha landmark forma una cohorte con 100% de exposición durante cinco
sesiones. Se reportan rendimientos de cohorte; no se encadenan porque se solapan.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def regular(path):
    x = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
    clock = x["Date"].dt.strftime("%H:%M")
    return x.loc[(clock >= "09:30") & (clock <= "15:45")].reset_index(drop=True)


def price_after(frame, timestamp, column="Open"):
    rows = frame.index[frame["Date"] > timestamp]
    return float(frame.loc[rows[0], column]) if len(rows) else np.nan


def price_at_or_before(frame, timestamp, column="Close"):
    rows = frame.index[frame["Date"] <= timestamp]
    return float(frame.loc[rows[-1], column]) if len(rows) else np.nan


def date_bootstrap(frame, columns, reps=10000, seed=4201):
    rng = np.random.default_rng(seed)
    values = frame[columns].to_numpy(float)
    index = rng.integers(0, len(values), size=(reps, len(values)))
    means = values[index].mean(axis=1)
    rows = []
    for j, column in enumerate(columns):
        low, high = np.quantile(means[:, j], [0.025, 0.975])
        rows.append({"Metric": column, "Mean": values[:, j].mean(),
                     "Median": np.median(values[:, j]), "CI95_Low": low,
                     "CI95_High": high, "Positive_Pct": 100 * (values[:, j] > 0).mean()})
    return pd.DataFrame(rows)


def balanced_difference(episodes, reps=10000, seed=4202):
    """Equal-count resampling within dates containing both groups."""
    rng = np.random.default_rng(seed)
    paired = []
    for _, day in episodes.groupby("Landmark_Date"):
        yes = day.loc[day["Confirmed_By_Landmark"], "Forward_Return_Pct"].to_numpy(float)
        no = day.loc[~day["Confirmed_By_Landmark"], "Forward_Return_Pct"].to_numpy(float)
        if len(yes) and len(no):
            paired.append((yes, no))
    draws = np.empty(reps)
    for r in range(reps):
        deltas = []
        for yes, no in paired:
            k = min(len(yes), len(no))
            deltas.append(rng.choice(yes, k, replace=False).mean() -
                          rng.choice(no, k, replace=False).mean())
        draws[r] = np.mean(deltas)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"Paired_Dates": len(paired), "Mean_Delta_Pp": draws.mean(),
            "CI95_Low": low, "CI95_High": high,
            "Probability_Delta_Positive_Pct": 100 * (draws > 0).mean()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", required=True)
    p.add_argument("--intraday-cache", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Output is not empty; create a versioned directory")
    out.mkdir(parents=True, exist_ok=True)
    episodes = pd.read_csv(a.episodes, parse_dates=["Landmark_Time", "Entry_Time", "Exit_Time"])
    episodes["Confirmed_By_Landmark"] = episodes["Confirmed_By_Landmark"].astype(bool)
    episodes["Landmark_Date"] = episodes["Landmark_Time"].dt.date.astype(str)
    qqq = regular(Path(a.intraday_cache) / "QQQ.csv")
    market = []
    for _, row in episodes.drop_duplicates("Landmark_Date").iterrows():
        entry = price_after(qqq, row["Landmark_Time"])
        exit_price = price_at_or_before(qqq, row["Exit_Time"])
        market.append({"Landmark_Date": row["Landmark_Date"],
                       "QQQ_Return_Pct": (exit_price / entry - 1) * 100})
    qqq_by_date = pd.DataFrame(market).set_index("Landmark_Date")["QQQ_Return_Pct"]
    cohorts = []
    for date, day in episodes.groupby("Landmark_Date", sort=True):
        confirmed = day.loc[day["Confirmed_By_Landmark"], "Forward_Return_Pct"]
        all_return = day["Forward_Return_Pct"].mean()
        qqq_return = qqq_by_date.get(date, np.nan)
        selected = confirmed.mean() if len(confirmed) else qqq_return
        cohorts.append({
            "Landmark_Date": date, "Observed_Tickers": len(day),
            "Confirmed_Tickers": len(confirmed), "Used_QQQ_Fallback": len(confirmed) == 0,
            "CONFIRMED_OR_QQQ_Pct": selected, "ALL_OBSERVED_Pct": all_return,
            "QQQ_ALWAYS_Pct": qqq_return,
            "Delta_vs_All_Pp": selected - all_return,
            "Delta_vs_QQQ_Pp": selected - qqq_return,
        })
    cohorts = pd.DataFrame(cohorts).dropna()
    cohorts.to_csv(out / "date_cohorts.csv", index=False)
    metrics = date_bootstrap(cohorts, ["CONFIRMED_OR_QQQ_Pct", "ALL_OBSERVED_Pct",
                                       "QQQ_ALWAYS_Pct", "Delta_vs_All_Pp", "Delta_vs_QQQ_Pp"])
    metrics["Decision_Dates"] = len(cohorts)
    metrics["Confirmed_Selection_Dates"] = int((~cohorts["Used_QQQ_Fallback"]).sum())
    metrics["QQQ_Fallback_Dates"] = int(cohorts["Used_QQQ_Fallback"].sum())
    metrics.to_csv(out / "policy_summary.csv", index=False)
    balance = pd.DataFrame([balanced_difference(episodes)])
    balance.to_csv(out / "balanced_within_date.csv", index=False)
    print(metrics.to_string(index=False))
    print("\nEqual-count within-date control")
    print(balance.to_string(index=False))
    print(f"Output: {out.resolve()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Seleccion anual previsible de tickers para el filtro largo del experimento 36.

Usa dos anos previos, embarga labels que alcanzan el ano evaluado y conserva
todos los anos de prueba. No es una simulacion de cartera ni de recompras.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--first-test-year", type=int, default=2023)
    p.add_argument("--last-test-year", type=int, default=2026)
    p.add_argument("--train-years", type=int, default=2)
    p.add_argument("--min-eligible", type=int, default=30)
    p.add_argument("--min-other", type=int, default=80)
    p.add_argument("--top-fraction", type=float, default=0.2)
    return p.parse_args()


def select_tickers(train, min_eligible, min_other, fraction):
    stats = train.groupby(["Ticker", "Eligible"], observed=True)["Forward_Return_Pct"].agg(
        ["size", "mean"]
    ).unstack()
    if True not in stats["size"] or False not in stats["size"]:
        return pd.DataFrame()
    selected = pd.DataFrame({
        "Eligible_Days_Train": stats["size"][True],
        "Other_Days_Train": stats["size"][False],
        "Eligible_Mean_Train": stats["mean"][True],
        "Other_Mean_Train": stats["mean"][False],
    }).dropna()
    selected = selected.loc[
        (selected["Eligible_Days_Train"] >= min_eligible)
        & (selected["Other_Days_Train"] >= min_other)
    ].copy()
    selected["Train_Uplift_Pp"] = selected["Eligible_Mean_Train"] - selected["Other_Mean_Train"]
    selected = selected.sort_values(["Train_Uplift_Pp", "Ticker"], ascending=[False, True])
    selected["Train_Rank"] = np.arange(1, len(selected) + 1)
    selected["Selected"] = selected["Train_Rank"] <= math.ceil(len(selected) * fraction)
    return selected.reset_index()


def same_date_comparison(test, selected):
    eligible = test.loc[test["Eligible"]].copy()
    eligible["Selected"] = eligible["Ticker"].isin(selected)
    by_date = eligible.groupby(["date", "Selected"], observed=True)["Forward_Return_Pct"].agg(
        ["mean", "size"]
    ).unstack()
    if True not in by_date["mean"] or False not in by_date["mean"]:
        return pd.DataFrame(columns=["date", "Selected_Mean_Pct", "Other_Mean_Pct", "Delta_Pp", "Selected_Tickers", "Other_Tickers"])
    matched = pd.DataFrame({
        "Selected_Mean_Pct": by_date["mean"][True],
        "Other_Mean_Pct": by_date["mean"][False],
        "Selected_Tickers": by_date["size"][True],
        "Other_Tickers": by_date["size"][False],
    }).dropna()
    matched["Delta_Pp"] = matched["Selected_Mean_Pct"] - matched["Other_Mean_Pct"]
    return matched.reset_index()


def main():
    a = arguments()
    if not 0 < a.top_fraction < 1:
        raise SystemExit("top-fraction must be between 0 and 1")
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Output is not empty; create a new versioned directory")
    out.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(a.panel, parse_dates=["date", "Entry_Date", "Exit_Date"])
    panel = panel.loc[panel["Forward_Return_Pct"].notna() & panel["Weekly_Bars_Seen"].ge(20)].copy()
    panel["Eligible"] = panel["Eligible"].astype(bool)
    picks, matched_rows, summary = [], [], []
    for year in range(a.first_test_year, a.last_test_year + 1):
        begin = pd.Timestamp(year=year, month=1, day=1)
        next_begin = pd.Timestamp(year=year + 1, month=1, day=1)
        train_start = pd.Timestamp(year=year - a.train_years, month=1, day=1)
        train = panel.loc[(panel["date"] >= train_start) & (panel["Exit_Date"] < begin)]
        test = panel.loc[(panel["date"] >= begin) & (panel["Exit_Date"] < next_begin)]
        choice = select_tickers(train, a.min_eligible, a.min_other, a.top_fraction)
        if choice.empty:
            summary.append({"Test_Year": year, "Status": "NO_TRAINABLE_TICKERS"})
            continue
        choice.insert(0, "Test_Year", year)
        picks.append(choice)
        top = set(choice.loc[choice["Selected"], "Ticker"])
        test = test.loc[test["Ticker"].isin(choice["Ticker"])].copy()
        matched = same_date_comparison(test, top)
        matched.insert(0, "Test_Year", year)
        matched_rows.append(matched)
        event_top = test.loc[test["Eligible"] & test["Ticker"].isin(top)]
        event_other = test.loc[test["Eligible"] & ~test["Ticker"].isin(top)]
        summary.append({
            "Test_Year": year, "Status": "EVALUATED", "Rankable_Tickers": len(choice),
            "Selected_Tickers": len(top), "Selected_Eligible_Events": len(event_top),
            "Other_Eligible_Events": len(event_other),
            "Selected_Forward_Mean_Pct": event_top["Forward_Return_Pct"].mean(),
            "Other_Forward_Mean_Pct": event_other["Forward_Return_Pct"].mean(),
            "Selected_Positive_Pct": 100 * (event_top["Forward_Return_Pct"] > 0).mean(),
            "Matched_Dates": len(matched),
            "Matched_Date_Mean_Delta_Pp": matched["Delta_Pp"].mean(),
            "Matched_Date_Positive_Pct": 100 * (matched["Delta_Pp"] > 0).mean(),
        })
    pd.concat(picks, ignore_index=True).to_csv(out / "annual_picks.csv", index=False)
    pd.concat(matched_rows, ignore_index=True).to_csv(out / "matched_dates.csv", index=False)
    result = pd.DataFrame(summary)
    result.to_csv(out / "summary.csv", index=False)
    print(result.to_string(index=False))
    print(f"Output: {out.resolve()}")


if __name__ == "__main__":
    main()

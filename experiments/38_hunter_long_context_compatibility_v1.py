#!/usr/bin/env python3
"""Audita si Hunter diario y filtro SMI largo pueden coexistir.

Traduccion diaria de Hunter v1.3 con defaults 40 barras, -40 SMI y 3% zona.
No replica el Monitor 1H/4H ni constituye una entrada negociable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Output is not empty; create a versioned directory")
    out.mkdir(parents=True, exist_ok=True)
    x = pd.read_csv(a.panel, parse_dates=["date"])
    x = x.sort_values(["Ticker", "date"]).copy()
    grouped = x.groupby("Ticker", sort=False)
    high = grouped["high"].transform(lambda s: s.rolling(40, min_periods=40).max())
    low = grouped["low"].transform(lambda s: s.rolling(40, min_periods=40).min())
    x["Hunter_Range_Position_Pct"] = (x["close"] - low) / (high - low) * 100
    x["Hunter_Condition"] = (x["Daily_SMI"] <= -40) & (x["Hunter_Range_Position_Pct"] <= 3)
    x["Hunter_Trigger"] = x["Hunter_Condition"] & ~x.groupby("Ticker")["Hunter_Condition"].shift(fill_value=False)
    events = x.loc[x["Hunter_Trigger"] & x["Forward_Return_Pct"].notna()].copy()
    events["Context"] = "NO_WEEKLY_ASCENT"
    events.loc[events["Weekly_Ascending"], "Context"] = "WEEKLY_ASCENT_ONLY"
    events.loc[events["Eligible"], "Context"] = "DAILY_AND_WEEKLY_ASCENT"
    events["Year"] = events["date"].dt.year
    events.to_csv(out / "events.csv", index=False)
    summary = events.groupby(["Year", "Context"], observed=True).agg(
        Events=("Ticker", "size"), Tickers=("Ticker", "nunique"),
        Mean_Forward_Return_Pct=("Forward_Return_Pct", "mean"),
        Positive_Pct=("Forward_Return_Pct", lambda s: 100 * (s > 0).mean()),
    ).reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Output: {out.resolve()}")


if __name__ == "__main__":
    main()

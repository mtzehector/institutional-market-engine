#!/usr/bin/env python3
"""Experimento 36: filtro diario/semanal SMI con calendario walk-forward.

No entrena un ranking ni simula capital. Cada senal se conoce al cierre de t;
el desenlace fijo comienza en la apertura de t+1. Salidas siempre versionadas.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PRICE = ["open", "high", "low", "close", "volume"]


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--eod-cache", required=True)
    p.add_argument("--intraday-cache", required=True)
    p.add_argument("--audit-15m", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--min-weekly-bars", type=int, default=20)
    p.add_argument("--max-overlap-diff-pct", type=float, default=1.0)
    return p.parse_args()


def smi(frame, k=5, d=3):
    high = frame["high"].rolling(k, min_periods=k).max()
    low = frame["low"].rolling(k, min_periods=k).min()
    rel = frame["close"] - (high + low) / 2
    width = high - low
    numerator = rel.ewm(span=d, adjust=False, min_periods=1).mean().ewm(
        span=d, adjust=False, min_periods=1
    ).mean()
    denominator = width.ewm(span=d, adjust=False, min_periods=1).mean().ewm(
        span=d, adjust=False, min_periods=1
    ).mean() / 2
    result = 100 * numerator / denominator.replace(0, np.nan)
    return result, result.ewm(span=d, adjust=False, min_periods=1).mean()


def latest_eod_paths(root):
    chosen = {}
    for path in Path(root).glob("*.csv"):
        parts = path.stem.rsplit("_", 2)
        if len(parts) != 3:
            continue
        ticker, start, end = parts
        if ticker not in chosen or (end, start) > chosen[ticker][0]:
            chosen[ticker] = ((end, start), path)
    return {ticker: pair[1] for ticker, pair in chosen.items()}


def regular_intraday_daily(path):
    bars = pd.read_csv(path, parse_dates=["Date"])
    times = bars["Date"].dt.strftime("%H:%M")
    bars = bars.loc[(times >= "09:30") & (times <= "15:45")].copy()
    bars["date"] = bars["Date"].dt.normalize()
    blocks = bars.groupby("date", sort=True)
    daily = blocks.agg(open=("Open", "first"), high=("High", "max"),
                       low=("Low", "min"), close=("Close", "last"),
                       volume=("Volume", "sum"), bars=("Date", "size"))
    return daily.loc[daily["bars"] == 26, PRICE]


def prepare_ticker(ticker, eod_path, intraday_path, max_diff):
    eod = pd.read_csv(eod_path, parse_dates=["date"])
    eod = eod.drop_duplicates("date", keep="last").set_index("date").sort_index()[PRICE]
    intra = regular_intraday_daily(intraday_path)
    overlap = eod[["close"]].join(intra[["close"]], how="inner", lsuffix="_eod", rsuffix="_15m")
    if len(overlap) < 20:
        raise ValueError(f"insufficient_overlap:{len(overlap)}")
    diff = ((overlap["close_15m"] / overlap["close_eod"] - 1).abs() * 100)
    # One corporate-action or vendor-scale mismatch invalidates the merged series.
    if float(diff.max()) > max_diff:
        raise ValueError(f"overlap_price_mismatch:{diff.max():.4f}%")
    extended = pd.concat([eod, intra.loc[intra.index > eod.index.max()]]).sort_index()
    if extended.index.has_duplicates:
        raise ValueError("duplicate_daily_dates")
    extended["Ticker"] = ticker
    return extended, {"Ticker": ticker, "EOD_Last": str(eod.index.max().date()),
                      "Last": str(extended.index.max().date()), "Overlap_Days": len(overlap),
                      "Overlap_Max_Close_Diff_Pct": float(diff.max()),
                      "Appended_Days": len(extended) - len(eod), "Status": "ACCEPTED"}


def daily_weekly_panel(frame, min_weekly_bars):
    x = frame.copy()
    x.index.name = "date"
    x["Daily_SMI"], x["Daily_Signal"] = smi(x)
    x["Daily_Ascending"] = (x["Daily_SMI"] > x["Daily_Signal"]) & (x["Daily_SMI"].diff() > 0)
    group = x.groupby(x.index.to_period("W-FRI"), sort=True)
    weekly = group.agg(open=("open", "first"), high=("high", "max"),
                       low=("low", "min"), close=("close", "last"),
                       volume=("volume", "sum"))
    weekly["Last_Session"] = group.apply(lambda z: z.index.max())
    weekly["Weekly_SMI"], weekly["Weekly_Signal"] = smi(weekly)
    weekly["Weekly_Ascending"] = (weekly["Weekly_SMI"] > weekly["Weekly_Signal"]) & (weekly["Weekly_SMI"].diff() > 0)
    weekly["Weekly_Bars_Seen"] = np.arange(1, len(weekly) + 1)
    # A weekly value becomes usable on the next trading session after its
    # actual last session. This works even when Friday is a holiday.
    last_pos = x.index.get_indexer(weekly["Last_Session"])
    available_pos = last_pos + 1
    weekly = weekly.loc[available_pos < len(x)].copy()
    weekly["Available_Session"] = x.index[available_pos[available_pos < len(x)]]
    eligible = weekly.set_index("Available_Session")[[
        "Weekly_SMI", "Weekly_Signal", "Weekly_Ascending", "Weekly_Bars_Seen"
    ]]
    x = x.join(eligible.reindex(x.index, method="ffill"))
    x["Weekly_Ascending"] = x["Weekly_Ascending"].fillna(False).astype(bool)
    x["Eligible"] = x["Daily_Ascending"] & x["Weekly_Ascending"] & (
        x["Weekly_Bars_Seen"] >= min_weekly_bars)
    return x


def forward_outcome(frame, horizon):
    x = frame.copy()
    # Session t's signal is known at close; entry is next open, exit after h sessions.
    x["Entry_Date"] = pd.Series(x.index, index=x.index).shift(-1)
    x["Exit_Date"] = pd.Series(x.index, index=x.index).shift(-horizon)
    x["Forward_Return_Pct"] = (x["close"].shift(-horizon) / x["open"].shift(-1) - 1) * 100
    return x


def main():
    a = arguments()
    if a.horizon < 1:
        raise SystemExit("horizon must be positive")
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Output is not empty; create a new versioned directory")
    out.mkdir(parents=True, exist_ok=True)
    audit15 = pd.read_csv(a.audit_15m)
    accepted15 = set(audit15.loc[
        audit15["Status"].isin(["UPDATED", "UPDATED_WITH_WARNING"])
        & (pd.to_numeric(audit15["Added_Sessions"], errors="coerce") == 5), "Ticker"].astype(str))
    eod_paths = latest_eod_paths(a.eod_cache)
    tickers = sorted(set(eod_paths) & accepted15)
    panels, audit = [], []
    for i, ticker in enumerate(tickers, 1):
        try:
            daily, info = prepare_ticker(ticker, eod_paths[ticker],
                                         Path(a.intraday_cache) / f"{ticker}.csv",
                                         a.max_overlap_diff_pct)
            panel = forward_outcome(daily_weekly_panel(daily, a.min_weekly_bars), a.horizon)
            panel["Ticker"] = ticker
            panels.append(panel.reset_index())
            audit.append(info)
        except Exception as exc:
            audit.append({"Ticker": ticker, "Status": "EXCLUDED", "Reason": str(exc)})
        if i % 50 == 0 or i == len(tickers):
            print(f"[{i}/{len(tickers)}] {ticker}", flush=True)
    pd.DataFrame(audit).to_csv(out / "audit.csv", index=False)
    if not panels:
        raise SystemExit("No ticker passed audit")
    panel = pd.concat(panels, ignore_index=True)
    panel.to_csv(out / "panel.csv", index=False)
    valid = panel.loc[panel["Forward_Return_Pct"].notna() & panel["Weekly_Bars_Seen"].ge(a.min_weekly_bars)].copy()
    valid["Group"] = np.select(
        [valid["Eligible"], valid["Daily_Ascending"]],
        ["DAILY_AND_WEEKLY", "DAILY_ONLY"], default="OTHER")
    daily = valid.groupby(["date", "Group"], observed=True).agg(
        Tickers=("Ticker", "nunique"), Mean_Forward_Return_Pct=("Forward_Return_Pct", "mean"),
        Median_Forward_Return_Pct=("Forward_Return_Pct", "median"),
        Positive_Pct=("Forward_Return_Pct", lambda s: 100 * (s > 0).mean()),
    ).reset_index()
    daily.to_csv(out / "cross_section_by_date.csv", index=False)
    summary = valid.groupby("Group", observed=True).agg(
        Observations=("Ticker", "size"), Tickers=("Ticker", "nunique"),
        Dates=("date", "nunique"), Mean_Forward_Return_Pct=("Forward_Return_Pct", "mean"),
        Median_Forward_Return_Pct=("Forward_Return_Pct", "median"),
        Positive_Pct=("Forward_Return_Pct", lambda s: 100 * (s > 0).mean()),
    ).reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    print(f"Accepted: {sum(row['Status'] == 'ACCEPTED' for row in audit)}/{len(tickers)}")
    print(summary.to_string(index=False))
    print(f"Output: {out.resolve()}")


if __name__ == "__main__":
    main()

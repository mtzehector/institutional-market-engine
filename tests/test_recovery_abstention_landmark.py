"""Pruebas del calendario landmark del experimento 41."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "exp41", ROOT / "experiments" / "41_recovery_abstention_landmark_v1.py"
)
exp41 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp41)


def test_date_block_bootstrap_uses_only_dates_with_both_groups():
    rows = []
    for date, yes, no in (("2026-01-02", 3.0, 1.0), ("2026-01-05", 4.0, 1.0)):
        rows.extend([
            {"Landmark_Date": date, "Confirmed_By_Landmark": True, "Forward_Return_Pct": yes},
            {"Landmark_Date": date, "Confirmed_By_Landmark": False, "Forward_Return_Pct": no},
        ])
    rows.append({"Landmark_Date": "2026-01-06", "Confirmed_By_Landmark": True,
                 "Forward_Return_Pct": -100.0})
    low, high, matched = exp41.bootstrap_date_difference(pd.DataFrame(rows), reps=500, seed=1)
    assert matched == 2
    assert low >= 2.0
    assert high <= 3.0

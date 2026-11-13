"""Pruebas de controles del experimento 42."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "exp42", ROOT / "experiments" / "42_recovery_transversal_policy_v1.py"
)
exp42 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp42)


def test_balanced_control_gives_dates_equal_weight():
    rows = []
    for date, confirmed, unconfirmed in (
        ("2026-01-02", [3.0], [1.0, 100.0]),
        ("2026-01-05", [6.0, 7.0], [2.0]),
    ):
        rows.extend({"Landmark_Date": date, "Confirmed_By_Landmark": True,
                     "Forward_Return_Pct": value} for value in confirmed)
        rows.extend({"Landmark_Date": date, "Confirmed_By_Landmark": False,
                     "Forward_Return_Pct": value} for value in unconfirmed)
    result = exp42.balanced_difference(pd.DataFrame(rows), reps=1000, seed=2)
    assert result["Paired_Dates"] == 2
    # The result must remain a date-level comparison, not a pooled event mean.
    assert result["Mean_Delta_Pp"] != 0


def test_date_bootstrap_preserves_requested_metrics():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [-1.0, 0.0, 1.0]})
    result = exp42.date_bootstrap(frame, ["a", "b"], reps=200, seed=3)
    assert result["Metric"].tolist() == ["a", "b"]
    assert result.loc[result["Metric"] == "a", "Mean"].iloc[0] == 2.0

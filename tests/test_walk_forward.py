from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from market_engine.backtesting import walk_forward as module
from market_engine.gaps.predictor import GapPrediction


def _prediction() -> GapPrediction:
    return GapPrediction(
        probability_up=0.60,
        probability_down=0.20,
        probability_no_gap=0.20,
        base_up=0.10,
        base_down=0.10,
        lift_up=6.0,
        lift_down=2.0,
        brier_up=0.2,
        brier_down=0.1,
        roc_auc_up=0.6,
        roc_auc_down=0.7,
        training_rows=180,
        historical_up_cases=18,
        historical_down_cases=18,
        mode_up="TEST",
        mode_down="TEST",
    )


def test_walk_forward_uses_origin_to_predict_next_session(monkeypatch):
    rows = 185
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-01", periods=rows),
            "smart_money_pct": np.linspace(45, 65, rows),
            "atr_pct": 2.0,
            "gap_threshold_pct": 1.0,
            "gap_next_pct": 0.0,
            "target_gap_up_next": 0.0,
            "target_gap_down_next": 0.0,
        }
    )
    frame.loc[181, "gap_next_pct"] = 1.5
    frame.loc[181, "target_gap_up_next"] = 1.0

    seen_last_dates: list[pd.Timestamp] = []

    def fake_predict(history: pd.DataFrame) -> GapPrediction:
        seen_last_dates.append(pd.Timestamp(history.iloc[-1]["date"]))
        return _prediction()

    monkeypatch.setattr(module, "predict_next_gap", fake_predict)

    target_date = pd.Timestamp(frame.iloc[182]["date"]).date()
    result = module.walk_forward_gap(
        frame,
        from_date=target_date,
        to_date=target_date,
        min_history_rows=180,
    )

    assert len(result.predictions) == 1
    assert result.predictions.iloc[0]["target_date"].date() == target_date
    assert seen_last_dates[0] == pd.Timestamp(frame.iloc[181]["date"])
    assert bool(result.predictions.iloc[0]["actual_gap_up"])

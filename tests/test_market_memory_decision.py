from __future__ import annotations

import numpy as np
import pandas as pd

from market_engine.evaluation.market_memory_decision import (
    add_causal_thresholds,
    compare_policies,
)


def _recommendations(rows: int = 12) -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=rows, freq="B")
    return pd.DataFrame(
        {
            "target_date": dates,
            "global_novelty_score": np.linspace(1.0, 12.0, rows),
            "similarity_confidence": np.linspace(70.0, 30.0, rows),
            "outcome_stability": np.linspace(65.0, 25.0, rows),
        }
    )


def test_causal_thresholds_require_prior_history() -> None:
    result = add_causal_thresholds(_recommendations(), calibration_history=6)
    assert result.loc[:5, "novelty_veto_threshold"].isna().all()
    assert result.loc[6:, "novelty_veto_threshold"].notna().all()


def test_policy_comparison_rewards_coverage_and_quality() -> None:
    frame = pd.DataFrame(
        {
            "policy": ["A", "A", "B", "B"],
            "accepted": [True, True, True, False],
            "advantage_vs_universe": [10.0, 8.0, 12.0, -5.0],
            "oracle_regret": [4.0, 5.0, 2.0, 20.0],
            "selected_was_oracle": [False, False, True, False],
        }
    )
    comparison = compare_policies(frame)
    assert set(comparison["policy"]) == {"A", "B"}
    assert comparison["policy_score"].notna().all()
    assert comparison.loc[comparison["policy"] == "A", "coverage_rate"].iloc[0] == 1.0

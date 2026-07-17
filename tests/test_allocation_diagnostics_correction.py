from __future__ import annotations

import pandas as pd

from market_engine.evaluation.allocation_diagnostics_correction import (
    _build_corrected_diagnostics,
    _family,
)


def _base_allocations() -> pd.DataFrame:
    dates = pd.date_range("2026-06-01", periods=3, freq="B")
    rows: list[dict[str, object]] = []
    champions = [
        "INSTITUTIONAL_CONFIRMED",
        "CONTROL_SMART_MONEY",
        "CONTROL_MOMENTUM",
    ]
    memory_scores = [82.0, 79.0, 40.0]
    current_scores = [58.0, 60.0, 88.0]
    for index, date in enumerate(dates):
        for champion, memory, current in zip(
            champions, memory_scores, current_scores, strict=True
        ):
            rows.append(
                {
                    "target_date": date,
                    "policy": "MEMORY_ONLY",
                    "champion": champion,
                    "memory_score": memory + index,
                    "memory_score_percentile": {
                        "INSTITUTIONAL_CONFIRMED": 100.0,
                        "CONTROL_SMART_MONEY": 66.7,
                        "CONTROL_MOMENTUM": 33.3,
                    }[champion],
                    "current_intelligence_score": current - index,
                }
            )
    return pd.DataFrame(rows)


def test_real_memory_margin_uses_scores_not_rank_spacing() -> None:
    diagnostics = _build_corrected_diagnostics(_base_allocations())
    assert not diagnostics.empty
    assert (diagnostics["memory_real_margin"] == 3.0).all()
    assert diagnostics["memory_certainty_corrected"].between(0.0, 100.0).all()
    assert diagnostics["memory_ambiguity_corrected"].nunique() > 1


def test_disagreement_is_reduced_for_related_families() -> None:
    assert _family("INSTITUTIONAL_CONFIRMED") == "INSTITUTIONAL"
    assert _family("CONTROL_SMART_MONEY") == "INSTITUTIONAL"
    diagnostics = _build_corrected_diagnostics(_base_allocations())
    assert diagnostics["family_disagreement"].between(0.0, 100.0).all()
    assert diagnostics["memory_current_disagreement_corrected"].between(0.0, 100.0).all()


def test_current_consistency_is_causal_and_bounded() -> None:
    diagnostics = _build_corrected_diagnostics(_base_allocations())
    assert diagnostics["current_consistency_causal"].between(0.0, 100.0).all()
    assert diagnostics.iloc[0]["current_consistency_causal"] == 50.0

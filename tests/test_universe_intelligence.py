from pathlib import Path

import pandas as pd

from market_engine.evaluation.universe_intelligence import build_universe_intelligence


def test_universe_report_builds_rankings_and_coverage(tmp_path: Path) -> None:
    source = tmp_path / "universe.xlsx"
    frame = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "predictability_score": 80.0,
                "predictability_grade": "A_EXCELENTE",
                "observations": 50,
                "directional_accuracy": 0.75,
                "mean_brier_skill": 0.10,
                "mean_calibration_error": 0.08,
                "memory_label": "180_ROWS",
                "memory_rows": 180,
            },
            {
                "ticker": "BBB",
                "predictability_score": 40.0,
                "predictability_grade": "D_DEBIL",
                "observations": 20,
                "directional_accuracy": 0.45,
                "mean_brier_skill": -0.20,
                "mean_calibration_error": 0.20,
                "memory_label": "EXPANDING",
            },
        ]
    )
    with pd.ExcelWriter(source, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="Mejor_Memoria", index=False)

    result = build_universe_intelligence(
        source,
        expected_tickers=3,
        engine_version="0.8.0",
        regime_label="TEST",
    )

    assert result.top_tickers.iloc[0]["ticker"] == "AAA"
    assert result.bottom_tickers.iloc[0]["ticker"] == "BBB"
    coverage = dict(zip(result.coverage["metric"], result.coverage["value"]))
    assert coverage["evaluated_tickers"] == 2
    assert coverage["missing_tickers"] == 1
    manifest = dict(zip(result.manifest["field"], result.manifest["value"]))
    assert len(manifest["source_sha256"]) == 64

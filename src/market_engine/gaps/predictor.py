from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from market_engine.gaps.features import FEATURE_COLUMNS


@dataclass
class GapPrediction:
    probability_up: float
    probability_down: float
    probability_no_gap: float
    base_up: float
    base_down: float
    lift_up: float
    lift_down: float
    brier_up: float
    brier_down: float
    roc_auc_up: float
    roc_auc_down: float
    training_rows: int
    historical_up_cases: int
    historical_down_cases: int
    mode_up: str
    mode_down: str


def _pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def _comparable_probability(
    modeling: pd.DataFrame,
    latest: pd.Series,
    target: str,
    neighbors: int = 40,
) -> float:
    columns = [
        "smart_money_pct",
        "smart_money_delta_3d",
        "relative_volume",
        "atr_pct",
        "return_1d_pct",
        "return_5d_pct",
        "close_position_pct",
        "gap_today_pct",
        "qqq_return_1d_pct",
        "volatility_10d_pct",
    ]
    candidates = modeling.dropna(subset=[target]).copy()
    median = candidates[columns].median()
    scale = candidates[columns].std().replace(0, 1)
    current = latest[columns].fillna(median)
    normalized = (candidates[columns].fillna(median) - current) / scale
    candidates["distance"] = np.sqrt(np.square(normalized).sum(axis=1))
    sample = candidates.nsmallest(min(neighbors, len(candidates)), "distance")
    positives = int(sample[target].sum())
    return (positives + 1) / (len(sample) + 2)


def _fit_one(
    modeling: pd.DataFrame,
    latest: pd.DataFrame,
    target: str,
    min_positive_cases: int = 12,
) -> dict[str, Any]:
    positives = int(modeling[target].sum())
    negatives = len(modeling) - positives
    if positives < min_positive_cases or negatives < min_positive_cases:
        return {
            "probability": np.nan,
            "brier": np.nan,
            "roc_auc": np.nan,
            "mode": "COMPARABLES_BAYESIANOS",
        }

    split = max(1, int(len(modeling) * 0.85))
    development = modeling.iloc[:split]
    test = modeling.iloc[split:]
    model = _pipeline()
    model.fit(development[FEATURE_COLUMNS], development[target].astype(int))
    probability = float(model.predict_proba(latest[FEATURE_COLUMNS])[:, 1][0])

    brier = np.nan
    auc = np.nan
    if not test.empty:
        test_probability = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
        test_target = test[target].astype(int)
        brier = float(brier_score_loss(test_target, test_probability))
        if test_target.nunique() == 2:
            auc = float(roc_auc_score(test_target, test_probability))

    return {
        "probability": probability,
        "brier": brier,
        "roc_auc": auc,
        "mode": "LOGIT_70_COMPARABLES_30",
    }


def predict_next_gap(features: pd.DataFrame) -> GapPrediction:
    latest = features.dropna(subset=["smart_money_pct", "atr_pct"]).iloc[-1]
    modeling = features.iloc[:-1].dropna(
        subset=["target_gap_up_next", "target_gap_down_next"]
    )
    if len(modeling) < 180:
        raise ValueError("Se requieren al menos 180 sesiones maduras para entrenar.")

    latest_frame = pd.DataFrame([latest])
    up = _fit_one(modeling, latest_frame, "target_gap_up_next")
    down = _fit_one(modeling, latest_frame, "target_gap_down_next")
    comparable_up = _comparable_probability(modeling, latest, "target_gap_up_next")
    comparable_down = _comparable_probability(modeling, latest, "target_gap_down_next")

    probability_up = (
        comparable_up
        if pd.isna(up["probability"])
        else 0.70 * up["probability"] + 0.30 * comparable_up
    )
    probability_down = (
        comparable_down
        if pd.isna(down["probability"])
        else 0.70 * down["probability"] + 0.30 * comparable_down
    )

    total = probability_up + probability_down
    if total > 0.95:
        scale = 0.95 / total
        probability_up *= scale
        probability_down *= scale

    base_up = float(modeling["target_gap_up_next"].mean())
    base_down = float(modeling["target_gap_down_next"].mean())
    return GapPrediction(
        probability_up=float(probability_up),
        probability_down=float(probability_down),
        probability_no_gap=float(1 - probability_up - probability_down),
        base_up=base_up,
        base_down=base_down,
        lift_up=float(probability_up / base_up) if base_up else np.nan,
        lift_down=float(probability_down / base_down) if base_down else np.nan,
        brier_up=float(up["brier"]),
        brier_down=float(down["brier"]),
        roc_auc_up=float(up["roc_auc"]),
        roc_auc_down=float(down["roc_auc"]),
        training_rows=len(modeling),
        historical_up_cases=int(modeling["target_gap_up_next"].sum()),
        historical_down_cases=int(modeling["target_gap_down_next"].sum()),
        mode_up=str(up["mode"]),
        mode_down=str(down["mode"]),
    )

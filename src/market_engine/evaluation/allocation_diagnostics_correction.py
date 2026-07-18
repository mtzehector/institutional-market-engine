from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from market_engine.evaluation.allocation_calibration import (
    AllocationCalibrationResult,
    run_allocation_calibration_laboratory,
)
from market_engine.evaluation.market_memory import build_date_champion_quality


FINALIST_POLICIES = ("BOUNDED_ADAPTIVE", "LOGISTIC", "PIECEWISE")

CHAMPION_FAMILIES: dict[str, str] = {
    "CHAMPION_VOLUME_CONFIDENCE_STRUCTURE": "VOLUME_STRUCTURE",
    "PARSIMONIOUS_MOMENTUM_VOLUME": "MOMENTUM_VOLUME",
    "INSTITUTIONAL_STABLE": "INSTITUTIONAL",
    "INSTITUTIONAL_CONFIRMED": "INSTITUTIONAL",
    "CONTROL_VOLUME": "VOLUME_STRUCTURE",
    "CONTROL_MOMENTUM": "MOMENTUM_VOLUME",
    "CONTROL_SMART_MONEY": "INSTITUTIONAL",
}

WARMUP_POLICIES = (
    "SIMILARITY_FALLBACK",
    "FIXED_60",
    "EXPANDING_THRESHOLD",
)


@dataclass(frozen=True)
class AllocationDiagnosticsCorrectionResult:
    recommendations: pd.DataFrame
    policy_comparison: pd.DataFrame
    diagnostics_by_date: pd.DataFrame
    family_disagreement: pd.DataFrame
    warmup_comparison: pd.DataFrame
    conviction_validation: pd.DataFrame
    summary: pd.DataFrame
    calibration_result: AllocationCalibrationResult


def _numeric(value: object, default: float = 0.0) -> float:
    return float(value) if pd.notna(value) else default


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or abs(denominator) <= 1e-12:
        return 0.0
    return float(numerator / abs(denominator))


def _causal_percentile_rank(history: pd.Series, value: object) -> float:
    numeric = pd.to_numeric(history, errors="coerce").dropna()
    current = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if numeric.empty or pd.isna(current):
        return 0.5
    return float((numeric <= current).mean())


def _family(champion: object) -> str:
    return CHAMPION_FAMILIES.get(str(champion), "OTHER")


def _family_distance(left: str, right: str) -> float:
    if left == right:
        return 0.0
    partial_pairs = {
        frozenset({"VOLUME_STRUCTURE", "MOMENTUM_VOLUME"}),
        frozenset({"VOLUME_STRUCTURE", "INSTITUTIONAL"}),
    }
    return 0.5 if frozenset({left, right}) in partial_pairs else 1.0


def _unique_memory_allocations(base_allocations: pd.DataFrame) -> pd.DataFrame:
    """Return one canonical MEMORY_ONLY row per date and champion."""
    base = base_allocations.loc[base_allocations["policy"] == "MEMORY_ONLY"].copy()
    if base.empty:
        raise ValueError("No existe la política MEMORY_ONLY en la asignación base")
    return (
        base.sort_values(["target_date", "champion", "memory_score"], ascending=[True, True, False])
        .drop_duplicates(["target_date", "champion"], keep="first")
        .reset_index(drop=True)
    )


def _build_corrected_diagnostics(base_allocations: pd.DataFrame) -> pd.DataFrame:
    base = _unique_memory_allocations(base_allocations)
    rows: list[dict[str, object]] = []
    raw_current_margins: list[float] = []

    for target_date, sample in base.groupby("target_date", sort=True):
        memory_order = sample.sort_values("memory_score", ascending=False).reset_index(drop=True)
        current_order = sample.sort_values(
            "current_intelligence_score", ascending=False
        ).reset_index(drop=True)

        memory_top = memory_order.iloc[0]
        memory_second = memory_order.iloc[1] if len(memory_order) > 1 else memory_top
        current_top = current_order.iloc[0]
        current_second = current_order.iloc[1] if len(current_order) > 1 else current_top

        memory_top_score = _numeric(memory_top.get("memory_score"))
        memory_second_score = _numeric(memory_second.get("memory_score"))
        memory_margin = max(memory_top_score - memory_second_score, 0.0)
        memory_margin_ratio = np.clip(
            _safe_ratio(memory_margin, memory_top_score), 0.0, 1.0
        )
        memory_certainty = 100.0 * memory_margin_ratio

        current_top_score = _numeric(current_top.get("current_intelligence_score"))
        current_second_score = _numeric(current_second.get("current_intelligence_score"))
        current_margin = max(current_top_score - current_second_score, 0.0)
        history = pd.Series(raw_current_margins, dtype=float)
        current_consistency = 100.0 * _causal_percentile_rank(history, current_margin)
        raw_current_margins.append(current_margin)

        memory_champion = str(memory_top["champion"])
        current_champion = str(current_top["champion"])
        memory_family = _family(memory_champion)
        current_family = _family(current_champion)
        family_disagreement = 100.0 * _family_distance(memory_family, current_family)

        score_vectors = sample[
            ["memory_score_percentile", "current_intelligence_score"]
        ].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        vector_disagreement = float(
            (
                score_vectors["memory_score_percentile"]
                - score_vectors["current_intelligence_score"]
            ).abs().mean()
        )
        corrected_disagreement = float(
            np.clip(
                0.55 * family_disagreement + 0.45 * vector_disagreement,
                0.0,
                100.0,
            )
        )

        rows.append(
            {
                "target_date": target_date,
                "memory_top_champion": memory_champion,
                "current_top_champion": current_champion,
                "memory_top_family": memory_family,
                "current_top_family": current_family,
                "memory_top_score": memory_top_score,
                "memory_second_score": memory_second_score,
                "memory_real_margin": memory_margin,
                "memory_margin_ratio": memory_margin_ratio,
                "memory_certainty_corrected": memory_certainty,
                "memory_ambiguity_corrected": 100.0 - memory_certainty,
                "current_top_score": current_top_score,
                "current_second_score": current_second_score,
                "current_raw_margin": current_margin,
                "current_consistency_causal": current_consistency,
                "family_disagreement": family_disagreement,
                "vector_disagreement": vector_disagreement,
                "memory_current_disagreement_corrected": corrected_disagreement,
                "top_families_agree": bool(memory_family == current_family),
            }
        )
    return pd.DataFrame(rows)


def _warmup_weight(row: pd.Series, policy: str) -> float:
    threshold = _numeric(row.get("novelty_veto_threshold"), np.nan)
    novelty = max(_numeric(row.get("global_novelty_score"), 0.0), 0.0)
    similarity = np.clip(
        _numeric(row.get("similarity_confidence"), 50.0) / 100.0, 0.0, 1.0
    )

    if pd.notna(threshold) and threshold > 1e-12:
        ratio = novelty / threshold
        return float(
            np.clip(
                0.50 * (1.0 - 0.55 * min(ratio, 1.8)) + 0.50 * similarity,
                0.15,
                0.90,
            )
        )
    if policy == "SIMILARITY_FALLBACK":
        return float(np.clip(similarity, 0.15, 0.90))
    if policy == "FIXED_60":
        return 0.60
    if policy == "EXPANDING_THRESHOLD":
        expanding = _numeric(row.get("expanding_novelty_threshold"), np.nan)
        if pd.notna(expanding) and expanding > 1e-12:
            return float(np.clip(1.0 - 0.55 * novelty / expanding, 0.15, 0.90))
        return 0.60
    raise ValueError(f"Política warm-up desconocida: {policy}")


def _date_thresholds(base: pd.DataFrame) -> pd.DataFrame:
    """Calculate one causal expanding novelty threshold per date."""
    dates = (
        base[["target_date", "global_novelty_score"]]
        .drop_duplicates("target_date")
        .sort_values("target_date")
        .reset_index(drop=True)
    )
    thresholds: list[float] = []
    novelty = pd.to_numeric(dates["global_novelty_score"], errors="coerce")
    for index in range(len(dates)):
        history = novelty.iloc[:index].dropna()
        thresholds.append(float(history.quantile(0.80)) if len(history) >= 2 else np.nan)
    dates["expanding_novelty_threshold"] = thresholds
    return dates[["target_date", "expanding_novelty_threshold"]]


def _recompute_recommendations(
    base_allocations: pd.DataFrame,
    diagnostics: pd.DataFrame,
    date_quality: pd.DataFrame,
) -> pd.DataFrame:
    base = _unique_memory_allocations(base_allocations)
    base = base.merge(diagnostics, on="target_date", how="left")
    base = base.merge(_date_thresholds(base), on="target_date", how="left")

    frames: list[pd.DataFrame] = []
    for warmup_policy in WARMUP_POLICIES:
        sample = base.copy()
        sample["warmup_policy"] = warmup_policy
        sample["memory_weight_corrected"] = sample.apply(
            lambda row: _warmup_weight(row, warmup_policy), axis=1
        )
        sample["current_intelligence_weight_corrected"] = (
            1.0 - sample["memory_weight_corrected"]
        )
        sample["allocation_score_corrected"] = (
            sample["memory_weight_corrected"] * sample["memory_score_percentile"]
            + sample["current_intelligence_weight_corrected"]
            * sample["current_intelligence_score"]
        )
        sample["allocation_rank_corrected"] = sample.groupby("target_date")[
            "allocation_score_corrected"
        ].rank(ascending=False, method="first")
        frames.append(sample)

    allocations = pd.concat(frames, ignore_index=True)
    winners = allocations.loc[allocations["allocation_rank_corrected"] == 1].copy()

    quality = date_quality.rename(
        columns={"origin_date": "target_date", "quality_score": "quality_score_actual"}
    )
    oracle = (
        quality.sort_values(
            ["target_date", "quality_score_actual"], ascending=[True, False]
        )
        .groupby("target_date", as_index=False)
        .first()[["target_date", "champion", "quality_score_actual"]]
        .rename(
            columns={
                "champion": "oracle_champion",
                "quality_score_actual": "oracle_quality",
            }
        )
    )
    universe = quality.loc[
        quality["champion"] == "UNIVERSE",
        ["target_date", "quality_score_actual"],
    ].rename(columns={"quality_score_actual": "universe_quality"})

    result = winners.merge(oracle, on="target_date", how="left").merge(
        universe, on="target_date", how="left"
    )
    if result["universe_quality"].isna().all():
        raise ValueError("No se pudo recuperar la calidad de UNIVERSE desde Selecciones")

    result["advantage_vs_universe"] = (
        result["actual_quality_score"] - result["universe_quality"]
    )
    result["oracle_regret"] = result["oracle_quality"] - result["actual_quality_score"]
    result["selected_was_oracle"] = result["champion"] == result["oracle_champion"]

    agreement = 100.0 - pd.to_numeric(
        result["memory_current_disagreement_corrected"], errors="coerce"
    ).clip(0, 100)
    strength = pd.to_numeric(
        result["allocation_score_corrected"], errors="coerce"
    ).clip(0, 100)
    novelty_confidence = 100.0 / (
        1.0
        + pd.to_numeric(result["global_novelty_score"], errors="coerce").clip(lower=0)
    )
    memory_certainty = pd.to_numeric(
        result["memory_certainty_corrected"], errors="coerce"
    ).clip(0, 100)
    current_consistency = pd.to_numeric(
        result["current_consistency_causal"], errors="coerce"
    ).clip(0, 100)
    result["adaptive_conviction_index_corrected"] = (
        0.30 * strength
        + 0.25 * agreement
        + 0.15 * novelty_confidence
        + 0.15 * memory_certainty
        + 0.15 * current_consistency
    ).clip(0, 100)

    result = result.rename(columns={"champion": "recommended_champion"})
    if result.columns.duplicated().any():
        duplicates = result.columns[result.columns.duplicated()].tolist()
        raise ValueError(f"Columnas duplicadas en recomendaciones corregidas: {duplicates}")
    return result.sort_values(["target_date", "warmup_policy"]).reset_index(drop=True)


def _compare_warmup_policies(recommendations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for policy, sample in recommendations.groupby("warmup_policy", sort=False):
        advantage = pd.to_numeric(sample["advantage_vs_universe"], errors="coerce")
        regret = pd.to_numeric(sample["oracle_regret"], errors="coerce")
        rows.append(
            {
                "warmup_policy": policy,
                "dates": int(len(sample)),
                "mean_memory_weight": float(sample["memory_weight_corrected"].mean()),
                "mean_advantage_vs_universe": float(advantage.mean()),
                "positive_advantage_rate": float((advantage > 0).mean()),
                "mean_oracle_regret": float(regret.mean()),
                "maximum_oracle_regret": float(regret.max()),
                "oracle_hit_rate": float(sample["selected_was_oracle"].mean()),
                "mean_aci_corrected": float(
                    sample["adaptive_conviction_index_corrected"].mean()
                ),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["diagnostics_policy_score"] = 100.0 * (
        0.30 * result["positive_advantage_rate"].fillna(0.0)
        + 0.25 * (result["mean_advantage_vs_universe"].clip(-50, 50) + 50) / 100
        + 0.20 * (1.0 - result["mean_oracle_regret"].clip(0, 100) / 100)
        + 0.15 * (1.0 - result["maximum_oracle_regret"].clip(0, 100) / 100)
        + 0.10 * result["oracle_hit_rate"].fillna(0.0)
    )
    return result.sort_values("diagnostics_policy_score", ascending=False).reset_index(drop=True)


def _validate_conviction(recommendations: pd.DataFrame) -> pd.DataFrame:
    data = recommendations.copy()
    data["conviction_bucket"] = pd.qcut(
        pd.to_numeric(data["adaptive_conviction_index_corrected"], errors="coerce"),
        q=min(4, max(1, data["adaptive_conviction_index_corrected"].nunique())),
        labels=False,
        duplicates="drop",
    )
    return (
        data.groupby(["warmup_policy", "conviction_bucket"], dropna=False)
        .agg(
            dates=("target_date", "count"),
            mean_aci=("adaptive_conviction_index_corrected", "mean"),
            mean_advantage_vs_universe=("advantage_vs_universe", "mean"),
            positive_advantage_rate=(
                "advantage_vs_universe",
                lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean()),
            ),
            mean_oracle_regret=("oracle_regret", "mean"),
            oracle_hit_rate=("selected_was_oracle", "mean"),
        )
        .reset_index()
    )


def run_allocation_diagnostics_correction_laboratory(
    daily_states: pd.DataFrame,
    selections: pd.DataFrame,
    *,
    ks: tuple[int, ...] = (3, 5, 7, 10),
    baseline_k: int = 5,
    minimum_history: int = 10,
    calibration_history: int = 6,
    novelty_percentile: float = 0.80,
) -> AllocationDiagnosticsCorrectionResult:
    calibration = run_allocation_calibration_laboratory(
        daily_states,
        selections,
        ks=ks,
        baseline_k=baseline_k,
        minimum_history=minimum_history,
        calibration_history=calibration_history,
        novelty_percentile=novelty_percentile,
    )

    base_allocations = calibration.allocation_by_date
    diagnostics = _build_corrected_diagnostics(base_allocations)
    date_quality = build_date_champion_quality(selections)
    recommendations = _recompute_recommendations(
        base_allocations, diagnostics, date_quality
    )
    warmup_comparison = _compare_warmup_policies(recommendations)
    conviction_validation = _validate_conviction(recommendations)

    policy_comparison = calibration.policy_comparison.loc[
        calibration.policy_comparison["calibration_policy"].isin(FINALIST_POLICIES)
    ].copy()
    summary = warmup_comparison.head(1).copy()
    if not summary.empty:
        summary = summary.rename(columns={"warmup_policy": "recommended_warmup_policy"})

    return AllocationDiagnosticsCorrectionResult(
        recommendations=recommendations,
        policy_comparison=policy_comparison,
        diagnostics_by_date=diagnostics,
        family_disagreement=diagnostics[
            [
                "target_date",
                "memory_top_champion",
                "current_top_champion",
                "memory_top_family",
                "current_top_family",
                "family_disagreement",
                "vector_disagreement",
                "memory_current_disagreement_corrected",
                "top_families_agree",
            ]
        ].copy(),
        warmup_comparison=warmup_comparison,
        conviction_validation=conviction_validation,
        summary=summary,
        calibration_result=calibration,
    )

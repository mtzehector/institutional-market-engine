# v0.9.7 — Champion Survival Engine

## Purpose

Estimate how likely each champion is to remain in its current stabilized lifecycle state for the next evaluation horizons.

The engine does not use future observations when producing a historical estimate. For each date it uses only state spells that had already ended before that date.

## Core concepts

- `state_age`: number of consecutive synchronized observations in the current lifecycle state.
- `survival_probability_N`: empirical probability of remaining in the same state for at least N more observations, conditional on having already survived the current age.
- `survival_reference_scope`: champion-specific history when enough completed spells exist; otherwise pooled history for the same lifecycle state.
- `survival_confidence_score`: combines empirical persistence, lifecycle health, and current direction.
- `survival_adjusted_authority`: persistence probability multiplied by confidence.

## Output sheets

- `Resumen`
- `Supervivencia_Actual`
- `Historial_Supervivencia`
- `Episodios_Estado`
- `Curvas_Supervivencia`
- `Ciclo_Actual`

## Interpretation

The output is an experimental research signal, not a probability guarantee or an automatic trading instruction. Small numbers of completed spells are smoothed with a neutral prior and are explicitly reported through support columns.

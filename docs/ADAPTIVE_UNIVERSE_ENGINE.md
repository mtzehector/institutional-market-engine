# Adaptive Universe Engine — v0.8.3

## Research question

Does a daily, leakage-safe cohort of bullish tickers reduce predictive uncertainty relative to evaluating the full universe indiscriminately?

The engine separates two responsibilities:

1. **Universe construction:** rank tickers at each close using only information available on that origin date.
2. **Conditional prediction:** compare the existing gap model inside Top 5, Top 10 and Top 20 cohorts against the complete universe.

## Leakage control

The bullish score is computed from the origin-date close and is matched with the prediction for the following session. No future return or target value participates in selection.

## Bullish score

The first auditable specification combines cross-sectional percentiles:

- 35% momentum: returns over 1, 3 and 5 sessions;
- 20% participation: relative volume and close position;
- 25% institutional condition: Smart Money level and five-session slope;
- 20% structure/context: distance from VWAP and QQQ return;
- mild penalty for the most extreme ATR values.

Weights are research parameters, not final truths. They must be validated outside sample.

## Outputs

The Excel report contains:

- `Comparacion_Cohortes`: Rare Event Evaluation metrics for Top 5, Top 10, Top 20 and Universe;
- `Selecciones_Diarias`: daily score, rank and score components;
- `Predicciones_Cohorte`: complete auditable predictions, duplicated only across the cohorts to which each observation belongs;
- `Diagnostico_Diario`: daily universe size and score dispersion;
- `Errores`: ticker-level failures.

## Success criteria

A reduced cohort is not considered better merely because its scores are more homogeneous. It must demonstrate improvements such as:

- higher balanced accuracy;
- higher macro F1 and rare-event F1;
- better Brier Skill;
- lower calibration error;
- stable performance across dates;
- economically favorable subsequent behavior.

## Example

```bat
market-adaptive-universe ^
  --tickers config\lista_completa_nasdaq.txt ^
  --from-date 2026-01-02 ^
  --to-date 2026-07-16 ^
  --cohort-sizes 5,10,20 ^
  --memory-input reports\baseline\adaptive_memory_2026.xlsx ^
  --memory-sheet Mejor_Memoria ^
  --step 5 ^
  --export ^
  --output reports\adaptive_universe\adaptive_universe_v083.xlsx
```

For the first validation, use `step 5`. After confirming the pipeline, rerun promising cohort sizes with `step 1`.

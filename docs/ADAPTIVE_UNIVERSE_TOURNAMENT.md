# Adaptive Universe Tournament — v0.8.4

## Research question

Does the way the daily universe is constructed materially change out-of-sample uncertainty and rare-event detection?

The tournament compares several leakage-safe universe constructors over the same dates, tickers and model predictions. The predictor is not retrained differently for each strategy. Only the daily selection rule changes.

## Competitors

- `MOMENTUM`: ranks recent 1, 3 and 5-session returns.
- `VOLUME`: ranks relative volume and close position inside the daily range.
- `SMART_MONEY`: ranks Smart Money level and five-session slope.
- `MODEL_CONFIDENCE`: ranks the largest predicted class probability available at the close.
- `LOW_VOLATILITY`: favors lower cross-sectional ATR percent.
- `HYBRID`: combines momentum, volume, Smart Money, model confidence, low volatility and price structure, with a mild overextension penalty.
- `UNIVERSE`: control group containing every available prediction.

Each strategy is evaluated at Top 5, Top 10 and Top 20 by default.

## Scientific safeguards

1. All rankings are calculated independently for each `origin_date`.
2. Only values available at the close of `origin_date` are used.
3. The v0.8.3 workbook contains duplicated observations across cohorts. The tournament reads only `UNIVERSE` rows before constructing new competitors.
4. Institutional inertia is not included yet because the current regime report summarizes full periods. Using those summaries on earlier dates would leak future information.
5. The predictor remains unchanged, isolating universe selection as the experimental variable.

## Tournament metrics

The leaderboard combines percentile ranks of:

- Rare Event F1 — higher is better.
- Balanced Accuracy — higher is better.
- Macro F1 — higher is better.
- Mean Brier Skill — higher is better.
- Mean Calibration Error — lower is better.

The resulting `tournament_score` ranges from 0 to 100 and is intended for comparison inside this experiment. It is not a probability or expected return.

## Command

```bat
market-universe-tournament ^
  --input reports\adaptive_universe\adaptive_universe_v083_step5.xlsx ^
  --sheet Predicciones_Cohorte ^
  --cohort-sizes 5,10,20 ^
  --export ^
  --output reports\adaptive_universe\adaptive_universe_tournament_v084.xlsx
```

## Workbook sheets

- `Leaderboard`: ordered tournament results and component scores.
- `Metricas`: complete evaluation metrics for every strategy and cohort size.
- `Scores_Diarios`: causal daily scores and ranks for every ticker.
- `Predicciones_Torneo`: complete selected observations and realized outcomes.
- `Diagnostico_Diario`: daily universe size and score dispersion.

## Interpretation

A strategy should not be declared superior because of accuracy alone. Prefer competitors that improve several dimensions simultaneously and retain enough dates and rare events.

A useful result may also be negative. If all specialized universes underperform `UNIVERSE`, the evidence would indicate that the current selection rules remove useful diversity or select exhausted moves.

## Next causal extension

The next tournament round may include institutional inertia after regime duration, confirmed crossings and volume-backed persistence are available as daily rolling features rather than full-period summaries.

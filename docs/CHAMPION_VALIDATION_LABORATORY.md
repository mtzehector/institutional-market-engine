# Champion Validation Laboratory — v0.9.2

The Champion Validation Laboratory evaluates the finalist universe constructors across successive calendar windows.

## Purpose

The objective is not to find the highest aggregate score over the complete period. It is to determine which family remains useful when market conditions change.

The laboratory compares four finalist families:

- `VOLUME + MODEL_CONFIDENCE + STRUCTURE`, Top 20.
- `MOMENTUM + VOLUME`, Top 20.
- `SMART_MONEY + LOW_VOLATILITY`, Top 10.
- `VOLUME + SMART_MONEY + MODEL_CONFIDENCE`, Top 10.

Controls:

- Volume, Top 20.
- Momentum, Top 20.
- Smart Money, Top 20.
- Full universe.

## Temporal windows

The default frequency is monthly. Each observation is assigned according to `origin_date`, so all selection variables remain known at the close of that session.

Quarterly analysis is available with `--frequency Q` when monthly samples are too small.

## Outputs

### Ranking_Estabilidad

Ranks the families by mean temporal quality minus a variability penalty.

### Metricas_Ventanas

Contains Rare Event F1, balanced accuracy, Macro F1, Brier Skill, calibration and predictability score for every family and window.

### Degradacion

Measures the change from each window to the next. This sheet is intended to reveal abrupt deterioration during a regime transition.

### Ventanas

Documents the effective date range and number of observed sessions in every window.

### Selecciones

Preserves the selected ticker observations for full auditability.

## Interpretation

A strong champion should not merely win on average. It should:

- preserve acceptable minimum performance;
- avoid extreme month-to-month dispersion;
- maintain useful rare-event detection;
- produce positive probabilistic skill in multiple windows;
- degrade less than the full universe during adverse regimes.

The stability score is a research ranking, not a probability, return forecast or trading recommendation.

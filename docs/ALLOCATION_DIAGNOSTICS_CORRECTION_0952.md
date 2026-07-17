# v0.9.5.2 — Allocation Diagnostics Correction

## Purpose

This laboratory corrects four weaknesses detected in v0.9.5.1 before new intelligence sources are introduced:

1. Memory ambiguity is calculated from real memory scores instead of percentile-rank spacing.
2. Memory/current disagreement is measured by intelligence family as well as champion name.
3. Current-intelligence consistency is normalized causally against prior margins.
4. Warm-up behavior is tested explicitly instead of silently falling back to one rule.

## Finalist curves retained

The report preserves the three strongest v0.9.5.1 curves for reference:

- `BOUNDED_ADAPTIVE`
- `LOGISTIC`
- `PIECEWISE`

## Warm-up policies compared

- `SIMILARITY_FALLBACK`: uses similarity before a novelty threshold exists.
- `FIXED_60`: gives memory 60% authority during warm-up.
- `EXPANDING_THRESHOLD`: starts calibrating novelty after two prior observations.

Once the causal novelty threshold is available, every warm-up policy uses the same bounded adaptive rule. Therefore differences isolate the start-up phase.

## Corrected diagnostics

### Real memory margin

The engine compares the first and second raw `memory_score` values and exports:

- `memory_real_margin`
- `memory_margin_ratio`
- `memory_certainty_corrected`
- `memory_ambiguity_corrected`

### Family disagreement

Champions are grouped into:

- `INSTITUTIONAL`
- `MOMENTUM_VOLUME`
- `VOLUME_STRUCTURE`

Exact family agreement has zero family disagreement. Related families receive partial disagreement, while unrelated families receive full disagreement.

### Current consistency

The margin between the first and second current-intelligence scores is ranked causally against prior margins. This yields `current_consistency_causal` on a 0–100 scale.

### Corrected conviction

`adaptive_conviction_index_corrected` combines decision strength, family-aware agreement, novelty confidence, real memory certainty, and causal current consistency.

## Command

```bat
market-allocation-diagnostics ^
  --input reports\adaptive_universe\regime_transition_v0931.xlsx ^
  --states-sheet Estados_Diarios ^
  --selections-sheet Selecciones ^
  --neighbors-grid 3,5,7,10 ^
  --baseline-neighbors 5 ^
  --minimum-history 10 ^
  --calibration-history 6 ^
  --novelty-percentile 0.80 ^
  --export ^
  --output reports\adaptive_universe\allocation_diagnostics_v0952.xlsx
```

## Main sheets

- `Comparacion_Warmup`
- `Curvas_Finalistas`
- `Recomendaciones`
- `Diagnosticos_Corregidos`
- `Desacuerdo_Familias`
- `Validacion_ACI`

## Evaluation questions

1. Does a better warm-up preserve the strong April memory decisions?
2. Does the corrected disagreement vary meaningfully across dates?
3. Does corrected ACI order advantage and regret better than the previous ACI?
4. Does the laboratory preserve the June and July corrections achieved by v0.9.5?

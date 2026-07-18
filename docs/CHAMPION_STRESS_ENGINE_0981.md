# v0.9.8.1 — Champion Stress Engine

## Objective

Measure the complete adverse trajectory of every champion instead of evaluating only whether it eventually recovered.

## Core metrics

- deterioration velocity
- maximum damage
- cumulative damage
- time to bottom
- stress volatility
- recovery efficiency
- recovery triggers
- relapse risk
- stress score and current stress-risk score

## Method

The engine reuses the stabilized lifecycle and resilience outputs. Every adverse episode is reconstructed from entry into `DETERIORATING` or `OBSOLETE` until recovery into a favorable state or the end of the available history.

The implementation is causal for lifecycle classification. Trigger and relapse reports are retrospective diagnostics and are not used to generate historical predictions.

## Interpretation

A resilient champion may still experience severe stress. The engine therefore separates:

- recovery capacity;
- severity of the fall;
- efficiency of the rebound;
- probability of relapse.

## Command

```bat
market-champion-stress ^
  --input reports\adaptive_universe\regime_transition_v0931.xlsx ^
  --states-sheet Estados_Diarios ^
  --selections-sheet Selecciones ^
  --short-window 3 ^
  --long-window 6 ^
  --minimum-history 4 ^
  --minimum-state-persistence 2 ^
  --minimum-own-episodes 3 ^
  --minimum-family-episodes 5 ^
  --trigger-lookback 2 ^
  --relapse-horizon 3 ^
  --export ^
  --output reports\adaptive_universe\champion_stress_v0981.xlsx
```

## Research caution

The number of completed adverse episodes is initially small. Scores are exploratory and must not be treated as calibrated probabilities or automatic investment instructions.

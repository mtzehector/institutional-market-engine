# v0.9.8 — Champion Resilience Engine

## Objective

Measure how often, how quickly and how solidly each champion recovers after entering an adverse life-cycle state.

The engine is causal: every historical resilience estimate uses only recovery episodes completed before the evaluated date.

## Main concepts

- **Recovery probability:** empirical, smoothed frequency of recovery from adverse episodes.
- **Full recovery probability:** return to `MATURE` with non-negative advantage over the universe.
- **Adaptation speed:** inverse transformation of median recovery duration.
- **Recovery depth:** loss in advantage accumulated during the adverse episode.
- **Post-recovery persistence:** observations spent in favorable states after recovery.
- **Recovery quality:** combination of full recovery, restoration of prior quality and persistence.
- **Resilience score:** exploratory 0–100 composite of probability, speed, quality and low damage.

## Evidence hierarchy

1. `CHAMPION` when enough own episodes exist.
2. `FAMILY` when the champion lacks history but its strategic family has enough episodes.
3. `POOLED` as the final fallback.

Every output exposes the reference scope and `LOW`, `MEDIUM` or `HIGH` evidence strength.

## Command

```bat
market-champion-resilience ^
  --input reports\adaptive_universe\regime_transition_v0931.xlsx ^
  --states-sheet Estados_Diarios ^
  --selections-sheet Selecciones ^
  --short-window 3 ^
  --long-window 6 ^
  --minimum-history 4 ^
  --minimum-state-persistence 2 ^
  --minimum-own-episodes 3 ^
  --minimum-family-episodes 5 ^
  --export ^
  --output reports\adaptive_universe\champion_resilience_v098.xlsx
```

## Interpretation

This laboratory does not predict an exact future recovery date. It evaluates whether a champion has historically demonstrated an ability to regain usefulness after deterioration, how costly that process was and how much evidence supports the conclusion.

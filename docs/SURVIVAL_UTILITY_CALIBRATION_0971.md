# v0.9.7.1 — Survival Utility Calibration

## Purpose

The Champion Survival Engine estimates how likely a champion is to remain in its current lifecycle state. This correction separates persistence from desirability: a deteriorating state can be highly persistent without deserving higher authority.

## Main additions

- Strategic utility by lifecycle state.
- Favorable-state survival versus adverse-state persistence.
- Short-term and medium-term authority.
- Wilson uncertainty interval for the longest configured horizon.
- Evidence strength (`LOW`, `MEDIUM`, `HIGH`) based on support, completed spells and reference scope.
- Utility-adjusted authority with an explicit penalty for persistent adverse states.

## Initial state utility

| Lifecycle state | Utility |
|---|---:|
| MATURE | 1.00 |
| RECOVERING | 0.80 |
| EMERGING | 0.60 |
| DISCOVERY | 0.35 |
| DETERIORATING | 0.15 |
| OBSOLETE | 0.00 |

These values are experimental and must be evaluated with future snapshots.

## Command

```bat
market-survival-utility ^
  --input reports\adaptive_universe\regime_transition_v0931.xlsx ^
  --states-sheet Estados_Diarios ^
  --selections-sheet Selecciones ^
  --short-window 3 ^
  --long-window 6 ^
  --minimum-history 4 ^
  --minimum-state-persistence 2 ^
  --horizons 1,3,5 ^
  --minimum-completed-spells 3 ^
  --export ^
  --output reports\adaptive_universe\champion_survival_utility_v0971.xlsx
```

## Interpretation

- `favorable_state_survival`: persistence that supports continued authority.
- `adverse_state_persistence`: persistence that supports caution or reduced authority.
- `short_term_authority`: combines health, utility and survival over one to three observations.
- `medium_term_authority`: combines health, utility and survival over the longest configured horizon.
- `utility_adjusted_authority`: blended authority after adverse-persistence penalty.
- `evidence_strength`: strength of the empirical support, not certainty of future performance.

The output remains a research artifact and is not an automated investment instruction.

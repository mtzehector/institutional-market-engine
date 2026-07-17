# Regime Transition Laboratory — v0.9.3

## Purpose

The Regime Transition Laboratory tests whether the preferred universe constructor should change when the market environment changes.

It does not yet choose a live strategy automatically. It first creates an auditable map of:

- daily market states;
- detected transitions;
- champion performance by regime;
- deterioration or improvement around transitions.

## Causal design

Every regime label uses information available at the close of `origin_date` and references only prior observations through shifted rolling baselines. Future observations are not used to classify an earlier date.

## Daily state variables

- QQQ one-day return;
- one-day positive breadth;
- five-day positive breadth;
- median relative volume;
- median ATR percentage;
- institutional breadth, defined as Smart Money above 50% with a positive five-day slope.

## Initial regimes

- `RISK_ON_EUPHORIA`
- `RISK_ON_BROAD`
- `QUIET_ACCUMULATION`
- `ROTATION_VOLATILE`
- `RISK_OFF_STRESS`
- `NEUTRAL`

These labels are research hypotheses, not permanent market truths. Their thresholds should be revised only after reviewing the exported evidence.

## Champion families

The laboratory evaluates the same finalist families defined in v0.9.2 plus the universal control. This isolates the effect of regime rather than changing the competing strategies.

## Main outputs

### Estados_Diarios

The regime assigned to each origin date, all market-state measurements, causal references and transition strength.

### Transiciones

Each change from one regime to another, its date, duration of the prior regime and state measurements at the transition.

### Ranking_por_Regimen

The best champion inside each regime according to rare-event detection, balanced accuracy, macro F1, Brier skill and calibration.

### Impacto_Transicion

Pre/post comparison around each transition. This is the principal sheet for identifying which champion deteriorated first and which one resisted the new environment.

## Interpretation discipline

A champion ranked first in a regime is only a promising specialist when:

- the regime has enough dates and rare events;
- performance is not driven by a single transition;
- probability calibration remains acceptable;
- the advantage survives later unseen dates.

The v0.9.3 result should guide a later causal champion selector, not be treated as an automatic trading rule.

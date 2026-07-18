# Champion Life Cycle Engine — v0.9.6

## Purpose

The Champion Life Cycle Engine tracks how every champion evolves through time instead of treating it as a permanent static winner.

Lifecycle states:

- `DISCOVERY`: insufficient causal history.
- `EMERGING`: improving and beginning to outperform.
- `MATURE`: stable positive advantage with controlled drawdown.
- `DETERIORATING`: recent loss of advantage, negative slope, or meaningful drawdown.
- `OBSOLETE`: persistent underperformance and deep deterioration.
- `RECOVERING`: returns to positive advantage after deterioration or obsolescence.

## Causal metrics

For each champion and date, the engine uses only observations available up to that date:

- quality score;
- advantage versus `UNIVERSE`;
- short- and long-window advantage;
- advantage slope;
- quality volatility;
- positive advantage rate;
- drawdown from historical quality peak;
- consecutive underperformance;
- lifecycle health score.

## Operational interpretation

The current state maps to a research action:

- `MATURE` → `ACTIVE`
- `RECOVERING` → `INCREASE_GRADUALLY`
- `EMERGING` → `MONITOR_AND_TEST`
- `DISCOVERY` → `OBSERVE`
- `DETERIORATING` → `REDUCE_AUTHORITY`
- `OBSOLETE` → `SUSPEND`

These actions are research labels, not automated trading instructions.

## Command

```bat
market-champion-lifecycle ^
  --input reports\adaptive_universe\regime_transition_v0931.xlsx ^
  --states-sheet Estados_Diarios ^
  --selections-sheet Selecciones ^
  --short-window 3 ^
  --long-window 6 ^
  --minimum-history 4 ^
  --export ^
  --output reports\adaptive_universe\champion_lifecycle_v096.xlsx
```

## Output sheets

- `Resumen`
- `Estado_Actual`
- `Historial_Ciclo`
- `Transiciones`
- `Desempeno_Regimen`

## Scientific question

The laboratory asks not only which champion is best today, but whether its intelligence is emerging, mature, deteriorating, obsolete, or recovering.

# v0.9.5.1 — Allocation Calibration Laboratory

## Propósito

Calibrar la autoridad relativa entre memoria histórica e inteligencia del estado actual. La versión compara cinco curvas de asignación sin modificar el predictor ni añadir nuevos factores.

## Curvas evaluadas

- `LINEAR`
- `LOGISTIC`
- `EXPONENTIAL`
- `PIECEWISE`
- `BOUNDED_ADAPTIVE`

Cada curva transforma la relación entre novedad observada y umbral causal de novedad en un peso de memoria entre 0 y 1.

## Diagnósticos nuevos

- `memory_current_disagreement`
- `memory_internal_ambiguity`
- `current_intelligence_consistency`
- `adaptive_conviction_index`

El ACI no representa probabilidad de éxito. Es un índice comparativo de solidez de la decisión combinada.

## Criterios de evaluación

Cada curva se compara por:

- ventaja media frente al universo;
- tasa de ventaja positiva;
- arrepentimiento medio frente al oráculo;
- arrepentimiento máximo;
- acierto exacto del oráculo;
- peso medio, mínimo y máximo de memoria.

## Ejecución

```bat
market-allocation-calibration ^
  --input reports\adaptive_universe\regime_transition_v0931.xlsx ^
  --states-sheet Estados_Diarios ^
  --selections-sheet Selecciones ^
  --neighbors-grid 3,5,7,10 ^
  --baseline-neighbors 5 ^
  --minimum-history 10 ^
  --calibration-history 6 ^
  --novelty-percentile 0.80 ^
  --export ^
  --output reports\adaptive_universe\allocation_calibration_v0951.xlsx
```

## Hojas de salida

- `Resumen`
- `Comparacion_Curvas`
- `Recomendaciones`
- `Calibration_Report`
- `Conviction_Report`
- `Disagreement_Report`
- `Allocation_Functions`

## Interpretación prudente

La curva ganadora será una hipótesis de asignación, no una regla definitiva. La muestra sincronizada sigue siendo pequeña y debe ampliarse con nuevas fechas antes de usar el ACI o los pesos para dimensionamiento de capital real.

# Regime Transition Synchronization — v0.9.3.1

La versión 0.9.3.1 corrige la principal limitación metodológica detectada en la primera ejecución del Regime Transition Laboratory: algunas fechas contenían una sola acción y eran tratadas como si representaran el universo completo.

## Principios

1. El régimen solo se calcula sobre fechas con cobertura transversal suficiente.
2. El tamaño máximo observado se usa como universo de referencia.
3. Una fecha es válida cuando cumple simultáneamente el tamaño mínimo absoluto y el porcentaje mínimo de cobertura.
4. Las transiciones requieren persistencia antes de ser confirmadas.
5. Las comparaciones antes y después de una transición utilizan únicamente fechas sincronizadas.

## Parámetros principales

- `--minimum-universe-size`: mínimo absoluto de tickers por fecha. Valor predeterminado: 80.
- `--minimum-coverage-ratio`: cobertura mínima respecto al máximo observado. Valor predeterminado: 0.70.
- `--minimum-regime-persistence`: fechas consecutivas necesarias para confirmar un nuevo régimen. Valor predeterminado: 2.

El umbral efectivo es:

```text
max(minimum_universe_size, ceil(reference_universe_size * minimum_coverage_ratio))
```

## Auditoría

El Excel incorpora la hoja `Auditoria_Cobertura`, que contiene:

- tamaño observado por fecha;
- universo de referencia;
- porcentaje de cobertura;
- umbral efectivo;
- indicador de fecha sincronizada;
- motivo de exclusión.

`Estados_Diarios` conserva tanto `raw_regime` como `regime`. El primero es la clasificación inmediata; el segundo es el régimen confirmado después de aplicar persistencia.

## Ejecución recomendada

```bat
market-regime-transition ^
  --input reports\adaptive_universe\adaptive_universe_tournament_v084.xlsx ^
  --sheet Predicciones_Torneo ^
  --lookback 12 ^
  --minimum-observations 10 ^
  --transition-radius 3 ^
  --minimum-universe-size 80 ^
  --minimum-coverage-ratio 0.70 ^
  --minimum-regime-persistence 2 ^
  --export ^
  --output reports\adaptive_universe\regime_transition_v0931.xlsx
```

## Interpretación

Una transición confirmada representa un cambio persistente en las variables transversales disponibles. No demuestra por sí sola causalidad ni garantiza rentabilidad. Su utilidad es estudiar si las familias de universos reaccionan de forma diferente ante cambios del entorno.

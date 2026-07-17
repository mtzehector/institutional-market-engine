# Market Memory Decision Laboratory — v0.9.4.2

Esta versión compara políticas de aceptación sobre las mismas recomendaciones de Market Memory.

## Objetivo

Reducir arrepentimientos extremos sin eliminar la mayoría de las oportunidades. La versión evita promediar todas las señales en un único score y prueba reglas de veto causales.

## Políticas

- `BASELINE_ACCEPT_ALL`: acepta todas las recomendaciones.
- `COMPOSITE_SCORE`: conserva la política de v0.9.4.1 como control.
- `NOVELTY_ONLY`: rechaza estados cuya novedad supera un percentil calculado solo con fechas anteriores.
- `NOVELTY_STABILITY`: añade veto por inestabilidad histórica del campeón.
- `HIERARCHICAL_VETO`: aplica en orden novedad, similitud y estabilidad.

## Umbrales causales

Los percentiles se recalculan en cada fecha usando únicamente observaciones previas. Las primeras fechas sin historia suficiente quedan como `INSUFFICIENT_CALIBRATION_HISTORY` para las políticas nuevas.

## Métricas

Cada política reporta cobertura, ventaja media y positiva frente al universo, arrepentimiento medio y máximo, acierto del oráculo y desempeño de las fechas rechazadas.

El `policy_score` es comparativo y combina cobertura, tasa de ventaja positiva, ventaja media y bajo arrepentimiento. No es una probabilidad ni una señal operativa.

# Experimento 42 — política transversal con exposición comparable

Estado: cohortes históricas solapadas; no cartera ejecutable ni recomendación.

## Política evaluada

Cada una de las 40 fechas landmark asigna 100% de su cohorte durante cinco sesiones:

- `CONFIRMED_OR_QQQ`: igual ponderación entre recuperaciones confirmadas; si no hay ninguna, QQQ.
- `ALL_OBSERVED`: igual ponderación entre todos los episodios observados de la fecha.
- `QQQ_ALWAYS`: QQQ durante el mismo intervalo.

Las cohortes no se capitalizan secuencialmente porque sus periodos se solapan. Hacerlo fingiría disponer varias veces del mismo capital.

## Resultados por fecha

| Política/métrica | Media | Mediana | IC 95% por fecha |
|---|---:|---:|---:|
| Confirmados o QQQ | +0,772% | +0,701% | [−0,498%, +2,136%] |
| Todos los observados | +0,775% | +0,034% | [−0,636%, +2,280%] |
| QQQ | +0,696% | +0,801% | [−0,221%, +1,618%] |
| Confirmados/QQQ menos todos | −0,004 pp | 0,000 pp | [−1,154, +1,148] |
| Confirmados/QQQ menos QQQ | +0,075 pp | 0,000 pp | [−1,077, +1,280] |

Hubo selección confirmada en 22 fechas y fallback a QQQ en 18. Los ceros generados por el fallback explican parte de las medianas y de la tasa de diferencias positivas; no deben interpretarse como derrotas o victorias.

## Control transversal con igual número de tickers

En las 16 fechas con ambos grupos, se igualó aleatoriamente el número de confirmados y no confirmados dentro de cada fecha y se dio el mismo peso a las fechas. En 10 000 repeticiones:

- diferencia media: +3,187 pp;
- intervalo 95%: [+1,254, +5,425] pp.

Esto refuerza que la confirmación discrimina entre candidatos simultáneos en esta muestra. Sin embargo, la política completa no mejora de forma detectable porque hay pocas fechas con selección, las cohortes son pequeñas y el beneficio transversal no aparece de manera uniforme a través del calendario.

## Decisión metodológica

No ajustar ahora el plazo de dos sesiones, los umbrales 3%/10%, SMI −40 ni las velocidades. El resultado favorable dentro de fecha y el empate de política deben conservarse juntos. El siguiente bloque necesita datos cronológicamente nuevos para probar los parámetros congelados. Mientras se acumulan, puede construirse el simulador de estados `WAIT → OBSERVE → CONFIRMED/ABSTAIN → HOLD → EXIT`, pero su primera ejecución deberá tratarse como prueba mecánica, no como nueva optimización.

Artefactos: `experiments/42_recovery_transversal_policy_v1.py`, `reports/recovery_transversal_policy_2026_09_11_v1/` y `tests/test_recovery_transversal_policy.py`.

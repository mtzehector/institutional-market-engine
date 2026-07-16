# Baseline parcial — 2026-07-16

## Alcance

Esta fotografía inicial utiliza exclusivamente la evidencia disponible en `SNDK_adaptive_memory_recent_listing.xlsx`. Es un punto de partida **parcial**, no representa todavía al universo de tickers ni al desempeño económico del sistema.

## Identificación

| Campo | Valor |
|---|---|
| Fecha de fotografía | 2026-07-16 |
| Versión del motor | 0.7.0 |
| Ticker | SNDK |
| Régimen | REGIME_2026_RECENT_LISTING |
| Periodo evaluado | 2026-01-08 a 2026-07-10 |
| Mejor memoria | 180_ROWS |
| Observaciones | 26 |
| Clasificación | MUESTRA_INSUFICIENTE |

## Métricas conservadas

| Métrica | Valor |
|---|---:|
| Predictability Score | 27.7285 |
| Exactitud direccional | 0.615385 |
| Exactitud ponderada por confianza | 0.620527 |
| Exactitud de alta confianza | 0.666667 |
| Observaciones de alta confianza | 12 |
| Casos Gap Up | 6 |
| Casos Gap Down | 2 |
| Precision Gap Up | 0.333333 |
| Recall Gap Up | 0.166667 |
| Precision Gap Down | 0.000000 |
| Recall Gap Down | 0.000000 |
| Brier Up | 0.201397 |
| Brier Skill Up | -0.134539 |
| Brier Down | 0.118707 |
| Brier Skill Down | -0.671785 |
| Brier Skill medio | -0.403162 |
| Error de calibración Up | 0.118509 |
| Error de calibración Down | 0.145628 |
| Error medio de calibración | 0.132069 |
| Gap absoluto promedio | 3.847291% |

## Comparación de memorias

| Memoria | Score | Exactitud direccional | Brier Skill medio | Error medio de calibración |
|---|---:|---:|---:|---:|
| 180_ROWS | 27.7285 | 0.615385 | -0.403162 | 0.132069 |
| 252_ROWS | 14.1654 | 0.500000 | -0.818082 | 0.221280 |
| EXPANDING | 14.0244 | 0.500000 | -0.804937 | 0.240014 |

## Lectura científica

- La memoria de 180 sesiones produjo el mejor score dentro de las ventanas válidas.
- La muestra de 26 observaciones impide considerar estable el resultado.
- El Brier Skill medio negativo indica que las probabilidades del modelo fueron peores que la referencia base en esta muestra.
- La exactitud direccional por sí sola no debe interpretarse como ventaja porque la clase `SIN_GAP` puede dominar el resultado.
- SNDK debe permanecer en un grupo de emisoras recientes con criterios de muestra y memoria diferentes a los tickers maduros.

## Uso futuro

Las fotografías posteriores deberán indicar:

1. cuáles métricas se conservaron;
2. cuáles cambiaron de definición;
3. cuáles se agregaron;
4. el valor anterior, actual y la variación;
5. si el cambio se considera mejora, deterioro o no comparable;
6. el universo, régimen, periodo y configuración empleados.

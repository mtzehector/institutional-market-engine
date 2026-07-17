# Smart Money Regime Dynamics — v0.8.2

La versión 0.8.2 estudia la relación entre Smart Money, persistencia, volumen y memoria útil sin incorporar todavía estas métricas al predictor.

## Pregunta de investigación

¿La predictibilidad disminuye cuando la memoria óptima contiene muchos cambios de control institucional, y mejora cuando los regímenes son persistentes y están respaldados por volumen relativo alto?

## Estados

Se utiliza una banda de histéresis:

- `RETAIL_DOMINANT`: Smart Money menor a 49%.
- `EQUILIBRIUM`: Smart Money entre 49% y 51%.
- `SMART_MONEY_DOMINANT`: Smart Money mayor a 51%.

Un cambio requiere persistencia mínima configurable, por defecto dos sesiones.

## Métricas principales

- `equilibrium_cross_count`
- `crossings_per_100_sessions`
- `median_regime_duration`
- `mean_regime_duration`
- `current_regime_duration`
- `regime_duration_cv`
- `false_regime_rate`
- `mean_crossing_velocity`
- `median_crossing_velocity`
- `median_relative_volume`
- `current_regime_relative_volume`
- `high_volume_cross_rate`
- `median_institutional_regime_strength`
- `current_institutional_regime_strength`
- `memory_regime_load`
- `volume_weighted_memory_regime_load`
- `smart_money_inertia_index`
- `smart_money_regime_coherence`

## Fuerza de régimen

La fuerza del régimen combina duración, volumen relativo mediano y persistencia:

```text
institutional_regime_strength
= log(1 + duration_sessions)
  × median_relative_volume
  × persistence_factor
```

No representa una señal de compra. Mide la fuerza observada del régimen, independientemente de su dirección.

## Carga ponderada de la memoria

Cada cruce recibe un peso proporcional a:

```text
abs(crossing_velocity)
× cross_relative_volume
× persistence_factor
```

La suma dentro de la ventana produce `volume_weighted_memory_regime_load`.

## Comando

```bat
market-regime ^
  --tickers config\tickers_rare_event_piloto.txt ^
  --from-date 2026-01-02 ^
  --to-date 2026-07-16 ^
  --memory-rows 180 ^
  --lower-equilibrium 49 ^
  --upper-equilibrium 51 ^
  --minimum-persistence 2 ^
  --high-volume-threshold 1.25 ^
  --evaluation-input reports\rare_event\piloto_rare_event_v081.xlsx ^
  --evaluation-sheet Mejor_Memoria ^
  --export ^
  --output reports\regimes\piloto_smart_money_regimes_v082.xlsx
```

## Hojas del Excel

- `Resumen_Ticker`: métricas agregadas y variables del reporte v0.8.1 cuando se proporciona.
- `Regimenes`: cada permanencia institucional detectada.
- `Cruces`: detalle del cambio de régimen, velocidad y volumen alrededor del cruce.
- `Correlaciones`: correlaciones Spearman entre métricas de régimen y resultados del modelo.
- `Errores`: tickers no evaluados.

## Criterio de integración futura

Estas variables no deben entrar al predictor solo porque presenten correlación en el piloto. Para promover una métrica al modelo debe:

1. conservar signo y magnitud razonables en el universo completo;
2. mantenerse en otro periodo o régimen;
3. mejorar walk-forward fuera de muestra;
4. mejorar eventos raros, calibración o estabilidad sin fuga de información.

# Experimentos 39–40 — recuperación confirmada y controles

Estado: evento histórico causal; no estrategia de cartera ni recomendación.

## Definición

Se evaluó la secuencia descubierta en el experimento 38: último SMI semanal confirmado ascendente → SMI diario del cierre anterior ≤ −40 → precio intradía dentro del 3% inferior del rango reconstruido de 40 sesiones → recuperación geométrica de 10 puntos de rango → velocidades positivas en bloques cerrados 1H y 4H → entrada hipotética en la siguiente barra de 15 minutos. Resultado: cierre después de cinco sesiones.

La implementación es deliberadamente más conservadora que el Pine publicado: no usa SMI diario en formación. Las agrupaciones 1H/4H aproximan bloques regulares desde barras 15m y no deben llamarse réplica numérica de TradingView hasta disponer de una exportación de comparación.

## Resultado bruto

- 174 tickers procesados, sin errores.
- 61 eventos confirmados en 42 tickers, pero sólo 24 fechas de confirmación independientes.
- Retorno medio a cinco sesiones: +0,982%; mediana: +0,221%; 55,74% positivos.
- Excursión favorable media: +5,302%; excursión adversa media: −4,460%.
- Intervalo 95% del retorno medio, remuestreando fechas completas: [−0,890%, +2,694%].

## Controles

- Exceso medio frente a QQQ durante el mismo intervalo: −0,230 pp; mediana −0,011 pp; intervalo 95% por fecha [−1,709, +1,270].
- Diferencia de entrar tras confirmación frente a entrar en la barra posterior a la observación: −0,300 pp de media; mediana −0,282 pp; intervalo 95% [−1,141, +0,828].
- El top anual 2026 del experimento 37 estaba congelado con datos anteriores: 12 eventos en nueve tickers, retorno medio +0,889%, mediana +3,881%, 66,7% positivos y exceso medio frente a QQQ −0,862 pp. El resto tuvo 49 eventos, retorno medio +1,004% y exceso −0,076 pp. La muestra top es demasiado pequeña y no mejora el benchmark.

## Interpretación

La señal encuentra rebotes y su retorno absoluto medio es positivo, pero en este bloque no añade rendimiento demostrado frente al mercado. Esperar confirmación tiene un costo medio de entrada y no se ha demostrado que lo compense reduciendo riesgo. La excursión adversa sigue siendo material. No avanzar todavía a una simulación optimizada de recompra/salida basada en estos valores.

El siguiente experimento debe medir el valor de la confirmación como decisión: probabilidad de recuperación frente a fallo, tiempo hasta máximo/fallo y utilidad de abstenerse en episodios no confirmados. Debe comparar reglas predefinidas, no ajustar umbrales sobre estas 24 fechas. También hace falta ampliar cronológicamente el caché 15m para obtener un bloque futuro realmente nuevo.

Artefactos: `experiments/39_recovery_monitor_confirmed_walkforward_v1.py`, `experiments/40_recovery_confirmation_controls_v1.py`, `reports/recovery_monitor_confirmed_walkforward_2026_09_11_v1/` y `reports/recovery_confirmation_controls_2026_09_11_v1/`.

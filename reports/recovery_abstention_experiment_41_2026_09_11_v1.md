# Experimento 41 — valor de abstenerse tras un plazo común

Estado: señal exploratoria; no estrategia ni recomendación.

## Diseño

Cada episodio empieza cuando el último SMI semanal confirmado está ascendente, el SMI diario del cierre anterior es ≤ −40 y el precio entra en el 3% inferior de su rango de 40 sesiones. Se permiten exactamente dos sesiones para alcanzar la confirmación geométrica y velocidades positivas 1H/4H. Confirmados y no confirmados se evalúan desde la apertura de la tercera sesión y durante cinco sesiones. Así, conocer la etiqueta de confirmación no consume parte distinta del horizonte de resultado.

Los umbrales proceden de los indicadores publicados y no se optimizaron con esta salida.

## Resultados agregados

| Estado al landmark | Episodios | Tickers | Fechas | Retorno medio | Mediana | Positivos | MFE media | MAE media |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No confirmado | 63 | 46 | 34 | +1,135% | +0,738% | 57,14% | +4,824% | −3,874% |
| Confirmado | 45 | 36 | 22 | +0,339% | +0,391% | 53,33% | +5,155% | −4,690% |

Agregado por eventos, abstenerse de los no confirmados habría eliminado un grupo que rindió mejor y sufrió menos excursión adversa. No demuestra utilidad de cartera.

## Comparación dentro de la misma fecha

Sólo 16 fechas contienen simultáneamente confirmados y no confirmados. Dando el mismo peso a cada fecha, la diferencia confirmado menos no confirmado fue +3,175 pp de media, mediana +3,662 pp, positiva en 14 de 16 fechas. Bootstrap por fecha: intervalo 95% [+0,228, +5,937] pp.

La aparente contradicción con el agregado es composición temporal: muchos eventos no confirmados se concentraron en fechas de mercado favorables, especialmente 29 de abril y 1 de mayo; en esas dos fechas los no confirmados ganaron por amplio margen. Esto es un ejemplo de por qué 108 episodios no equivalen a 108 observaciones independientes.

## Lectura

Existe una señal prometedora de discriminación **entre tickers observados en una misma fecha**, pero no evidencia todavía de que una política de abstención eleve el patrimonio de una cartera a través del tiempo. Además, los confirmados no redujeron MAE. El resultado depende de 16 fechas comparables, todas dentro de 2026, y no constituye una prueba futura intacta.

El próximo control debe congelar este clasificador sin cambiar plazo ni umbrales y evaluarlo con sesiones nuevas. Paralelamente puede medirse una política transversal que, en cada fecha, elija confirmados frente a controles elegibles y mantenga exposición comparable; sólo después se justificaría añadir recompra y salida.

Artefactos: `experiments/41_recovery_abstention_landmark_v1.py`, `reports/recovery_abstention_landmark_2026_09_11_v1/` y `tests/test_recovery_abstention_landmark.py`.

# Rare Event Evaluation — v0.8.1

## Problema

La exactitud global puede verse artificialmente alta cuando `SIN_GAP` domina el histórico. Un modelo que siempre predice ausencia de gap puede acertar muchas sesiones sin aportar conocimiento sobre los eventos que realmente interesan.

## Objetivo

La versión 0.8.1 evalúa explícitamente la detección de eventos raros y compara el modelo contra una referencia ingenua que siempre predice `SIN_GAP`.

## Nuevas métricas

Por ticker se exportan:

- `actual_gap_up_count`
- `actual_gap_down_count`
- `actual_no_gap_count`
- `predicted_gap_up_count`
- `predicted_gap_down_count`
- `predicted_no_gap_count`
- precisión, recall y F1 para cada clase
- `balanced_accuracy`
- `macro_f1`
- `no_gap_baseline_accuracy`
- `incremental_accuracy_vs_no_gap`
- `actual_rare_event_count`
- `predicted_rare_event_count`
- `rare_event_precision`
- `rare_event_recall`
- `rare_event_f1`

## Nuevo score

El `predictability_score` deja de depender principalmente de la exactitud global. Ahora combina:

- 25% balanced accuracy
- 20% macro F1
- 20% F1 de eventos raros
- 10% mejora contra la referencia `SIN_GAP`
- 10% Brier Skill
- 10% calibración
- 5% confiabilidad por tamaño de muestra

Además se aplica una compuerta de confiabilidad según la cantidad de gaps reales observados. Una muestra con pocos eventos raros no puede recibir la misma confianza que otra con evidencia suficiente.

## Compatibilidad

Se conservan las columnas históricas `gap_up_cases` y `gap_down_cases` como alias de los nuevos conteos reales.

## Interpretación

Una exactitud alta acompañada por:

- `rare_event_recall = 0`
- `rare_event_f1 = 0`
- `incremental_accuracy_vs_no_gap = 0`

indica que el modelo no supera la estrategia trivial de predecir siempre `SIN_GAP`.

La versión 0.8.1 no modifica las variables predictoras. Solo mejora la honestidad y profundidad de la evaluación fuera de muestra.

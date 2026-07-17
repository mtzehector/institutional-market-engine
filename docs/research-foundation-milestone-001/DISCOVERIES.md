# Registro de descubrimientos

Un descubrimiento no es una verdad universal. Es una conclusión respaldada por evidencia bajo condiciones explícitas.

## D-001 — Existe una memoria útil dependiente del ticker

- **Estado:** PARCIAL
- **Fecha de registro:** 2026-07-16
- **Motor:** Adaptive Memory Engine
- **Evidencia:** distintas ventanas producen scores diferentes y la ventana ganadora varía por ticker.
- **Limitaciones:** falta validación en múltiples regímenes y periodos.
- **Pregunta derivada:** ¿qué propiedades explican la longitud de memoria óptima?

## D-002 — Más historia no siempre mejora el modelo

- **Estado:** PARCIAL
- **Fecha de registro:** 2026-07-16
- **Motor:** Adaptive Memory Engine
- **Evidencia:** existen casos donde ventanas mayores reducen el Predictability Score.
- **Limitaciones:** puede existir sensibilidad a muestra pequeña y al diseño del score.
- **Pregunta derivada:** ¿cómo estimar la caducidad del conocimiento?

## D-003 — La capacidad del modelo varía entre tickers

- **Estado:** EN_EVALUACION
- **Fecha de registro:** 2026-07-16
- **Motor:** Model Evaluation Engine y Hall of Fame
- **Evidencia:** distribución no uniforme de scores, calibración y exactitud.
- **Limitaciones:** se requiere estabilidad temporal y comparación por sectores.
- **Pregunta derivada:** ¿qué características hacen que un ticker sea más modelable?

## D-004 — Una línea base reproducible cambia la calidad de la investigación

- **Estado:** CONFIRMADA_METODOLOGICAMENTE
- **Fecha de registro:** 2026-07-16
- **Motor:** Research Baseline Engine y Universe Intelligence Engine
- **Evidencia:** snapshots JSON, Excel y Markdown permiten comparar versiones, métricas nuevas y métricas retiradas.
- **Limitaciones:** la comparación exige conservar universo, periodo, régimen y definiciones.
- **Pregunta derivada:** ¿cuánto conocimiento adicional produce cada versión?

## D-005 — Comprender un ticker no implica que valga la pena operarlo

- **Estado:** HIPOTESIS_FORMALIZADA
- **Fecha de registro:** 2026-07-16
- **Origen:** propuesta de Market Potential Engine
- **Evidencia:** todavía pendiente.
- **Pregunta derivada:** ¿cómo combinar predictibilidad, potencial favorable y riesgo adverso?

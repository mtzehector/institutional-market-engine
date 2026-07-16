# Diario de investigación

Este documento conserva las hipótesis, experimentos, resultados y decisiones metodológicas del proyecto. No sustituye los artefactos de datos; los referencia y resume.

## Plantilla de entrada

```text
Fecha:
Versión:
ID de hipótesis:
Pregunta:
Hipótesis:
Datos y régimen:
Método:
Métricas de aceptación:
Resultado:
Conclusión:
Limitaciones:
Próximo experimento:
Artefactos:
```

---

## 2026-07-16 — Punto de partida reproducible

**Versión:** 0.7.0  
**ID:** KB-0001  
**Pregunta:** ¿Podemos conservar una fotografía cuantitativa del estado del proyecto y compararla objetivamente con versiones futuras?  
**Hipótesis:** Un snapshot versionado de métricas, universo, régimen y configuración permitirá distinguir mejoras reales de cambios aparentes.  
**Método:** Research Baseline Engine con salidas JSON, Excel y Markdown.  
**Estado:** Implementación inicial.

### Evidencia disponible

El archivo compartido `SNDK_adaptive_memory_recent_listing.xlsx` constituye una fotografía parcial para un ticker de cotización reciente.

Hallazgos principales:

- mejor memoria evaluada: `180_ROWS`;
- régimen: `REGIME_2026_RECENT_LISTING`;
- observaciones walk-forward: 26;
- predictability score: 27.7285;
- clasificación: `MUESTRA_INSUFICIENTE`;
- exactitud direccional: 61.5385%;
- exactitud ponderada por confianza: 62.0527%;
- predicciones de alta confianza: 12;
- exactitud de alta confianza: 66.6667%;
- Brier Skill medio: -0.4032;
- error medio de calibración: 0.1321;
- gap absoluto promedio: 3.8473%;
- ventanas con score inferior: `252_ROWS` = 14.1654 y `EXPANDING` = 14.0244;
- ventanas 120 y 150 sin predicciones válidas para el periodo y configuración utilizados.

### Interpretación provisional

La memoria de 180 sesiones supera a las memorias mayores para esta evaluación, pero la muestra es insuficiente y el Brier Skill negativo indica que el modelo probabilístico aún no mejora la referencia base para SNDK. No debe considerarse evidencia de rentabilidad ni de predictibilidad estable.

### Próximo experimento

Generar una fotografía del universo piloto y posteriormente del universo completo bajo un régimen y configuración homogéneos. Comparar cada nueva versión contra este punto de partida mediante métricas conservadas y nuevas métricas explícitamente catalogadas.

---

## Hipótesis acumuladas

| ID | Hipótesis | Estado |
|---|---|---|
| KB-0001 | Las fotografías versionadas permiten medir avance relativo del proyecto. | En implementación |
| KB-0002 | La memoria óptima varía por ticker y régimen. | Evidencia preliminar |
| KB-0003 | Las variables institucionales mejoran la probabilidad de gaps fuera de muestra. | Pendiente |
| KB-0004 | Algunos tickers son consistentemente más modelables que otros. | En evaluación |
| KB-0005 | El ADN temporal permite detectar mutaciones antes que métricas estáticas. | Pendiente |

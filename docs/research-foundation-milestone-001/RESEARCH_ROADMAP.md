# Research Roadmap

Este documento no es una lista cerrada de tareas. Es un mapa vivo de preguntas, hipótesis y ramas de conocimiento.

## Estados

- `CONFIRMADA`: evidencia consistente bajo la definición y régimen estudiados.
- `PARCIAL`: evidencia favorable, todavía limitada por muestra, régimen o universo.
- `EN_EVALUACION`: experimento implementado o en ejecución.
- `PENDIENTE`: hipótesis formulada sin evidencia suficiente.
- `RECHAZADA`: la evidencia no sostiene la hipótesis bajo las condiciones evaluadas.
- `REFORMULADA`: la hipótesis original produjo una definición más precisa.

## Árbol de investigación

### A. Predictibilidad

| ID | Pregunta o hipótesis | Estado | Evidencia principal |
|---|---|---|---|
| H-P001 | Algunos tickers son más modelables que otros. | EN_EVALUACION | Model Evaluation Engine y Hall of Fame. |
| H-P002 | La calibración aporta más valor que el acierto aislado. | EN_EVALUACION | Brier Score, Brier Skill y ECE. |
| H-P003 | La predictibilidad cambia por régimen. | PENDIENTE | Comparación de snapshots futuros. |
| H-P004 | La predictibilidad cambia por sector. | PENDIENTE | Requiere taxonomía sectorial homogénea. |

### B. Memoria

| ID | Pregunta o hipótesis | Estado | Evidencia principal |
|---|---|---|---|
| H-M001 | Existe una memoria óptima por ticker. | PARCIAL | Adaptive Memory Engine. |
| H-M002 | Más historia puede reducir la predictibilidad. | PARCIAL | Comparación de ventanas en SNDK y universo. |
| H-M003 | La memoria óptima depende del régimen. | PENDIENTE | Requiere snapshots en regímenes distintos. |
| H-M004 | Puede estimarse una vida media del conocimiento por ticker. | PENDIENTE | Evolution Engine. |

### C. Comportamiento institucional

| ID | Pregunta o hipótesis | Estado | Evidencia principal |
|---|---|---|---|
| H-I001 | Puede medirse un costo de conquista de Smart Money. | PARCIAL | Scripts históricos de costo de conquista. |
| H-I002 | Las campañas institucionales son detectables. | PARCIAL | Detector de campañas. |
| H-I003 | El escalonamiento VWAP/TWAP deja una huella cuantificable. | PENDIENTE | Módulo de escalonamiento. |
| H-I004 | IPAI y campañas mejoran la predicción fuera de muestra. | PENDIENTE | Comparación contra baseline. |

### D. Market DNA y evolución

| ID | Pregunta o hipótesis | Estado | Evidencia principal |
|---|---|---|---|
| H-D001 | Cada ticker posee una personalidad cuantificable. | EN_EVALUACION | Predictibilidad, memoria, gaps y volatilidad. |
| H-D002 | La personalidad cambia con el tiempo. | PENDIENTE | DNA temporal. |
| H-D003 | Las mutaciones pueden detectarse antes que una pérdida prolongada del modelo. | PENDIENTE | Mutation Detector. |
| H-D004 | La similitud entre ADN permite transferir conocimiento entre tickers. | PENDIENTE | DNA Explorer. |

### E. Potencial económico

| ID | Pregunta o hipótesis | Estado | Evidencia principal |
|---|---|---|---|
| H-O001 | El recorrido máximo teórico favorable difiere significativamente entre tickers. | PENDIENTE | Market Potential Engine v0.9. |
| H-O002 | El recorrido adverso evita confundir volatilidad con buena oportunidad. | PENDIENTE | Potencial negativo y asimetría. |
| H-O003 | Predictibilidad × oportunidad identifica un universo prioritario. | PENDIENTE | Opportunity–Understandability Matrix. |
| H-O004 | La densidad de oportunidad es más útil que el máximo absoluto. | PENDIENTE | Percentiles y frecuencias. |
| H-O005 | La curva de potencial por horizonte forma parte del Market DNA. | PENDIENTE | Horizontes 1, 2, 3, 5 y 10 sesiones. |
| H-O006 | La eficiencia de captura explica mejor la utilidad real que el potencial teórico. | PENDIENTE | Requiere estrategias operables. |

### F. Contexto del ecosistema

| ID | Pregunta o hipótesis | Estado | Evidencia principal |
|---|---|---|---|
| H-C001 | El contexto macroeconómico explica pérdidas de predictibilidad. | PENDIENTE | Context Engine. |
| H-C002 | La relevancia de cada variable contextual depende del ticker y sector. | PENDIENTE | Modelos contextuales. |
| H-C003 | El sistema puede detectar cuándo necesita mirar fuera de precio y volumen. | PENDIENTE | Residual Uncertainty Monitor. |
| H-C004 | Geopolítica, tecnología y demografía deben incorporarse de forma incremental. | PENDIENTE | Regla de evidencia incremental. |

### G. Portafolio y utilidad

| ID | Pregunta o hipótesis | Estado | Evidencia principal |
|---|---|---|---|
| H-R001 | Los tickers altamente comprensibles y con alto potencial superan al universo. | PENDIENTE | Backtest de portafolio. |
| H-R002 | El tamaño de posición debe depender de predictibilidad, oportunidad y riesgo. | PENDIENTE | Portfolio Intelligence Engine. |
| H-R003 | La rentabilidad debe evaluarse después de costos y restricciones reales. | PENDIENTE | Simulador de ejecución. |

## Capítulos previstos

1. Fundamentos y línea base científica.
2. Predictibilidad y calibración.
3. Memoria adaptativa.
4. Inteligencia del universo.
5. Market Potential Engine.
6. Market DNA y evolución.
7. Context Engine.
8. Knowledge Engine.
9. Portfolio Intelligence.
10. Generación asistida de nuevas hipótesis.

El mapa debe ampliarse cuando la evidencia revele preguntas nuevas. Nunca debe cerrarse artificialmente.

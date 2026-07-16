# Filosofía del Institutional Market Engine

## Propósito

El objetivo del proyecto no es eliminar la incertidumbre del mercado ni afirmar que puede anticiparse con precisión la conducta conjunta de todos sus participantes.

El objetivo es **reducir la incertidumbre de forma medible, verificable y reproducible**, identificar en qué tickers y regímenes esa reducción es mayor, y convertir la evidencia acumulada en mejores decisiones de investigación, riesgo y asignación de capital.

## Principios

1. **Probabilidades, no certezas.** Una salida del modelo expresa una probabilidad condicionada a los datos y al régimen observados.
2. **Calibración antes que espectacularidad.** Si el motor emite 70%, ese conjunto de predicciones debe cumplirse aproximadamente siete de cada diez veces fuera de muestra.
3. **Sin fuga de información.** Toda predicción histórica debe reproducir la información disponible antes del evento evaluado.
4. **Toda mejora debe medirse.** Una nueva variable o módulo debe demostrar que mejora la calibración, el Brier Skill, la estabilidad, la capacidad discriminativa o una métrica de utilidad previamente definida.
5. **No todos los tickers son igualmente modelables.** El proyecto evaluará continuamente dónde el modelo es confiable, débil o no modelable con las variables actuales.
6. **La memoria del mercado caduca.** La historia útil puede variar por ticker y régimen; no se asumirá que más datos siempre producen mejores modelos.
7. **Separar descubrimiento de decisión.** Market DNA, campañas y mutaciones describen el activo. La decisión de inversión requiere además expectativa, riesgo, liquidez y tamaño de posición.
8. **Registrar también los resultados negativos.** Una hipótesis rechazada evita repetir errores y forma parte de la base de conocimiento.
9. **La rentabilidad no se presume.** Se demostrará fuera de muestra y después de costos, deslizamiento, restricciones operativas y control de riesgo.
10. **Ninguna operación individual valida el modelo.** La evidencia surge de muestras suficientes, múltiples regímenes y repetición.

## Ciclo de investigación

```text
Hipótesis
    ↓
Definición matemática
    ↓
Implementación reproducible
    ↓
Walk-forward sin fuga
    ↓
Métricas y calibración
    ↓
Conclusión: confirmar, rechazar o reformular
    ↓
Registro del conocimiento
```

## Regla de aceptación de funcionalidades

Una funcionalidad entra al núcleo del proyecto cuando produce conocimiento medible y cumple al menos una de estas condiciones:

- reduce el Brier Score fuera de muestra;
- mejora la calibración;
- aumenta la precisión o recall en una clase relevante sin deterioro desproporcionado de las demás;
- identifica con estabilidad qué tickers o regímenes son modelables;
- mejora una métrica económica después de costos y con riesgo controlado;
- detecta cambios de comportamiento antes que la referencia vigente.

## Interpretación responsable

Un modelo puede poseer ventaja estadística y aun así atravesar secuencias de pérdidas. Por ello, toda señal futura deberá acompañarse de:

- probabilidad estimada;
- confianza estadística y tamaño de muestra;
- régimen y memoria utilizados;
- métricas históricas fuera de muestra;
- incertidumbre residual;
- explicación de variables principales;
- límites de riesgo y condiciones de invalidez.

## Frase guía

> No perseguimos operaciones aisladas exitosas; construimos modelos que reduzcan la incertidumbre del mercado con precisión creciente y evidencia reproducible.

# Market Memory Engine — v0.9.4

## Propósito

El Market Memory Engine reemplaza una decisión rígida basada únicamente en etiquetas de régimen por una pregunta causal y auditable:

> ¿Qué estados históricos del mercado se parecían más al estado actual y qué campeón funcionó mejor en ellos?

La versión 0.9.4 es un laboratorio de investigación. No ejecuta operaciones ni cambia automáticamente de estrategia.

## Entrada

Utiliza el Excel producido por `market-regime-transition` v0.9.3.1:

- `Estados_Diarios`: una fila por fecha transversal sincronizada.
- `Selecciones`: predicciones y resultados de cada campeón para esas fechas.

## Vector de estado

Cada fecha se representa mediante:

- retorno diario de QQQ;
- amplitud positiva a 1 día;
- amplitud positiva a 5 días;
- volumen relativo mediano;
- ATR porcentual mediano;
- amplitud institucional.

## Causalidad

Para una fecha objetivo `t`:

1. Solo se utilizan estados anteriores a `t`.
2. La mediana y escala robusta se calculan con el pasado disponible.
3. Se buscan los `k` vecinos históricos más cercanos.
4. Cada vecino pesa inversamente a su distancia.
5. Se estima qué campeón habría sido más adecuado usando solamente su desempeño en esos vecinos anteriores.

## Evaluación

La recomendación se compara posteriormente con lo realmente sucedido en la fecha objetivo:

- calidad real del campeón recomendado;
- calidad del universo completo;
- campeón oráculo de esa fecha;
- ventaja frente al universo;
- arrepentimiento frente al oráculo;
- tasa de selección exacta del oráculo.

El oráculo sirve únicamente como techo retrospectivo y nunca participa en la recomendación causal.

## Comando

```bat
market-market-memory ^
  --input reports\adaptive_universe\regime_transition_v0931.xlsx ^
  --states-sheet Estados_Diarios ^
  --selections-sheet Selecciones ^
  --neighbors 5 ^
  --minimum-history 8 ^
  --export ^
  --output reports\adaptive_universe\market_memory_v094.xlsx
```

## Hojas de salida

- `Resumen`: desempeño agregado de la memoria.
- `Recomendaciones`: campeón sugerido por fecha y resultado real posterior.
- `Scores_Campeones`: puntuación histórica estimada para cada campeón.
- `Estados_Similares`: vecinos usados y sus distancias.
- `Calidad_Real_Fecha`: desempeño real de cada campeón por fecha.

## Criterio de avance

La hipótesis gana apoyo si el selector por memoria presenta simultáneamente:

- ventaja media positiva frente al universo;
- tasa de ventaja positiva superior al 50%;
- arrepentimiento razonable frente al oráculo;
- recomendaciones estables cuando los vecinos son cercanos;
- resultados que no dependan de una sola fecha o un solo vecino.

## Limitaciones

La fotografía v0.9.3.1 conserva pocas fechas sincronizadas. La v0.9.4 debe interpretarse como prueba de arquitectura, no como validación definitiva para inversión. La memoria mejorará conforme se acumulen nuevas fotografías temporales comparables.

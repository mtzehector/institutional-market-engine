# v0.9.5 — Adaptive Intelligence Allocation Engine

## Objetivo

La memoria del mercado deja de actuar como una autoridad binaria. Su influencia cambia según la novedad, la similitud y la estabilidad de sus recuerdos. El peso restante se asigna a inteligencias de estado actual: Momentum, Volumen, Smart Money, Estructura y un perfil defensivo.

## Políticas comparadas

- `MEMORY_ONLY`: control de la v0.9.4.
- `NOVELTY_LINEAR`: reduce gradualmente el peso de la memoria conforme el mercado supera su umbral causal de novedad.
- `CONFIDENCE_BLEND`: usa directamente la confianza compuesta de la v0.9.4.1 como peso de memoria.
- `ADAPTIVE_ALLOCATION`: combina novedad, similitud y estabilidad para determinar el peso de memoria.

## Inteligencias actuales

Las señales se calculan causalmente mediante percentiles contra estados anteriores:

- Momentum: QQQ y amplitud positiva de 1 y 5 días.
- Volumen: volumen relativo y amplitud diaria.
- Smart Money: amplitud institucional.
- Defensiva: baja volatilidad, soporte institucional y amplitud de 5 días.
- Estructura: amplitud de 1 y 5 días y dirección de QQQ.

Estas señales no son aún modelos optimizados. Son especialistas auditables que permiten probar si ceder autoridad fuera de la memoria mejora la selección del campeón.

## Puntuación

Para cada campeón y fecha:

```text
allocation_score
=
memory_weight × memory_score_percentile
+
(1 - memory_weight) × current_intelligence_score
```

El motor selecciona el campeón con mayor `allocation_score` y compara su desempeño real contra el universo y contra el campeón oráculo.

## Ejecución

```bat
market-intelligence-allocation ^
  --input reports\adaptive_universe\regime_transition_v0931.xlsx ^
  --states-sheet Estados_Diarios ^
  --selections-sheet Selecciones ^
  --neighbors-grid 3,5,7,10 ^
  --baseline-neighbors 5 ^
  --minimum-history 10 ^
  --calibration-history 6 ^
  --novelty-percentile 0.80 ^
  --export ^
  --output reports\adaptive_universe\adaptive_intelligence_allocation_v095.xlsx
```

## Criterios de éxito

Una política debe superar a `MEMORY_ONLY` en una combinación de:

- ventaja media frente al universo;
- tasa de ventaja positiva;
- arrepentimiento medio y máximo frente al oráculo;
- estabilidad del peso de memoria;
- diversidad de campeones recomendados.

Una reducción del peor error acompañada por una caída extrema de la ventaja no será suficiente. El propósito es asignar autoridad, no apagar la memoria.

## Limitaciones

- Las inteligencias actuales se derivan del mismo vector de estado disponible; todavía no incluyen SMI, noticias ni variables macroeconómicas.
- El score de estado actual es heurístico y debe evaluarse antes de optimizar pesos.
- La muestra sincronizada sigue siendo pequeña.
- El resultado es un laboratorio retrospectivo, no una recomendación de inversión.

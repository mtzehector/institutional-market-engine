# Universe Intelligence Engine — versión 0.8.0

Convierte una fotografía de memoria adaptativa o un snapshot JSON en inteligencia agregada del universo.

## Comando

```bat
market-universe ^
  --input baseline_universo_260_adaptive_memory_2026.xlsx ^
  --sheet Mejor_Memoria ^
  --label BASELINE_UNIVERSE_INTELLIGENCE_2026_V0.8 ^
  --expected-tickers 260 ^
  --engine-version 0.8.0 ^
  --regime-label REGIME_2026 ^
  --top 20 ^
  --output-dir reports\universe\BASELINE_2026
```

## Productos

- Excel de auditoría.
- JSON canónico para comparaciones futuras.
- Markdown para conservar el informe en Git.

## Contenido

- resumen ejecutivo;
- cobertura del universo;
- estadísticos descriptivos y percentiles;
- distribución de grados;
- distribución de memoria óptima;
- top y bottom de predictibilidad;
- métricas completas por ticker;
- manifiesto reproducible con SHA-256 de la fuente.

## Regla de comparación

Una versión futura debe conservar o declarar explícitamente cambios en:

- universo de tickers;
- periodo evaluado;
- régimen;
- step;
- definición de gap;
- ventanas de memoria;
- versión del motor;
- archivo fuente y su SHA-256.

El informe describe la capacidad del modelo sobre el universo. No constituye una recomendación de inversión.

# Institutional Market Engine

> **"El asombro no es la recompensa de la investigación. Es el combustible que mantiene viva la investigación."**

Motor cuantitativo experimental para analizar campañas de Smart Money, costo de conquista, huellas de ejecución escalonada, memoria adaptativa, predictibilidad y probabilidades de gap para la siguiente sesión.

## Estado

Versión **0.7.0**. Todos los modelos son experimentales; no identifican directamente participantes institucionales ni constituyen recomendaciones de inversión.

El propósito del proyecto no es eliminar la incertidumbre del mercado, sino reducirla de forma medible, verificable y reproducible. Consulte [`PROJECT_PHILOSOPHY.md`](PROJECT_PHILOSOPHY.md) y [`RESEARCH_JOURNAL.md`](RESEARCH_JOURNAL.md).

## Capacidades actuales

- Proveedor reutilizable Financial Modeling Prep (FMP Starter).
- Límite global configurable, con máximo conservador de 250 solicitudes por minuto.
- Caché local de históricos EOD.
- Smart Money vs Retail Flow con RSI de Wilder.
- ATR porcentual y Efficiency Ratio.
- Construcción de variables sin fuga de datos para la apertura siguiente.
- Predictor independiente de gap up y gap down significativo.
- Umbral adaptativo: `max(1%, 0.5 × ATR porcentual)`.
- Regresión logística cronológica combinada con sesiones históricas comparables.
- Walk-forward sin fuga de información.
- Model Evaluation Engine con ranking de predictibilidad por ticker.
- Model Registry y Hall of Fame persistentes.
- Adaptive Memory Engine por ticker y régimen.
- Research Baseline Engine para fotografías versionadas y comparables.
- Brier Score, Brier Skill, ROC-AUC, calibración, precisión, recall y lift.
- Exportación a Excel, JSON y Markdown para auditoría.

## Instalación

```bash
python -m venv .venv
```

Windows:

```bat
.venv\Scripts\activate
pip install -e .[dev]
copy .env.example .env
```

Configura `.env`:

```ini
FMP_API_KEY=TU_LLAVE_REAL
FMP_RATE_LIMIT_PER_MINUTE=250
FMP_WORKERS=12
GAP_MIN_PCT=1.0
GAP_ATR_MULTIPLIER=0.50
ATR_LENGTH=14
```

## Predicción para la siguiente sesión

```bat
market-engine predict-gap --ticker TEAM --years 5 --export
```

## Walk-forward de un ticker

```bat
market-engine walk-forward ^
  --ticker TEAM ^
  --from-date 2026-01-02 ^
  --to-date 2026-07-15 ^
  --training-years 5 ^
  --step 1 ^
  --export
```

## Evaluar predictibilidad

```bat
market-engine evaluate-models ^
  --ticker TEAM ^
  --from-date 2026-01-02 ^
  --to-date 2026-07-15 ^
  --training-years 5 ^
  --step 1 ^
  --export ^
  --output TEAM_predictibilidad.xlsx
```

Para una lista, utilice `--tickers archivo.txt`. Durante exploración puede usar `--step 5`; para una evaluación definitiva use `--step 1`.

## Memoria adaptativa

```bat
market-memory ^
  --ticker TEAM ^
  --from-date 2026-01-02 ^
  --to-date 2026-07-15 ^
  --windows 180,252,378,504,756,expanding ^
  --regime-label REGIME_2026 ^
  --step 5 ^
  --export ^
  --output TEAM_adaptive_memory_2026.xlsx
```

## Registro y Hall of Fame

```bat
market-registry record ^
  --input ranking_predictibilidad.xlsx ^
  --sheet Ranking ^
  --model-version 0.7.0 ^
  --run-label REGIME_2026 ^
  --output hall_of_fame.xlsx
```

## Fotografía de investigación

Crea una fotografía reproducible usando uno o varios archivos de resultados:

```bat
market-snapshot create ^
  --input SNDK_adaptive_memory_recent_listing.xlsx ^
  --sheet Mejor_Memoria ^
  --label BASELINE_2026-07-16 ^
  --as-of-date 2026-07-16 ^
  --regime-label REGIME_2026_RECENT_LISTING ^
  --notes "Fotografía parcial inicial" ^
  --output-dir snapshots\2026-07-16
```

Genera:

- JSON canónico para comparaciones automáticas;
- Excel auditable;
- Markdown legible y versionable.

Compara una fotografía futura contra la inicial:

```bat
market-snapshot compare ^
  --baseline snapshots\2026-07-16\BASELINE_2026-07-16.json ^
  --current snapshots\2026-08-16\SNAPSHOT_2026-08-16.json ^
  --export ^
  --output snapshots\comparacion_2026-07-16_vs_2026-08-16.xlsx
```

La comparación conserva métricas existentes y marca explícitamente métricas nuevas, retiradas o cuya definición requiere revisión.

## Índice de Predictibilidad

El score de 0 a 100 utiliza:

- 45% componente de exactitud direccional de tres clases;
- 20% exactitud ponderada por confianza;
- 15% Brier Skill;
- 10% calibración;
- 10% confiabilidad por tamaño de muestra;
- compuerta direccional para impedir que métricas secundarias compensen una mala clasificación;
- ajuste de alta confianza cuando existen al menos diez observaciones.

Con menos de 40 observaciones se clasifica como `MUESTRA_INSUFICIENTE`, aunque se conserve el score provisional.

## Pruebas

```bat
pytest
```

Las pruebas validan el umbral adaptativo, la ausencia de fuga de datos, sesiones comparables, ranking de predictibilidad, memoria adaptativa, registro persistente y fotografías de investigación.

## Estructura principal

```text
src/market_engine/
├── providers/fmp.py
├── indicators/
├── gaps/
├── backtesting/walk_forward.py
├── evaluation/model_evaluation.py
├── evaluation/model_registry.py
├── evaluation/adaptive_memory.py
├── evaluation/research_snapshot.py
├── cli.py
├── registry_cli.py
├── memory_cli.py
└── snapshot_cli.py
```

## Seguridad

Nunca publiques `.env`, claves API, archivos de caché o resultados privados. Esos recursos están excluidos mediante `.gitignore`.

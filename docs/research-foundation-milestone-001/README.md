# Institutional Market Research Platform

> **No desarrollamos software para predecir el mercado.  
> Desarrollamos conocimiento para comprenderlo.**

Plataforma científica experimental dedicada a reducir incertidumbre mediante investigación cuantitativa reproducible, acumulación de evidencia y aprendizaje continuo.

## Identidad

Los mercados son ecosistemas complejos y adaptativos. La plataforma no promete certeza ni rentabilidad automática. Formula preguntas, construye hipótesis, valida modelos fuera de muestra, registra resultados y convierte cada conclusión en nuevas preguntas.

Documentos fundacionales:

- [`PROJECT_PHILOSOPHY.md`](PROJECT_PHILOSOPHY.md)
- [`RESEARCH_MANIFESTO.md`](RESEARCH_MANIFESTO.md)
- [`RESEARCH_ROADMAP.md`](RESEARCH_ROADMAP.md)
- [`RESEARCH_JOURNAL.md`](RESEARCH_JOURNAL.md)
- [`DISCOVERIES.md`](DISCOVERIES.md)
- [`BASELINES.md`](BASELINES.md)
- [`LEGACY.md`](LEGACY.md)

## Estado técnico

Versión de software actual: **0.8.0**.

Todos los modelos y métricas son experimentales. No identifican directamente a participantes institucionales ni constituyen recomendaciones de inversión.

## Capacidades actuales

- Financial Modeling Prep como proveedor reutilizable.
- Límite global y caché local.
- Smart Money y Retail Flow experimentales.
- ATR porcentual y Efficiency Ratio.
- Variables construidas sin fuga para la apertura siguiente.
- Predictor de gap up/down significativo.
- Walk-forward.
- Model Evaluation Engine.
- Model Registry y Hall of Fame.
- Adaptive Memory Engine.
- Research Baseline Engine.
- Universe Intelligence Engine.
- Brier Score, Brier Skill, calibración, precisión, recall y lift.
- Salidas auditables en Excel, JSON y Markdown.

## Método

```text
Pregunta → Hipótesis → Modelo → Implementación → Backtesting
→ Walk-forward → Evidencia → Conocimiento → Nueva pregunta
```

## Instalación

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
copy .env.example .env
```

Configura `.env` sin publicar la llave:

```ini
FMP_API_KEY=TU_LLAVE_REAL
FMP_RATE_LIMIT_PER_MINUTE=250
FMP_WORKERS=12
GAP_MIN_PCT=1.0
GAP_ATR_MULTIPLIER=0.50
ATR_LENGTH=14
```

## Comandos principales

```bat
market-engine predict-gap --ticker TEAM --years 5 --export
```

```bat
market-engine walk-forward ^
  --ticker TEAM ^
  --from-date 2026-01-02 ^
  --to-date 2026-07-15 ^
  --training-years 5 ^
  --step 1 ^
  --export
```

```bat
market-memory ^
  --ticker TEAM ^
  --from-date 2026-01-02 ^
  --to-date 2026-07-15 ^
  --windows 180,252,378,504,756,expanding ^
  --regime-label REGIME_2026 ^
  --step 5 ^
  --export
```

```bat
market-universe ^
  --input baseline_universo_260_adaptive_memory_2026.xlsx ^
  --sheet Mejor_Memoria ^
  --label BASELINE_UNIVERSE_INTELLIGENCE_2026_V0.8 ^
  --expected-tickers 260 ^
  --engine-version 0.8.0 ^
  --regime-label REGIME_2026 ^
  --output-dir reports\universe\BASELINE_2026
```

## Línea base científica

El universo inicial de 260 tickers constituye la fotografía oficial contra la que se evaluarán las mejoras futuras. Toda comparación debe conservar o declarar cambios en universo, periodo, régimen, configuración y definiciones de métricas.

## Próximo capítulo

**v0.9 — Market Potential Engine**

Pregunta central:

> ¿Cuánto valor económico potencial existe en los tickers que el modelo comprende mejor?

Métricas previstas:

- potencial teórico favorable;
- potencial teórico adverso;
- asimetría;
- densidad de oportunidad;
- curva por horizontes;
- matriz de oportunidad × comprensión;
- estructura para eficiencia de captura.

## Pruebas

```bat
pytest
```

## Seguridad

Nunca publiques `.env`, claves API, cachés privadas ni resultados sensibles.

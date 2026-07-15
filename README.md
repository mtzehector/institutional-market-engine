# Institutional Market Engine

Motor cuantitativo experimental para analizar campañas de Smart Money, costo de conquista, huellas de ejecución escalonada y probabilidades de gap para la siguiente sesión.

## Estado

Versión **0.4.0** en desarrollo. Las métricas de Smart Money, IPAI, campaña institucional, predictibilidad y probabilidad de gap son modelos experimentales; no identifican directamente participantes institucionales ni constituyen recomendaciones de inversión.

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
- Walk-forward con ventana expansiva.
- Model Evaluation Engine con ranking de predictibilidad por ticker.
- Brier Score, Brier Skill, ROC-AUC, calibración, precisión, recall y lift.
- Exportación a Excel para auditoría.

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
  --from-date 2025-01-02 ^
  --to-date 2026-07-14 ^
  --training-years 5 ^
  --step 1 ^
  --export
```

## Evaluar predictibilidad de un ticker

```bat
market-engine evaluate-models ^
  --ticker TEAM ^
  --from-date 2025-01-02 ^
  --to-date 2026-07-14 ^
  --training-years 5 ^
  --step 1 ^
  --export ^
  --output TEAM_predictibilidad.xlsx
```

## Ranking de una lista de tickers

El TXT debe contener un ticker por línea.

```bat
market-engine evaluate-models ^
  --tickers config\tickers_nasdaq.txt ^
  --from-date 2025-01-02 ^
  --to-date 2026-07-14 ^
  --training-years 5 ^
  --step 5 ^
  --export ^
  --output ranking_predictibilidad_FMP.xlsx
```

Para una evaluación definitiva use `--step 1`. Durante pruebas iniciales puede usar `--step 5` o `--step 10` para reducir el tiempo de procesamiento.

## Índice de Predictibilidad

El score de 0 a 100 combina:

- 35% exactitud direccional de tres clases: Gap Up, Gap Down y Sin Gap.
- 20% exactitud ponderada por la confianza emitida.
- 20% mejora del Brier Score respecto a la probabilidad histórica base.
- 15% calibración de las probabilidades.
- 10% confiabilidad por tamaño de muestra.

Las predicciones de alta confianza también aportan un ajuste adicional cuando existe una muestra suficiente.

### Grados

| Score | Grado |
|---:|---|
| 80–100 | A_EXCELENTE |
| 65–79.99 | B_CONFIABLE |
| 50–64.99 | C_EN_OBSERVACION |
| 35–49.99 | D_DEBIL |
| 0–34.99 | E_NO_CONFIABLE |

Con menos de 40 observaciones el grado es `MUESTRA_INSUFICIENTE`, aunque se siga mostrando el score provisional.

## Excel de evaluación

El comando `evaluate-models --export` genera:

- `Ranking`: puntaje y expediente completo por ticker.
- `Bandas_Confianza`: exactitud observada por nivel de probabilidad emitida.
- `Desempeno_Reciente`: ventanas de 20, 60, 120 y 250 sesiones.
- `Predicciones`: detalle cuando se usa un ticker o `--include-predictions`.
- `Errores`: tickers que no pudieron evaluarse.

## Pruebas

```bat
pytest
```

Las pruebas validan el umbral adaptativo, la ausencia de fuga de datos, el cálculo de sesiones comparables y el ranking de predictibilidad.

## Estructura

```text
src/market_engine/
├── providers/fmp.py
├── indicators/smart_money.py
├── indicators/volatility.py
├── gaps/features.py
├── gaps/predictor.py
├── backtesting/walk_forward.py
├── evaluation/model_evaluation.py
└── cli.py
```

## Seguridad

Nunca publiques `.env`, claves API, archivos de caché o resultados privados. Esos recursos están excluidos mediante `.gitignore`.

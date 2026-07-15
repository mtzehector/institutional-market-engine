# Institutional Market Engine

Motor cuantitativo experimental para analizar campañas de Smart Money, costo de conquista, huellas de ejecución escalonada y probabilidades de gap para la siguiente sesión.

## Estado

Versión 0.3.0 en desarrollo. Las métricas de Smart Money, IPAI, campaña institucional y probabilidad de gap son modelos experimentales; no identifican directamente participantes institucionales ni constituyen recomendaciones de inversión.

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
- Brier Score, ROC-AUC, probabilidad base y lift.
- Walk-forward con ventana expansiva y reentrenamiento para cada predicción.
- Tabla de calibración por intervalos de probabilidad.
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

## Predicción de la siguiente sesión

Con TEAM:

```bat
market-engine predict-gap --ticker TEAM --years 5 --export
```

También puede ejecutarse como módulo:

```bat
python -m market_engine.cli predict-gap --ticker TEAM --years 5 --export
```

El resultado se muestra en consola y, con `--export`, se genera:

```text
prediccion_gap_manana_FMP.xlsx
```

Hojas:

- `Prediccion`: probabilidad de gap up, gap down y ausencia de gap relevante.
- `Auditoria`: variables y etiquetas históricas para revisar el modelo.

## Walk-forward

Para reproducir qué habría predicho el modelo antes de las aperturas del 13 y 14 de julio de 2026:

```bat
market-engine walk-forward ^
  --ticker TEAM ^
  --from-date 2026-07-13 ^
  --to-date 2026-07-14 ^
  --training-years 5 ^
  --export ^
  --output TEAM_walk_forward_2026-07-13_2026-07-14.xlsx
```

La fecha evaluada es la fecha de apertura objetivo. Para predecir el 13 de julio, el motor solo usa información disponible hasta la sesión anterior. Para predecir el 14 de julio, incorpora el cierre del 13, pero nunca la apertura del 14.

Para evaluar un periodo más amplio:

```bat
market-engine walk-forward ^
  --ticker TEAM ^
  --from-date 2025-01-02 ^
  --to-date 2026-07-14 ^
  --training-years 5 ^
  --step 1 ^
  --decision-threshold 0.50 ^
  --export
```

El Excel contiene:

- `Predicciones`: probabilidad emitida, gap real y acierto por sesión.
- `Metricas`: Brier Score, ROC-AUC, precision y recall.
- `Calibracion`: probabilidad media frente a frecuencia real observada.

`--step 1` predice todas las sesiones. Para una prueba más rápida puede utilizarse `--step 5`, aunque eso evalúa solamente una de cada cinco sesiones.

## Interpretación

- `probability_up`: probabilidad estimada de gap up significativo.
- `probability_down`: probabilidad estimada de gap down significativo.
- `probability_no_gap`: probabilidad residual.
- `base_up` y `base_down`: personalidad histórica base del ticker.
- `lift_up` y `lift_down`: probabilidad actual dividida por la base histórica.
- `brier_up` y `brier_down`: menor es mejor.
- `roc_auc_up` y `roc_auc_down`: 0.5 equivale aproximadamente a azar.
- `correct_direction`: compara `GAP_UP`, `GAP_DOWN` o `SIN_GAP` usando el umbral de decisión.

## Pruebas

```bat
pytest
```

Las pruebas validan el umbral adaptativo, la ausencia de etiquetas futuras, la conversión numérica de sesiones comparables y la cronología del walk-forward.

## Estructura

```text
src/market_engine/
├── backtesting/walk_forward.py
├── providers/fmp.py
├── indicators/smart_money.py
├── indicators/volatility.py
├── gaps/features.py
├── gaps/predictor.py
└── cli.py
```

## Seguridad

Nunca publiques `.env`, claves API, archivos de caché o resultados privados. Esos recursos están excluidos mediante `.gitignore`.

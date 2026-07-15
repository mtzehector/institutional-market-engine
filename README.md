# Institutional Market Engine

Motor cuantitativo experimental para analizar campañas de Smart Money, costo de conquista, huellas de ejecución escalonada y probabilidades de gap para la siguiente sesión.

## Estado

Versión 0.2.0 en desarrollo. Las métricas de Smart Money, IPAI, campaña institucional y probabilidad de gap son modelos experimentales; no identifican directamente participantes institucionales ni constituyen recomendaciones de inversión.

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

## Primera ejecución

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

## Interpretación

- `probability_up`: probabilidad estimada de gap up significativo.
- `probability_down`: probabilidad estimada de gap down significativo.
- `probability_no_gap`: probabilidad residual.
- `base_up` y `base_down`: personalidad histórica base del ticker.
- `lift_up` y `lift_down`: probabilidad actual dividida por la base histórica.
- `brier_up` y `brier_down`: menor es mejor.
- `roc_auc_up` y `roc_auc_down`: 0.5 equivale aproximadamente a azar.

## Pruebas

```bat
pytest
```

Las pruebas iniciales validan que el umbral nunca sea inferior al mínimo y que la última fila no contenga la etiqueta futura, evitando fuga de datos.

## Estructura

```text
src/market_engine/
├── providers/fmp.py
├── indicators/smart_money.py
├── indicators/volatility.py
├── gaps/features.py
├── gaps/predictor.py
└── cli.py
```

## Seguridad

Nunca publiques `.env`, claves API, archivos de caché o resultados privados. Esos recursos están excluidos mediante `.gitignore`.

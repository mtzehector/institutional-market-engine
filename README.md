# Institutional Market Engine

Motor cuantitativo experimental para analizar campañas de Smart Money, costo de conquista, huellas de ejecución escalonada y probabilidades de gap para la siguiente sesión.

## Estado

Versión inicial en desarrollo. Las métricas de Smart Money, IPAI, campaña institucional y probabilidad de gap son modelos experimentales; no identifican directamente participantes institucionales ni constituyen recomendaciones de inversión.

## Capacidades previstas

- Acceso reutilizable a Financial Modeling Prep (FMP Starter).
- Límite global configurable, con máximo conservador de 250 solicitudes por minuto.
- Caché local de históricos y datos de referencia.
- Smart Money vs Retail Flow.
- Costo de conquista y línea base de equilibrio.
- Detección de campañas institucionales.
- Huella compatible con escalonamiento VWAP/TWAP e IPAI.
- Predictor de gap up/down para la siguiente sesión.
- Reportes Excel y ejecución por línea de comandos.

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
```

## Primera prueba

```bash
python -m market_engine.cli analyze --ticker VIRT --from-date 2025-01-01
```

## Estructura

```text
src/market_engine/
├── providers/       # FMP y futuros proveedores
├── indicators/      # Smart Money, momentum y volatilidad
├── campaigns/       # costo de conquista, campañas e intensidad
├── execution/       # escalonamiento VWAP/TWAP e IPAI
├── gaps/            # personalidad y predictor del gap siguiente
└── reports/         # Excel y salidas
```

## Seguridad

Nunca publiques `.env`, claves API, archivos de caché o resultados privados. Esos recursos están excluidos mediante `.gitignore`.

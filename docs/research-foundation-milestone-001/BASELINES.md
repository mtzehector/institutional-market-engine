# Índice de líneas base

Las líneas base son fotografías oficiales del estado del conocimiento. Nunca deben sobrescribirse.

## Reglas de comparabilidad

Una comparación válida debe conservar o declarar explícitamente cambios en:

- universo de tickers;
- fechas de evaluación;
- régimen;
- versión del motor;
- definición de gaps;
- ventanas de memoria;
- `step` walk-forward;
- umbral de decisión;
- proveedor y versión de datos;
- métricas y fórmulas.

## Baselines registradas

### B-0001 — SNDK Partial Baseline

- **Fecha:** 2026-07-16
- **Alcance:** parcial, un ticker de cotización reciente.
- **Régimen:** `REGIME_2026_RECENT_LISTING`.
- **Objetivo:** validar el mecanismo de snapshots y memoria adaptativa.
- **Limitación principal:** muestra insuficiente.

### B-0002 — Universo NASDAQ 260 v0.7

- **Fecha:** 2026-07-16
- **Alcance esperado:** 260 tickers.
- **Evaluados:** 259 según el resultado inicial compartido.
- **Régimen:** `REGIME_2026`.
- **Ventanas:** 180, 252, 378, 504, 756 y expanding.
- **Step:** 5.
- **Artefactos esperados:** snapshot JSON, Excel y Markdown; ranking adaptativo; Hall of Fame.
- **Uso:** referencia oficial para medir el avance relativo de versiones futuras.

## Convención de nombres

```text
BASELINE_<UNIVERSO>_<REGIMEN>_<FECHA>_V<VERSION>
```

Ejemplo:

```text
BASELINE_UNIVERSO_260_REGIME_2026_2026-07-16_V0.7
```

## Política de preservación

- El JSON es la referencia canónica.
- El Markdown es el resumen versionable.
- El Excel es el artefacto de auditoría humana.
- Toda baseline debe incluir huella SHA-256 de sus fuentes.
- Una definición de métrica modificada debe registrarse como métrica nueva o revisión de definición, nunca compararse silenciosamente.

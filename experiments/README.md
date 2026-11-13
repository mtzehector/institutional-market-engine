# Tendencia larga y recuperación — experimentos 36–42

Este bloque conserva la investigación local de septiembre de 2026. Es exploratorio: no implementa una cartera ejecutable ni acredita rentabilidad neta o el sistema de Ernesto.

## Recorrido

| Experimento | Pregunta |
| --- | --- |
| 36 | ¿SMI diario/semanal ascendente mejora los desenlaces futuros? |
| 37 | ¿La selección anual con información anterior mantiene ventaja? |
| 38 | ¿Los candidatos Hunter son compatibles con el contexto largo? |
| 39 | ¿Qué ocurre después de una recuperación confirmada? |
| 40 | ¿Añade valor frente a QQQ y frente a entrar tras observar? |
| 41 | ¿Confirmar dentro de un plazo común ayuda a discriminar episodios? |
| 42 | ¿La política transversal mejora controles con exposición comparable? |

El protocolo está en `docs/INDICATOR_ROTATION_WALKFORWARD_DESIGN_v1.md`. Los reportes interpretativos están en `reports/long_smi_experiments_36_37_2026_09_11_v1.md`, `reports/recovery_experiments_39_40_2026_09_11_v1.md`, `reports/recovery_abstention_experiment_41_2026_09_11_v1.md` y `reports/recovery_transversal_experiment_42_2026_09_11_v1.md`.

Los resultados no muestran una ventaja relativa estable del filtro largo ni una mejora detectable de la política completa de recuperación. La discriminación favorable entre candidatos simultáneos queda como hipótesis pendiente de confirmación cronológica. Conservar ambas conclusiones.

## Reproducción y datos

Los scripts tienen argumentos explícitos: consultar `python experiments/<archivo>.py --help` desde la raíz. El recorrido parte de 36; 37 y 38 consumen su panel, 39 lo combina con caché 15m y 40–42 evalúan controles/episodios posteriores. Revisar los argumentos requeridos de cada archivo antes de ejecutar; no asumir una descarga automática.

Los históricos FMP, las claves y el registro local de modelos no se publican. El panel derivado `reports/long_smi_filter_walkforward_2026_09_11_v1/panel.csv` (aproximadamente 51 MB) tampoco se publica: se genera en el experimento 36 con los insumos correspondientes. Los hashes de ese panel y del registro local están en `experiments/local_artifacts_manifest_2026_09_11_v1.json`. El clon por sí solo no permite reproducir los resultados sin obtener y auditar los insumos. Los CSV de reportes restantes conservan resultados y diagnósticos, no un nuevo bloque de prueba independiente.

`market_caps_demo.csv` contiene una muestra sintética AAA/BBB para demostraciones, no historia de mercado. `config/tickers_rare_event_piloto.txt` conserva una lista piloto exploratoria, no recomendaciones de compra.

El indicador `Multi-Timeframe SMI Regime Recovery Alert v0.1.txt` es experimental. Sus temporalidades superiores en formación pueden cambiar: sus marcadores históricos no equivalen a un backtest ni se asume equivalencia con las señales confirmadas de Python.

## Verificación de esta entrega

Se ejecuta una selección de 16 pruebas de variables de gaps, walk-forward, controles de tendencia/recuperación y confiabilidad de distribuciones. No representa la suite completa ni reproduce los resultados económicos. Las pruebas nuevas están en `tests/test_long_smi_walkforward_experiments.py`, `tests/test_recovery_abstention_landmark.py` y `tests/test_recovery_transversal_policy.py`.

# Experimentos 36–37: filtro largo y selección cronológica de tickers

Fecha de datos máxima: 2026-09-11. Estado: investigación descriptiva con desenlaces adelantados; **no** simulación de cartera, recompra, salida ni rendimiento realizado.

## Datos y auditoría

- 177 tickers coinciden entre el EOD local y el caché FMP 15m actualizado. Se unieron los EOD hasta julio de 2026 con sesiones completas reconstruidas de 15m posteriores.
- 174 aceptados. AZN, IESC y MNST excluidos por discrepancias de escala/precio mayores a 1% entre EOD y 15m en días solapados (aproximadamente 100%, 50% y 50%). No se corrigieron silenciosamente.
- La cohorte procede de un universo almacenado hoy: posible sesgo de supervivencia. Los desenlaces de diez sesiones se solapan; los conteos de ticker-días no representan observaciones independientes.

## Experimento 36 — filtro largo

SMI(5,3) y señal EMA(3) calculados en diario y semanal. “Ascendente” exige SMI > señal y pendiente SMI positiva en cada temporalidad. El SMI semanal se usa a partir de la sesión **posterior** al cierre real de la semana. Una señal al cierre de t se evalúa desde la apertura de t+1 hasta el cierre de t+10. Sin costos ni sizing.

| Grupo | Ticker-días | Retorno medio a diez sesiones | Desenlaces positivos |
|---|---:|---:|---:|
| Diario y semanal ascendentes | 38 662 | +0,707% | 53,51% |
| Sólo diario ascendente | 69 412 | +0,739% | 53,05% |
| Otros estados | 142 364 | +0,812% | 53,48% |

Estas medias brutas agregadas mezclan fechas/regímenes. En comparación transversal dentro de las mismas fechas, exigiendo al menos cinco tickers por grupo, el grupo diario+semanal tuvo una diferencia media de **−0,123 puntos porcentuales** frente al resto en 1 308 fechas. Por año: 2023 +0,092; 2024 −0,132; 2025 −0,247; 2026 +0,086 pp. El filtro largo por sí solo no muestra una mejora consistente.

## Experimento 37 — selección anual walk-forward

Para cada año 2023–2026 se estimó con los dos años anteriores la diferencia por ticker entre el retorno tras estado diario+semanal ascendente y el retorno en otros estados. Se exigieron al menos 30 días elegibles y 80 días de control; se eligió el quintil superior (35 de 174 tickers). Los labels cuyo cierre de salida alcanzaba el año de prueba se excluyeron del entrenamiento. El año de prueba se evaluó completo sin reajustar la selección.

| Año probado | Top: retorno medio tras señal | Resto elegible | Diferencia media emparejada por fecha |
|---|---:|---:|---:|
| 2023 | +1,348% | +1,892% | −0,936 pp |
| 2024 | +0,700% | +0,471% | +0,338 pp |
| 2025 | +0,806% | +1,164% | −0,575 pp |
| 2026 | +1,389% | +1,314% | +0,413 pp |

La selección superó al resto en dos de cuatro años y quedó por debajo en los otros dos. En 2026 la diferencia media fue positiva, pero sólo 48,4% de las fechas emparejadas favorecieron al top. **Hay subgrupos con retornos futuros positivos, pero no se ha demostrado una ventaja relativa estable.** No publicar el top como lista de compra.

## Límites y siguiente prueba

No se ha probado todavía la señal Hunter/Monitor, el ranking de recuperación condicional, la recompra tras una venta, el agotamiento diario/semanal ni la rotación con capital compartido. El próximo paso debe usar el filtro largo como **estrato de contexto**, no como ventaja estable asumida, y estudiar si los eventos de recuperación aportan información adicional respecto a controles del mismo ticker/fecha. Después podrían modelarse las etapas de cartera sólo si los componentes muestran utilidad suficiente. Reconciliar previamente las señales Pine con datos históricos confirmados.

Archivos reproducibles: `experiments/36_long_smi_filter_walkforward_v1.py`, `experiments/37_long_smi_ticker_selection_walkforward_v1.py`, `reports/long_smi_filter_walkforward_2026_09_11_v1/`, `reports/long_smi_ticker_selection_walkforward_2026_09_11_v1/`. Pruebas anti-fuga: `tests/test_long_smi_walkforward_experiments.py`.

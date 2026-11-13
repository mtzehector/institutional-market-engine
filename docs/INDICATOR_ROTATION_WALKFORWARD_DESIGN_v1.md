# Diseño v1 — validación walk-forward de tendencia larga, recuperación y rotación

Estado: protocolo de investigación, no indicador operativo ni backtest ejecutado.

## Fuentes existentes

- `src/indicators/Adaptive Range Opportunity Hunter.txt` (Pine v1.3): SMI y posición del cierre dentro de un rango local de 40 barras. `standardizedBenefitPct` es distancia geométrica hipotética al máximo local desde el umbral inferior; **no** es rendimiento esperado.
- `src/indicators/Adaptive Range Recovery Monitor.txt` (Pine v0.3): episodio de sobreventa diaria, zona baja del rango, recuperación geométrica y velocidad 1H/4H. No implementa filtro semanal, ranking transversal ni salida por agotamiento diario-semanal.
- El caché EOD local contiene históricos de varios años, pero termina en julio de 2026 para los archivos inspeccionados. El caché 15m del laboratorio FMP llega al 11 de septiembre de 2026 y cubre un periodo mucho menor. Antes de combinar señales hay que alinear periodo, ajustes corporativos, zona horaria y universo.

Los Pine Script publicados son referencias inmutables. Cualquier corrección o nueva lógica tendrá un archivo/version nuevos.

## Hipótesis separadas

H1. Un filtro de SMI diario y semanal ascendentes, confirmados al cierre de cada temporalidad, mejora la distribución de retornos futuros frente al mismo universo sin filtro.

H2. Entre los tickers elegibles, la información disponible al cierre anterior permite ordenar oportunidades de recuperación futura mejor que una selección neutra. La distancia al máximo local es una covariable, no el objetivo ni una predicción.

H3. Tras una venta, una recompra confirmada en un valle incrementa el número de acciones **y** el valor final neto frente a mantenerlas; evaluar ambos, porque aumentar acciones no garantiza mayor riqueza.

H4. Una salida por agotamiento SMI diario/semanal protege beneficio o reduce drawdown frente a controles de permanencia; medir también salidas tardías y giros falsos.

## Definiciones temporales obligatorias

- Señal diaria: utilizable desde el siguiente evento negociable posterior al cierre diario.
- Señal semanal: utilizable sólo después del cierre de la última sesión de esa semana. Durante la semana se conserva el último valor semanal confirmado.
- Señal 1H/4H: usar barras cerradas de esa temporalidad. Evitar mezclar valores parciales y confirmados en histórico/realtime.
- Ejecución simulada: precio posterior a la confirmación, nunca el cierre que definió la señal si ya no era negociable.
- Universo point-in-time y precios ajustados de forma consistente; registrar exclusiones y delistings cuando los datos permitan observarlos.

## Auditoría Pine → Python antes de cualquier rendimiento

1. Replicar SMI(5,3), EMA de señal, extremos locales, posición de rango y disparos de cada script en un panel de eventos, con pruebas de igualdad numérica frente a ejemplos exportados de TradingView.
2. Tratar `request.security(..., lookahead_off)` con cautela en el monitor: en tiempo real puede mostrar la barra diaria/4H aún en formación. Construir columnas separadas `latest_confirmed` y `live_partial`; walk-forward usa sólo la primera.
3. El episodio del monitor se inicia, pero no se desactiva explícitamente; antes de probarlo debe definirse una expiración/invalidez sin mirar el futuro.
4. `acceleration4H` se deriva de una serie 4H replicada en el gráfico 1H; no equivale necesariamente a aceleración calculada en barras 4H. Registrar ambas definiciones, sin corregir silenciosamente el Pine publicado.
5. `ta.pivotlow` necesita `pivotRight` barras posteriores; su disponibilidad real es la barra de confirmación, no el mínimo retrospectivo.

## Secuencia de experimentos sin optimización prematura

E1 — **Filtro largo**: baseline SMI diario, SMI semanal y conjunción, frente a todo el universo elegible. Reportar cobertura, tasas de estados, persistencia y retornos futuros con intervalos por bloques de fecha/ticker. Sin top ni trading.

E2 — **Recovery**: dentro de cada estado E1, comparar señales Hunter y Monitor con controles elegibles sin señal. Desenlaces prefijados: retorno neto a horizontes fijos, excursión favorable/adversa, probabilidad de recuperar máximo local antes de una caída definida y tiempo hasta el evento. La etiqueta se calcula sólo después del instante de señal.

E3 — **Ranking transversal**: entrenar sólo con fechas pasadas y congelar ranking antes del bloque de prueba. Empezar por reglas simples y comparar con selección aleatoria repetida, igual ponderación y una covariable de momentum simple. No escoger el top retrospectivamente.

E4 — **Salida**: comparar salida diaria, semanal y combinada con controles de tiempo fijo y buy-and-hold del mismo ticker. Analizar salida retrasada, falsos cambios y periodo sin posición.

E5 — **Cartera y recompras**: sólo si E1–E4 aportan evidencia. Simular estado de efectivo/acciones, cupos, rotación y operación posterior al evento; comparar patrimonio neto y acciones finales con buy-and-hold. Costos como curva de sensibilidad, sin fingir conocer un costo fijo universal.

## Validación

- Ventanas de entrenamiento y prueba cronológicas, embargo cuando las etiquetas futuras se solapen, y prueba final intacta.
- Repetir por periodos/regímenes y grupos de tickers; reportar tamaño efectivo por fechas, no sólo número de ticker-días.
- Congelar definiciones y candidatos antes de evaluar cada bloque. Registrar todos los candidatos probados, incluidos los adversos.
- No afirmar valor operativo por una correlación significativa pequeña o un solo periodo favorable.

## Primera entrega ejecutable propuesta

Un panel EOD point-in-time con `Ticker`, `Session`, OHLCV, SMI diario confirmado, último SMI semanal confirmado, eventos Hunter diarios y desenlaces futuros. Debe incluir auditoría de disponibilidad, pruebas sintéticas anti-lookahead y comparación con el baseline sin filtro. El componente 1H/4H del Monitor se incorpora sólo después de reconciliar su comportamiento con TradingView y disponer de cobertura intradía suficiente.

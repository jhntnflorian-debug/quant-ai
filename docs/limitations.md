# Limitaciones, fallos encontrados y riesgos pendientes

Actualizado: 2026-11-09.

## Resultados

- **El modelo ML base no supera a comprar y mantener** en el walk-forward 2020-01 a 2025-10: Sharpe 0.72 frente a 1.10. Tampoco supera al cruce EMA 24/168 (1.01). Su ventaja es el riesgo: drawdown de −49 % (−35 % con el motor de riesgo) frente a −80 %. La puerta de promoción lo rechazó y el modelo registrado es solo candidato.
- La señal es débil: AUC entre 0.544 y 0.554, y un Brier apenas mejor que la tasa base.
- Fuerte sensibilidad a costes: con costes ×3 el ensemble pierde (−26 %). Los costes acumulados equivalen al 115 % del capital inicial.
- El periodo 2020 a 2025 incluye un mercado alcista excepcional en cripto. Comprar y mantener es un baseline muy difícil de batir en ese tramo y sus resultados no se pueden extrapolar.
- Los IC95 % del Sharpe (bootstrap de bloques de 168 h) son anchos: el del ensemble va de −0.09 a 1.43. No hay evidencia estadística de que ninguna estrategia sea mejor que otra.
- El holdout final (desde 2025-11-01) sigue sin tocar. Ningún número de este proyecto proviene de él.

### Ronda v2 (Colab, 2026-11-10, tramo 2022-01 a 2025-10)

- **Ningún candidato v2 pasó la puerta**: 8 variantes ML, GRU, TCN y CNN de gráficos. El detalle está en el README y en `reports/runs/20261110T120156Z_v2final/report.md`.
- AUC fuera de muestra entre 0.49 y 0.54. Los indicadores técnicos, los datos externos, el meta-etiquetado y el volatility targeting no aportaron una ventaja medible.
- Las probabilidades calibradas rondan 0.45 y casi nunca superan el umbral de entrada de 0.55: los modelos pasan casi todo el tiempo en efectivo (9 a 84 operaciones en casi cuatro años). Sus retornos dependen de muy pocas operaciones.
- El mejor Sharpe (v3, 0.69) no es distinguible del de comprar y mantener (0.52): el IC95 % de la diferencia va de −1.26 a 1.45. Su retorno se concentra en 2025.
- El diseño de la ronda 1 (base, 12 h), re-evaluado en este tramo y con el motor actual, pierde: −5.7 %, Sharpe −0.12. El modelo fijado para paper pertenece a ese diseño.
- Van 30 intentos acumulados sobre el periodo de desarrollo. Cada intento nuevo sobre los mismos datos sube el listón del Deflated Sharpe.

## Fallos encontrados y corregidos durante el desarrollo

- Los archivos de data.binance.vision usan microsegundos desde 2025-01 (antes, milisegundos). Se normaliza por magnitud.
- Había una vela fuera de la grilla horaria (2018-02-09 09:28:14, reanudación tras un mantenimiento de Binance). Se alinea a la hora y se reporta.
- El primer "comprar y mantener" rebalanceaba a 50/50; ahora compra una sola vez.
- La regla "precio sobre EMA24 y EMA168" perdió −95.7 % por whipsaw y costes. Se conserva documentada y se agregó el cruce clásico EMA 24/168.
- En modo de prueba mínima, los baselines y el ML se comparaban sobre periodos distintos. Ahora todas las estrategias usan el mismo tramo.
- El límite absoluto por orden (3,000 USDT) era incompatible con el peso objetivo (50 % de 10,000) y habría bloqueado todas las compras en paper. Ahora es relativo al capital (55 %), con un tope absoluto de 20,000.
- `.gitignore` excluía `src/quant_ai/data` y `src/quant_ai/models`. Ahora las rutas están ancladas a la raíz (detectado por Codex).

## Limitaciones técnicas conocidas

- **Deep learning**: GRU y TCN entrenadas en Colab (2026-11-09). AUC fuera de muestra entre 0.50 y 0.53, sin mejora sobre el ensemble ML. El Transformer no se entrenó. Las pruebas unitarias de DL y RL todavía no se han ejecutado nunca: en local se omiten porque no hay torch ni gymnasium. El cuaderno ya tiene una etapa `pruebas` que corre las de DL en la nube antes de entrenar (las de RL solo con la etapa `rl` activa); su primer resultado llegará con la próxima sesión.
- **RL**: PPO perdió en 5 de 6 semillas y operó en exceso (307 a 729 operaciones). La penalización de rotación (0.001) es demasiado baja y 200,000 pasos son pocos. No es utilizable.
- **Benchmarks de GPU**: solo se midió la L4. Al ser modelos pequeños, el cuello de botella es la carga de datos, no la GPU.
- En GRU/TCN v1 el tramo de calibración se usa también para el early stopping y para ajustar Platt. El test sigue aislado, pero la calibración puede ser algo optimista. Para v2 está resuelto con `--separate-validation` (la sección 7b del cuaderno la pasa): el early stopping usa el último 20 % del ajuste, purgado contra esa validación, y la calibración queda solo para Platt, igual que en la CNN. Sin la bandera, v1 se reproduce exactamente como estaba.
- La incertidumbre del ensemble es la dispersión entre dos modelos, no un intervalo de confianza formal.
- El backtest no modela latencia ni el libro de órdenes. El spread se parametriza: 1 bps más 2 bps de slippage por lado, un supuesto conservador para BTC y ETH. En paper se usan el bid y el ask reales.
- En el backtest, el límite de participación usa el volumen de la vela anterior. Los stops se ejecutan al peor precio entre la apertura y el stop.
- En el backtest, el drawdown que activa el motor de riesgo se emula con una pausa de 30 días y un reinicio del pico. En vivo exige revisión manual.
- Las características ignoran huecos de datos (27 huecos y 120 velas faltantes desde 2018): las ventanas móviles cruzan el hueco. Las etiquetas sí se anulan cuando el horizonte cruza un hueco.
- El adaptador de Binance Spot Testnet no se ha probado contra el exchange porque faltan claves. Hay que crearlas en https://testnet.binance.vision/.
- Paridad de stops: en paper el stop-loss se revisa una vez por hora, en cada ciclo, contra el bid actual. En el backtest se evalúa con el mínimo intrabar de la vela. En paper un stop puede ejecutarse más tarde y a peor precio que en el backtest.
- El modelo de paper queda fijado en `state/paper/MODEL`. Para cambiarlo hay que editar ese archivo a mano, después de revisar la puerta de promoción.
- `docker/Dockerfile` y `docker/compose.yaml` no se han construido ni probado.
- No hay alertas externas (correo o Telegram) implementadas; solo `heartbeat.json`.
- El dashboard no tiene autenticación. Úsalo en `localhost` o detrás de un túnel.

## Ronda v2: correcciones tras la revisión adversarial de Codex (2026-11-09)

Todas se aplicaron ANTES de cualquier corrida completa de v2:
- Registro de intentos por configuración (GRU v1 y v2 ya no se fusionan) y dispersión del DSR con todos los intentos acumulados.
- Puerta v2 estricta: rechaza pruebas mínimas, métricas no finitas y referencias ausentes, y exige que el IC95 % del bootstrap pareado de la diferencia de Sharpe excluya 0.
- Pesos ausentes dentro del tramo = efectivo (antes un trimestre sin predicción podía congelar una posición).
- Mismo tratamiento de stops para ML, DL y reglas de tendencia también en la evaluación de DL.
- Tras un stop-loss se exige una señal nueva (antes la histéresis podía recomprar con p = 0.50). **Este cambio del motor no estaba cuando se calcularon las cifras de v1 del README**; la consolidación v2 re-evalúa todo con el motor actual.
- Volatility targeting con su sentido correcto: min(máximo, objetivo / volatilidad); objetivo de 30 % por posición.
- Variantes añadidas: v2b (datos externos sin técnico) y v6 (meta-etiqueta con triple barrera alineada con el stop).

## Correcciones previas a la corrida v2 (2026-11-09)

Se aplicaron antes de que existiera cualquier resultado v2. La sesión de Colab del 2026-11-10 las ejecutó completas: todas las etapas terminaron bien y las 42 pruebas con PyTorch pasaron. En Kaggle no se han ejecutado.
- **Registro de intentos del DSR.** El Sharpe de cada idea salía de la primera corrida en orden alfabético, que a menudo era una prueba mínima: `ml_ensemble` valía 1.59 (smoke de 1 fold, solo BTC, 6 meses) en lugar del 0.72 de la corrida completa, y `ml_v0_base_h12` −2.84 (smoke de 3 meses). Ahora el Sharpe sale de la corrida completa más reciente. Una idea vista solo en pruebas mínimas sigue contando como intento, pero sin Sharpe. Las carpetas smoke de DL y RL ya no crean identidades nuevas, y un Sharpe de 0.0 ya no se descarta como NaN. Siguen siendo 22 intentos, 18 de ellos con Sharpe. La desviación estándar de los Sharpe anuales que entran al DSR bajó de 1.39 (21 valores) a 1.16 (18 valores).
- **Caché de imágenes de la CNN.** Antes cada imagen se renderizaba en cada época: unos 13 millones de renders en la corrida completa, es decir, horas de CPU. Ahora cada activo pre-renderiza una sola vez las posiciones que usa en la RAM de la VM, con todos los núcleos. Son menos de 0.8 GB por activo y la caché se libera antes de pasar al siguiente. Las imágenes son idénticas byte a byte a las de antes. `--no-cache` vuelve al modo anterior.
- **Exportación incremental.** El ZIP de resultados se actualiza tras cada etapa v2 con un reemplazo atómico: en Colab queda en Drive y en Kaggle en `/kaggle/working`. Un corte ya no se lleva lo que estaba terminado. Una celda final resume cada etapa con su estado y duración y lo guarda en `reports/colab_sessions/resumen_<sesión>.json`.
- **Diagnóstico de errores.** Si un script falla, `run_step` repite sus últimas 40 líneas (stdout y stderr) bajo «ÚLTIMAS LÍNEAS DEL ERROR» y las guarda en el registro de la sesión. Si la descarga de datos falla, en Colab y en Kaggle se reintenta con `--no-rest`; si también falla pero los datos guardados cubren todo el periodo de desarrollo, se continúa con un aviso.
- **Pruebas en la nube.** La etapa `pruebas` ejecuta con torch las pruebas de secuencias, CNN, indicadores técnicos, experimento v2 y backtest antes de entrenar. Si alguna falla, la corrida no se detiene.

## Riesgos pendientes

- Cambio de régimen: los modelos se entrenaron con 2018 a 2025. El detector de fuera de distribución solo cubre desvíos extremos (|z| > 6).
- Riesgo de plataforma: cambios en la API, en los términos o en la lista de países de Binance (la lista oficial verificada está fechada en 2024-02-27).
- Data snooping acumulado: cada iteración sobre el periodo de desarrollo consume validez estadística. El holdout final debe usarse una sola vez, con el diseño congelado.
- El modo LIVE no existe en el código (está bloqueado por diseño). Habilitarlo exige tu autorización expresa, una revisión de controles y semanas de paper estable.

## Paper trading: re-entrada tras stop-loss y defectos pendientes (2026-11-09)

**Corregido.** El ciclo en vivo (`execution/runner.py`) aplica ya las dos reglas del backtest después de un stop-loss ejecutado:
- Un enfriamiento de `stop_cooldown_bars` velas (24 por defecto), medido en tiempo de vela, que también corre mientras el proceso está apagado.
- Una señal nueva: no se vuelve a comprar hasta ver una vela decidida con predicción inválida o con probabilidad menor que `exit_prob`.

El estado se guarda por símbolo en `state/paper/portfolio.json` (clave `stop_guard`) y cada compra bloqueada queda en `decisions.jsonl` como HOLD, con su motivo. Una prueba pasa la misma secuencia de 27 velas por `run_backtest` y por `run_cycle` y obtiene las mismas entradas y los mismos stops.

Diferencias que siguen abiertas frente al backtest:
- En vivo el stop se revisa una vez por ciclo contra el bid; el backtest usa el mínimo de la vela. En vivo puede dispararse más tarde, a peor precio o no dispararse.
- Si durante el enfriamiento la señal se apaga y vuelve a encenderse, el backtest puede recomprar con p entre `exit_prob` y `enter_prob`; en vivo se exige p ≥ `enter_prob`.
- En el experimento v2 también levantan la espera los ceros de la señal primaria y del volatility targeting; en vivo solo la probabilidad.
- Las velas que el proceso no llegó a decidir no levantan la espera en vivo, lo que lo hace más conservador.

**Defectos pendientes del ciclo de paper** (encontrados en la revisión, sin corregir todavía). No afectan al entrenamiento:
- Si falla la actualización de datos, el cálculo de características, la carga del modelo o `decide`, el ciclo aborta antes de revisar los stops. Mientras dure la falla, las posiciones abiertas quedan sin stop.
- Un símbolo sin modelo nunca revisa su stop.
- Con un broker que no sea el local, dos compras en el mismo ciclo borran el stop de la primera.
- Si el ATR no es finito, se compra sin stop.
- Tras un error del broker, el reintento del cierre en la misma vela se rechaza como orden duplicada.
- Las posiciones reconciliadas sin stop (saldos previos del testnet o compras manuales) nunca reciben uno.
- La recarga de la cartera tras una orden pierde `day`, `day_start_equity` y `peak_equity` fijados en ese ciclo.

**Descarga de datos.** `scripts/download_data.py` ya no se detiene si la API REST de Binance falla (por ejemplo, HTTP 451 desde la región de la VM): avisa, sigue con los datos guardados y solo termina con error si no cubren el periodo de desarrollo. El ciclo de paper mantiene el comportamiento estricto.

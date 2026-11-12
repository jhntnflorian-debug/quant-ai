# QUANT-AI

Sistema de trading algorítmico con IA para BTCUSDT y ETHUSDT spot (Binance, velas de 1 hora, long/flat, sin apalancamiento). Funciona en modo **PAPER** por defecto. El modo LIVE está bloqueado por código hasta que el propietario lo autorice de forma expresa.

Esta es la versión MVP 0.1 (2026-11-08): datos reales, validación walk-forward purgada, backtests con costes, motor de riesgo con veto, paper trading con precios en vivo, dashboard y notebooks para entrenar en Google Colab.

## Resultados medidos (fuera de muestra, 2020-01-01 a 2025-10-31)

> Nota: estas cifras v1 se calcularon antes de añadir al motor la regla de «señal nueva tras un stop» (2026-11-09). La ronda v2 re-evalúa todo con el motor actual.

Corrida `20261108T232548Z_full`: 48 folds trimestrales (2 activos × 24), costes de 10 bps de comisión, 1 bps de medio spread y 2 bps de slippage por lado. Señal al cierre de cada vela, ejecución en la apertura siguiente. Capital inicial de 10,000 USDT y 50 % por activo.

| Estrategia | Retorno | Sharpe (IC95 %) | Máx. drawdown | Operaciones |
|---|---|---|---|---|
| Comprar y mantener 50/50 | +2152.8 % | 1.10 (0.33, 1.93) | −79.9 % | 0 |
| Cruce EMA 24/168 | +650.2 % | 1.01 (0.09, 1.82) | −55.6 % | 488 |
| Momentum 168 h | +136.1 % | 0.56 | −69.0 % | 1,725 |
| Reversión RSI | −62.5 % | −0.24 | −70.3 % | 628 |
| ML ensemble (LR + LightGBM) | +146.4 % | 0.72 (−0.09, 1.43) | −49.1 % | 465 |
| ML ensemble + motor de riesgo | +154.4 % | 0.88 (0.01, 1.70) | −35.1 % | 316 |
| ML ensemble con costes ×2 | +34.7 % | 0.33 | −53.4 % | 465 |
| ML ensemble con costes ×3 | −26.3 % | −0.06 | −58.1 % | 465 |

Lectura honesta:

- El modelo ML base **no supera** a comprar y mantener ni al cruce EMA en Sharpe. Su aporte es la reducción del drawdown: tiene alguna posición abierta solo el 15.5 % del tiempo, con una exposición media del 9.8 % del capital, y en 2022 perdió −13 % frente a −67 % de comprar y mantener.
- Calidad predictiva débil: AUC entre 0.544 y 0.554. Cuando p ≥ 0.55 acierta entre 56 % y 60 %, frente a una tasa base de 43 % a 45 %, pero eso ocurre solo en el ~4 % de las velas.
- La puerta de promoción lo rechazó: el modelo registrado es **candidato**, no activo.
- El holdout final (desde 2025-11-01) está sellado y no se ha usado.

El detalle está en [reports/runs/20261108T232548Z_full/report.md](reports/runs/20261108T232548Z_full/report.md) y en el dashboard.

**Reproducibilidad:** la misma corrida, ejecutada en Colab con Python 3.13, pandas 2.2, scikit-learn 1.6 y LightGBM 4.6, dio cifras idénticas a las de este PC (Python 3.11, pandas 3.1, scikit-learn 1.9, LightGBM 4.7).

## Deep learning y RL (Colab, GPU L4, 2026-11-09)

Deep learning: últimos 4 trimestres del periodo de desarrollo (2025-01-01 a 2025-10-31), BTC + ETH, todas las estrategias sobre el mismo tramo.

| Estrategia | Retorno | Sharpe (IC95 %) | Máx. drawdown | Operaciones | AUC BTC / ETH |
|---|---|---|---|---|---|
| TCN (97k parámetros) | +21.8 % | 1.46 (−0.32, 3.68) | −10.5 % | 48 | 0.526 / 0.496 |
| GRU (45k parámetros) | +17.8 % | 1.36 (−0.36, 2.76) | −8.9 % | 35 | 0.526 / 0.520 |
| Comprar y mantener | +15.1 % | 0.58 | −44.3 % | 0 | — |
| ML ensemble | +7.7 % | 1.00 | −4.8 % | 22 | 0.535 / 0.516 |
| Cruce EMA 24/168 | −2.6 % | 0.07 | −38.8 % | 68 | — |

- El deep learning no predice mejor que el ensemble ML: todos los AUC están entre 0.50 y 0.54. Su mayor retorno viene de una cobertura algo mayor en un tramo alcista de BTC.
- Con 35 a 48 operaciones en 10 meses, todos los IC95 % incluyen 0. No hay evidencia de ventaja, y elegir "el mejor" de varios modelos en este mismo tramo sobreestimaría su rendimiento.

RL (PPO, 200,000 pasos, 3 semillas, evaluación 2024-01-01 a 2025-10-31, 50 % del capital):

| Activo | Comprar y mantener | PPO por semilla | Operaciones | Costes/capital |
|---|---|---|---|---|
| BTC | +79.0 % | −29.5 % / +4.5 % / −20.0 % | 307 a 478 | 32 % a 48 % |
| ETH | +33.7 % | −35.4 % / −47.0 % / −35.7 % | 410 a 729 | 45 % a 50 % |

El RL fracasó: aprendió a operar en exceso y los costes lo destruyeron. Es inestable entre semillas.

Benchmark de la L4 (medido): matmul de 64.6 TFLOPS en BF16 y 12.4 en FP32. Entrenamiento GRU de 257,000 muestras/s con batch 1024 y 0.27 GB de VRAM. Cada modelo DL tardó 71 a 88 s en total (8 folds), usando menos del 1 % de los 22 GB de VRAM: una GPU más grande no acortaría el entrenamiento.

## Ronda v2: resultados medidos (Colab, 2026-11-10)

Walk-forward de 2022-01-01 a 2025-10-31, BTC y ETH, costes de 13 bps por lado. Corrida `20261110T120156Z_v2final`: todos los candidatos evaluados juntos, con los mismos stops y las mismas referencias. **Ningún candidato pasó la puerta de promoción.**

| Estrategia | Retorno | Sharpe | Máx. DD | Operaciones | DSR | Sharpe frente a comprar y mantener (IC95 %) |
|---|---|---|---|---|---|---|
| ML base, 24 h (v1) | +18.3 % | 0.65 | −9.8 % | 25 | 0.008 | +0.13 (−1.31 a 1.76) |
| ML + técnico (v2) | +22.3 % | 0.50 | −13.4 % | 39 | 0.003 | −0.03 (−1.36 a 1.16) |
| ML + datos externos (v2b) | +5.5 % | 0.19 | −18.3 % | 30 | 0.001 | −0.33 (−1.63 a 1.01) |
| ML + técnico + externos (v3) | +28.9 % | 0.69 | −15.5 % | 35 | 0.010 | +0.17 (−1.26 a 1.45) |
| Meta-etiquetado (v4, v5, v6) | −2.1 % a +1.3 % | −0.11 a 0.21 | −2.0 % a −8.6 % | 9 a 21 | 0.001 o menos | −0.64 a −0.32 |
| ML base, 12 h (v0, diseño de la ronda 1) | −5.7 % | −0.12 | −21.5 % | 57 | 0.000 | −0.64 (−1.85 a 0.65) |
| GRU v2 | −3.2 % | −0.11 | −9.4 % | 31 | 0.000 | −0.63 (−2.12 a 0.73) |
| TCN v2 | +4.7 % | 0.20 | −10.4 % | 84 | 0.001 | −0.33 (−1.62 a 0.97) |
| CNN de gráficos | −10.1 % | −1.20 | −12.9 % | 20 | 0.000 | −1.73 (−3.19 a −0.15) |
| Comprar y mantener | +69.0 % | 0.52 | −69.6 % | 0 | — | referencia |
| Cruce EMA 24/168 | −21.3 % | −0.05 | −42.3 % | 318 | — | referencia |
| Cruce EMA + volatility targeting | −13.0 % | −0.01 | −37.1 % | 318 | — | referencia |

Lectura:
- **No hay señal predictiva demostrable.** El AUC fuera de muestra va de 0.49 a 0.54 en ML y de 0.49 a 0.52 en GRU, TCN y CNN. Agregar 82 indicadores técnicos y 32 variables externas no lo movió.
- **Los modelos casi nunca operan.** La probabilidad calibrada ronda 0.45 y supera el umbral de entrada (0.55) en menos del 3 % de las velas. El mejor candidato (v3) no hizo ninguna operación en 2023, cuando comprar y mantener rindió +125 %, y casi todo su retorno sale de 2025 (+37.4 %).
- **Ninguna diferencia frente a comprar y mantener es estadísticamente distinguible de cero**, salvo la CNN, que es peor. Con 30 intentos acumulados, el Deflated Sharpe de todos queda en 0.01 o menos.
- Lo único consistente es el menor drawdown (−10 % a −15 % frente a −70 %), que se explica por pasar casi todo el tiempo en efectivo.
- El holdout final (desde 2025-11-01) sigue sin tocar.

Sesión de Colab medida (GPU A100, 1 h 17 min): datos externos 5.7 min, experimento v2 23.7 min, GRU 6.0 min, TCN 6.0 min, CNN 29.8 min, consolidación 0.9 min. Las 42 pruebas con PyTorch pasaron en la nube.

## Ronda v2 (2026-11-09): más técnicas y datos de otras plataformas

Trabajo conjunto con Codex (GPT-6-Astra, esfuerzo medio). Los resultados están en la sección anterior.

| Componente | Contenido | Archivo |
|---|---|---|
| Análisis técnico ampliado | 82 características causales: Ichimoku, Donchian y breakouts, Keltner, Supertrend, Aroon, estocástico, StochRSI, CCI, Williams %R, divergencias RSI, OBV, MFI, CMF, CVD (volumen agresor), Amihud, volatilidad Parkinson/Garman-Klass/Yang-Zhang, velas japonesas (envolventes, martillo, estrella fugaz, doji, inside/outside), soportes y resistencias por pivotes confirmados, estructura de mercado (HH/HL), marcos de 4 h y diario (solo velas cerradas), autocorrelación, varianza, diferenciación fraccional y sesiones | `features/technical.py` |
| Datos de otras plataformas | 32 características con hora real de disponibilidad: funding, prima, basis, open interest y ratios long/short de Binance; prima de Coinbase; Deribit DVOL y su diferencia con la volatilidad realizada; Fear & Greed; stablecoins; VIX, S&P 500, bonos a 10 años y dólar (FRED); amplitud de altcoins y on-chain | `data/external.py`, `features/external.py` |
| Meta-etiquetado | La señal de tendencia (cruce EMA) decide la dirección y el modelo decide si tomarla | `training/experiment_v2.py` |
| Volatility targeting | Tamaño según la volatilidad de 7 días, cuantizado para no rotar de más | `decision_engine/policy.py` |
| CNN de gráficos | Imágenes de 48 velas al estilo de Jiang, Kelly y Xiu (2023) | `training/chart_images.py`, `models/chart_cnn.py` |
| Rigor estadístico | 8 variantes pre-registradas, ablación por grupos, 3 escenarios de costes (incluido BNB a 7.5 bps), PSR y Deflated Sharpe sobre todos los intentos acumulados, bootstrap pareado frente a las referencias, puerta v2 estricta y consolidación final con trato idéntico | `configs/experiments_v2.yaml`, `validation/trials.py`, `scripts/consolidate_v2.py` |

Comprobaciones:
- 77 pruebas en local, incluidas pruebas anti-fuga de cada característica nueva.
- Revisión adversarial de Codex: 7 problemas reales corregidos antes de entrenar (ver [docs/limitations.md](docs/limitations.md)).
- Verificación point-in-time con datos reales descargados: alterar todas las fuentes después de una fecha no cambia ninguna fila anterior.
- Las variantes v2 no se operan en paper hasta implementar su cálculo en vivo.

## Estado de los entregables (2026-11-08)

| # | Entregable | Estado |
|---|---|---|
| 1-2 | Investigación y comparativas con fuentes | Hecho: [docs/research.md](docs/research.md), con verificación marcada dato por dato |
| 3 | Arquitectura con diagramas | Hecho: [docs/architecture.md](docs/architecture.md) (Mermaid, también en el dashboard) |
| 4 | Dataset reproducible con procedencia | Hecho y verificado: 77,495 velas por par, SHA256 por archivo, `data/manifest.json` |
| 5 | Pipeline de características | Hecho y verificado: 39 características con pruebas anti-fuga |
| 6 | Modelos base | Hecho y medido: 5 reglas + LR + LightGBM + ensemble calibrado |
| 7 | Notebook de Colab | Hecho y ejecutado en Colab (GPU L4, 43 min en total, todas las etapas terminadas) |
| 8 | Benchmarks de GPU | L4 medida. G4, A100 y T4 sin medir (innecesario: la L4 usa menos del 1 % de su VRAM) |
| 9 | Entrenamiento y checkpoints DL | Hecho: GRU y TCN entrenadas en la L4; checkpoints en Drive |
| 10 | Backtesting con costes realistas | Hecho y verificado (coincide con el cálculo manual de comprar y mantener) |
| 11 | Reporte fuera de muestra | Hecho: walk-forward 2020-2025. El holdout final sigue sellado |
| 12 | Motor de riesgo independiente | Hecho y probado (kill switch, OOD por activo, límites, emergencia) |
| 13 | Paper trading | Hecho: broker local con precios reales de Binance. Adaptador de testnet escrito; **faltan tus claves** |
| 14 | Dashboard con datos reales | Hecho: 6 páginas, 17 pruebas |
| 15 | Documentación de instalación | Hecho (este README y docs/) |
| 16 | Pruebas automatizadas | 51 pasan en local; las de DL y RL se omiten sin torch ni gymnasium (sus scripts sí corrieron en Colab) |
| 17 | Plan de despliegue y presupuesto | Hecho: [docs/deployment.md](docs/deployment.md). Docker escrito, sin construir |
| 18 | Limitaciones y riesgos | Hecho: [docs/limitations.md](docs/limitations.md) |
| — | Aprendizaje por refuerzo | Entrenado (PPO, 3 semillas por activo): fracasó frente a comprar y mantener |

## Instalación local (liviana: sin PyTorch)

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
```

PyTorch, Gymnasium y Stable-Baselines3 no se instalan en el PC. Se usan en Colab (extras `dl` y `rl`).

## Uso

```powershell
# 1. Datos reales de Binance (incremental, ~21 MB)
.venv\Scripts\python.exe scripts\download_data.py

# 2. Experimentos (en local se limitan a 2 hilos)
.venv\Scripts\python.exe scripts\run_experiments.py --skip-ml            # solo baselines, ~10 s
.venv\Scripts\python.exe scripts\run_experiments.py --register           # walk-forward ML completo, ~2 min

# 3. Paper trading con precios en vivo (sin claves ni dinero real)
.venv\Scripts\python.exe scripts\run_paper.py --once                     # un ciclo
.venv\Scripts\python.exe scripts\run_paper.py                            # un ciclo por hora

# 4. Dashboard
.venv\Scripts\python.exe -m streamlit run dashboard\app.py --server.address localhost

# 5. Pruebas
.venv\Scripts\python.exe -m pytest -q
```

Kill switch: crear un archivo `KILL_SWITCH` en la raíz del proyecto bloquea toda compra nueva.

## Entrenamiento en Google Colab

1. `.venv\Scripts\python.exe scripts\package_for_colab.py` genera `dist/quant-ai.zip`: el código más `estado_previo/` (datos limpios e historial de intentos para el Deflated Sharpe; el cuaderno lo copia sin sobrescribir nada).
2. Súbelo a `Mi unidad/quant-ai/quant-ai.zip` y abre `notebooks/colab_training.ipynb` en Colab.
3. Ejecuta primero con `SMOKE=True`. Datos, modelos, checkpoints y reportes quedan en Drive, y si la sesión se corta, volver a ejecutar la celda reanuda el entrenamiento.

Guía completa: [docs/colab.md](docs/colab.md). Benchmark de GPU: `notebooks/gpu_benchmark.ipynb`.

## Estructura

```text
configs/base.yaml          configuración; holdout y umbrales pre-registrados
src/quant_ai/
  data/                    descarga (data.binance.vision + REST), calidad, almacenamiento con manifiesto
  features/                39 características causales por activo
  labeling/                retorno neto de costes y triple barrera
  validation/              walk-forward purgado con embargo y guardia del holdout
  models/                  baselines, ensemble LR + LightGBM calibrado, GRU/TCN/Transformer
  training/                dataset, experimentos, entrenador PyTorch (Colab)
  reinforcement_learning/  entorno Gymnasium (Colab)
  backtesting/             motor vela a vela y métricas con bootstrap
  decision_engine/         política long/flat con abstención y decisiones estructuradas
  risk_management/         motor de riesgo independiente con veto y kill switch
  brokers/                 interfaz común, paper local, Binance Spot Testnet; LIVE bloqueado
  execution/               ciclo de paper trading auditable
dashboard/                 Streamlit + Plotly (6 páginas)
notebooks/                 Colab: entrenamiento, benchmark de GPU, exploración
scripts/                   CLIs
tests/                     pruebas (fuga temporal, backtest, riesgo, ejecución, dashboard, DL y RL)
docs/                      investigación, arquitectura, Colab, despliegue, limitaciones
docker/                    imagen y compose para un VPS
```

Datos, modelos, reportes y estado quedan fuera de Git y se versionan con hashes en `data/manifest.json` y en `models/registry/<versión>/meta.json`.

## Documentación

- [docs/research.md](docs/research.md): investigación de mercados, proveedores, plataformas y frameworks, con fuentes y estado de verificación.
- [docs/architecture.md](docs/architecture.md): diagramas Mermaid, módulos, dimensiones y motor de riesgo.
- [docs/colab.md](docs/colab.md): entrenamiento en Colab, reanudación, cambio de GPU y presupuesto.
- [docs/deployment.md](docs/deployment.md): despliegue 24/7 y presupuesto.
- [docs/limitations.md](docs/limitations.md): limitaciones, fallos corregidos y riesgos.

## Avisos

Resultados simulados: no son beneficios reales. El modo LIVE no está implementado y está bloqueado. Para operar con dinero real hacen falta la autorización expresa del propietario, una revisión de controles y un periodo de paper estable.

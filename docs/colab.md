# Entrenamiento de quant-ai en Colab

Este flujo investiga BTCUSDT/ETHUSDT spot 1h, long/flat, en PAPER. Los modelos
secuenciales se entrenan en Colab. No instale torch ni ejecute entrenamientos,
benchmarks o experimentos ML en el PC de trabajo. TPU no está soportado.

## 1. Preparar el código

Genera el ZIP con el empaquetador del proyecto (no requiere commits de Git):

```powershell
.venv\Scripts\python.exe scripts\package_for_colab.py
```

Crea `dist/quant-ai.zip` con `src`, `scripts`, `configs`, `notebooks`, `tests`, `docs`,
`dashboard`, `pyproject.toml` y `README.md`, más una carpeta `estado_previo/` con los datos
limpios del PC y el historial de intentos (resúmenes de corridas anteriores). El cuaderno copia
`estado_previo/` a Drive sin sobrescribir nada: así el Deflated Sharpe cuenta también las ideas
probadas solo en el PC. Excluye modelos, predicciones, estado de paper, `.venv`, `.env` y cachés.
Con `--no-state` el ZIP lleva solo código.

Sube `dist/quant-ai.zip` a `Mi unidad/quant-ai/quant-ai.zip`. Abra
`notebooks/colab_training.ipynb` con Colab. Como alternativa, configure `REPO_URL`
con un repositorio accesible que contenga estos archivos. Con el ZIP, cada sesión
reemplaza la copia del código en la VM por la del ZIP más reciente de Drive (datos,
modelos y reportes viven en Drive, no ahí); con `REPO_URL` se conserva el checkout
existente.

## 2. Ejecutar desde cero

1. Seleccione un entorno GPU CUDA y Python compatible con el `pyproject.toml`
   existente: 3.10 o superior (probado en 3.11). El notebook verifica versión, GPU, VRAM, compute capability,
   bf16, torch/CUDA/cuDNN, CPUs, RAM, disco y `nvidia-smi`.
2. Configure la celda 0: `REPO_URL` opcional, `CODE_ZIP`, `DRIVE_DIR`, `MODELOS`,
   `RUN_TAG`, semilla y presupuesto. Use primero `SMOKE=True` en Colab para revisar
   el flujo. Smoke no sirve para comparar rentabilidad.
3. Ejecute las celdas 1–3: montar Drive, exportar `QUANT_AI_HOME`, obtener el código
   e instalar el paquete editable con `pip install -e . --no-deps`. Se instalan
   `pyarrow`, `pyyaml`, `lightgbm` y `scikit-learn` solamente si faltan. No se
   reinstala torch. Este paso está bloqueado fuera de Colab.
   La sección 3b (etapa `pruebas`) corre después las pruebas que en el PC se omiten
   por falta de torch (ver «Pruebas en la nube»).
4. Ejecute la descarga. La primera descarga también puede tardar en modo smoke.
   Si falla (por ejemplo, la API REST de Binance bloqueada en la región de la VM),
   se reintenta solo con los archivos mensuales (`--no-rest`). Si también falla pero
   los datos guardados en Drive llegan hasta la última vela antes del holdout, se
   continúa con un aviso; si no, la celda se detiene y explica qué hacer.
5. Ejecute el experimento ML en CPU con `--n-jobs -1 --register`. Smoke lo limita
   a un fold y un símbolo y omite el registro de producción.
6. Ejecute el benchmark y revise formatos soportados, memoria y muestras/s.
7. Ejecute los modelos secuenciales. Se guardan en
   `Drive/quant-ai/models/dl/<RUN_TAG>/<modelo>/` (sufijo `_smoke` cuando aplica).
8. Revise las tablas de evaluación y el presupuesto de la sesión.

Los comandos pasan `--config /content/quant-ai/configs/base.yaml` explícitamente.
Con la instalación editable la ruta por defecto ya resuelve a `configs/base.yaml` de la
raíz; pasarla explícitamente evita sorpresas si alguien instala el paquete sin `-e`. `QUANT_AI_HOME` mantiene datos/modelos/reportes
en Drive; los nuevos scripts usan la raíz del repositorio cuando la variable falta.

## 3. Reanudar

Después de reconectar, ejecute las celdas 0–3 y 7, conservando los parámetros y
`RUN_TAG`. No necesita repetir descarga, ML ni benchmark. Ejemplo en Colab:

```bash
python scripts/train_sequence.py --config configs/base.yaml --model gru \
  --out /content/drive/MyDrive/quant-ai/models/dl/secuencias_v1/gru --resume
```

Cada símbolo/fold tiene su propia carpeta:

```text
<out>/
  experiment.json
  run_info.json
  metrics.csv
  predictions.parquet
  backtest_summary.json
  equity.parquet
  folds/BTCUSDT/fold_023/
    checkpoints/last.pt
    checkpoints/best.pt
    metrics.csv
    run_info.json
    calibrator.json
    predictions.parquet
```

Dentro del `out` de cada entrenador, `checkpoints/last.pt` contiene la última época
completa y `best.pt` la de menor `val_loss`. Ambos guardan modelo, optimizador,
scheduler, scaler, época, mejor pérdida, paciencia, RNG Python/NumPy/torch/CUDA y
muestreo, hiperparámetros y normalización. La escritura usa archivo temporal y
reemplazo; conserve también una copia del experimento en Drive.

Una interrupción a mitad de época repite esa época desde `last.pt`. Los folds
terminados restauran pesos y regeneran predicciones sin volver a entrenarse. No
cambie datos, arquitectura, épocas, batch, acumulación ni semilla al reanudar:
se verifica una huella del dataset y OneCycle conserva su horizonte original.
Para cambiar esos parámetros, use otro `--out`/`RUN_TAG`. `--resume` exige `--out`.
Un corte abrupto del runtime puede dejar `status=running`; no impide reanudar.
`interrupted` se registra cuando Python recibe una interrupción controlada.

## 4. Diseño y límites de evaluación

- GRU usa su último estado. TCN usa convoluciones dilatadas con padding izquierdo
  y bloques residuales. Transformer usa embedding posicional aprendido, máscara
  causal y el último paso. Los tres producen un logit por ventana.
- Una ventana termina exactamente en `t`, incluida. Las ventanas son vistas del
  array float32 compartido, no una matriz preconstruida de todas las secuencias.
  Se excluyen ventanas que no contengan horas consecutivas.
- Media y desviación se calculan solo en filas de ajuste; NaN/inf se convierten
  en cero después de normalizar. El contexto anterior al inicio de calibración
  o prueba es válido: ya se conocía en la fecha de la predicción.
- Los folds reutilizan `purged_walk_forward` sin alterar purga ni embargo.
  `--last-folds 4` selecciona los últimos cuatro por símbolo. No se usa holdout.
- En v1 la calibración sirve también como validación para early stopping; sus AUC/Brier
  son diagnósticos de selección, no estimaciones independientes. Con
  `--separate-validation` (GRU/TCN v2, sección 7b) el early stopping usa el último
  20 % del ajuste, purgado contra esa validación, y la calibración queda solo para
  Platt, igual que en la CNN; el estandarizador se ajusta con el ajuste restante. La
  bandera entra en `experiment.json` solo cuando está activa, así que los manifiestos
  de v1 siguen coincidiendo al reanudar. Platt se ajusta
  después de restaurar el mejor modelo. Si solo hay una clase, la API existente
  deja la calibración como identidad y se registra en `calibrator.json`.
- Se predicen todas las velas de prueba con ventana válida, incluso sin etiqueta.
  En evaluación una predicción ausente implica efectivo. La evaluación usa desde
  la primera hasta la última predicción, con los mismos precios y símbolos para
  el modelo y todas las estrategias de `RULE_STRATEGIES`, incluido buy_and_hold.
- `dl_*` mantiene el comportamiento de la API existente: no activa las variantes
  de stop/riesgo especiales reservadas a `ml_*`. No cambia el motor de decisión.
- Smoke limita ajuste a 2048 finales, calibración a 512, un símbolo, un fold, dos
  épocas y ventana 16; conserva el contexto y la prueba completa del fold.

## 5. GPU, memoria y reproducibilidad

AMP usa bf16 si está disponible y, en otro caso, fp16 con GradScaler. CPU usa fp32.
La acumulación divide por el número real de muestras del grupo, también en el
último grupo incompleto. Se desescala antes del clipping. OneCycle avanza solo
cuando el optimizador ejecuta el paso. Referencia: [AMP de PyTorch](https://docs.pytorch.org/docs/2.3/notes/amp_examples.html).

Puede cambiar de GPU y reanudar: los pesos se cargan de forma portable y AMP se
detecta de nuevo. Si la nueva GPU no admite el batch, use una GPU con memoria
suficiente o inicie otro experimento con batch menor. `--accum` aumenta el batch
efectivo y `--compile` activa compilación opcional (apagada por defecto); compilación
y calentamiento pueden consumir tiempo y memoria adicionales.

Guarde el código/commit, los JSON de versiones y hashes generados por el notebook,
`experiment.json` y `run_info.json`. Los checkpoints guardan estados aleatorios;
la igualdad numérica exacta entre GPUs, kernels o versiones diferentes no se
garantiza. No hay transformaciones aleatorias en los workers de datos.

El benchmark mide matmul fp32/TF32/fp16/bf16 y entrenamiento de los tres modelos
para batches 256, 1024 y 4096. Sin memoria registra `oom`, sin soporte registra
`unsupported`. Sus datos sintéticos están residentes en el dispositivo, por lo
que muestras/s no mide lectura desde Drive. Pase filas reales aproximadas:

```bash
python scripts/gpu_benchmark.py --config configs/base.yaml \
  --rows-per-symbol BTCUSDT=<filas_reales> ETHUSDT=<filas_reales> \
  --n-features <numero_real>
```

Reemplace los marcadores antes de ejecutar. Los resultados se escriben en
`$QUANT_AI_HOME/reports/gpu_benchmark/<gpu>_<fecha>.json`. Si faltan filas no se
inventa el tiempo por época. `notebooks/gpu_benchmark.ipynb` solo llama al script
y muestra sus tablas. `notebooks/data_exploration.ipynb` lee calidad, retornos,
volatilidad, autocorrelación y correlación BTC–ETH sin entrenar.

## 6. Presupuesto y operación

Ingrese `UNITS_PER_HOUR` copiando el valor de su sesión. La celda de presupuesto
multiplica horas realmente transcurridas desde la celda 0 por esa tasa. Sin tasa,
solo muestra tiempo. El registro por etapa incluye intentos fallidos; una sesión
terminada abruptamente puede perder su último intervalo. La estimación no sustituye
el consumo que reporta Colab y no incluye sesiones anteriores no registradas.
Cambiar GPU puede cambiar la tasa: registre cada sesión por separado.

Desconecte el runtime cuando termine. Colab no es un servidor para un bot 24/7:
las sesiones se interrumpen y el almacenamiento temporal puede desaparecer. Este
flujo no despliega un bot ni habilita operaciones reales.

## 7. Verificación local permitida

No se necesita torch localmente para compilar los archivos ni validar notebooks:

```powershell
.venv\Scripts\python.exe -m py_compile src/quant_ai/models/sequence.py
.venv\Scripts\python.exe -c "import nbformat,sys; nbformat.validate(nbformat.read(sys.argv[1], as_version=4))" notebooks/colab_training.ipynb
.venv\Scripts\python.exe -m pytest tests/test_sequence.py -q
```

Repita compilación para cada Python nuevo y validación para los tres notebooks.
`pytest.importorskip("torch")` omite el módulo secuencial cuando falta torch;
pytest puede devolver código 5 al no quedar pruebas recolectadas en ese archivo.
`-ra` permite ver el motivo de omisión. En la nube las ejecuta la etapa `pruebas`
del cuaderno (ver «Pruebas en la nube»). No se afirma rendimiento GPU hasta medirlo allí.

## Aprendizaje por refuerzo

Para ejecutar solo RL en Colab, use las celdas 0–4 y la nueva sección final.
Esta instala `gymnasium stable-baselines3` solo si faltan y bloquea la ejecución
local. Cada modelo opera BTCUSDT o ETHUSDT spot 1h, sin cortos ni crédito.

```bash
python scripts/train_rl.py --config configs/base.yaml --algo ppo \
  --symbol BTCUSDT --seeds 0 1 2 --run ppo_btc_v1 --resume \
  --train-end 2024-01-01 --eval-start 2024-01-01 --eval-end 2025-11-01
```

Use `--algo dqn`, `--symbol ETHUSDT` u otro `--run` para experimentos distintos;
`--smoke` limita a 256 pasos por semilla y conserva la evaluación completa.
Los finales son **exclusivos**: el ejemplo incluye 2025-10-31. Se rechaza cualquier
solapamiento train/eval o invasión del holdout. Es un corte walk-forward configurable
de desarrollo; la evaluación repetida no sustituye la evaluación final intacta.

La normalización se ajusta solo en train y queda congelada. Train usa episodios
de 720 pasos con inicio aleatorio; evaluación recorre todo su rango. Los pesos
objetivo son `0/0.25/0.5` por defecto, con banda 0.05. La cantidad decidida al cierre
se ejecuta en la siguiente apertura con `CostModel.from_config`, volumen previo,
efectivo, lote y nocional mínimo. No hay stops ni liquidación ficticia al terminar.
El precio puede elevar el peso sobre el objetivo, pero nunca sobre 1.

Recompensa: log-retorno neto menos `dd_penalty × max(DD_nuevo − DD_anterior, 0)`
y `turnover_penalty × |cambio de peso ejecutado|`; defaults 0.1 y 0.001, configurables
por CLI (cero los desactiva). El peso se compara antes/después del fill a la misma
apertura: la deriva de mercado no es turnover. Los costes ya están en el capital.

Drive conserva `<models_dir>/rl/<run>/`: normalización y manifiesto, modelos,
checkpoints, curvas y `metrics.csv` por semilla, y `eval_summary.json` con métricas
financieras y estabilidad entre semillas. Comprar y mantener usa una entrada con
igual peso inicial y costes; conserva el fill recibido sin rebalancear. La primera
compra ocurre en la segunda vela, igual que RL. Repetir acción 2 puede rebalancear;
la prueba de mantener unidades utiliza banda 1.

`--resume` retoma el último checkpoint y el replay buffer DQN, o inicia desde cero
si aún no existe. Valida datos y parámetros contra el manifiesto. `--timesteps`
es el objetivo **total** (PPO completa rollouts); puede aumentarse. Reinicia el
episodio sin conservar el rollout parcial ni todos los RNG: no es una continuación
idéntica bit a bit.

MLP suele ir igual o mejor en CPU ([SB3/PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)).
Compare el mismo smoke con `--device cpu/cuda`, en carpetas distintas, usando los
tiempos de `run_step`; el benchmark secuencial no mide RL. Localmente no instale
ni entrene: `pytest.importorskip("gymnasium")` omite las pruebas RL.

## Ronda v2 (2026-11-09): todas las fuentes y técnicas

Antes de ejecutar, revisa el interruptor `ETAPAS` en la celda de parámetros. Por defecto omite lo que ya corrió en v1 (ML v1, benchmark, GRU/TCN v1 y RL), porque repetirlo daría exactamente lo mismo, y ejecuta solo lo nuevo:

| Etapa | Qué hace | Dónde corre | Tiempo estimado* |
|---|---|---|---|
| `pruebas` | Pruebas de secuencias, CNN, técnico, experimento v2 y backtest con torch (ver «Pruebas en la nube») | CPU | 1 a 3 min |
| `datos_externos` | Derivados de Binance, Coinbase, Deribit DVOL, Fear & Greed, stablecoins, FRED, alts y on-chain (`scripts/download_extended.py`) | Red | 10 a 15 min |
| `experimento_v2` | 8 variantes pre-registradas × 16 trimestres × 2 activos (256 ajustes de LightGBM y regresión logística) con 3 escenarios de costes (`scripts/run_experiments_v2.py`) | CPU | 45 a 75 min |
| `secuencias_v2` | GRU y TCN con base + técnico + externo, horizonte 24 h, mismo tramo 2022-2025, validación separada de la calibración (`--separate-validation`) | GPU | 5 a 10 min por modelo |
| `cnn` | CNN sobre imágenes de velas (ventana 48), 16 trimestres × 2 activos, con imágenes pre-renderizadas en RAM | CPU (pre-render) + GPU | sin cifra fiable (ver abajo) |
| consolidación | Re-evalúa juntos y con trato idéntico todos los candidatos v2 (ML, GRU/TCN v2 y CNN): DSR acumulado, bootstrap pareado y puerta (`scripts/consolidate_v2.py`). **Es el reporte que cuenta** | CPU | 1 a 3 min |

**Medido el 2026-11-10 en una A100** (sesión completa: 1 h 17 min): pruebas 0.7 min, datos externos 5.7 min, experimento v2 23.7 min, GRU 6.0 min, TCN 6.0 min, CNN 29.8 min y consolidación 0.9 min. La prueba SMOKE tardó 6 min. En otra GPU los tiempos cambian.

\* Las cifras de la tabla son las estimaciones previas a esa medición. **Sesión completa: 2 a 3 horas, sin contar la CNN.** Extrapola con los tiempos por fold que imprime cada celda. Mantén la pestaña abierta: el experimento v2 no se reanuda a mitad, y la caché `/content/qa_cache` se pierde si el entorno se desconecta. En v1, Colab tardó 165 s en el ML que este PC hizo en 111 s con 2 hilos, así que su CPU no es más rápida. El notebook registra el tiempo real de cada etapa en `reports/colab_sessions/`.

**CNN y caché de imágenes.** La cifra anterior de 30 a 60 min no se sostenía: cada imagen se renderizaba en cada época (3.1 ms por imagen en la medición de la revisión y entre 0.6 y 0.9 ms en este PC; unos 13 millones de renders en la corrida completa, es decir, horas de CPU con la GPU esperando). Como la imagen depende solo del activo, la ventana y la posición final, ahora `scripts/train_chart_cnn.py` la renderiza una sola vez por activo, con todos los núcleos, y la comparten todos los folds y épocas. La caché vive en la RAM de la VM, nunca en Drive: menos de 0.8 GB por activo con ventana 48, y se libera antes de pasar al siguiente activo. Las imágenes son idénticas byte a byte a las de antes. El script imprime cuántas imágenes pre-renderizó, cuánto tardó y cuánta memoria ocupan. En este PC fueron 0.86 ms por imagen en un solo proceso; en la nube no se ha medido. Como estimación, con 4 vCPU deberían bastar uno o dos minutos por activo. El entrenamiento en GPU de los 32 folds sigue sin medir. `--no-cache` vuelve al render por época (mismo resultado, mucho más lento) y no cambia `experiment.json`.

**Exportación incremental y resumen final.** Tras cada etapa v2 (datos externos, experimento v2, secuencias v2, CNN y consolidación), el notebook vuelve a escribir `quant-ai-resultados_<sesión>.zip`. En Colab queda en `Mi unidad/quant-ai/`; en Kaggle, en `/kaggle/working`. El ZIP se arma en el disco local de la VM y se copia de una vez con reemplazo atómico, así que nunca queda uno a medias. Lleva lo mismo que la exportación final: reportes, métricas, predicciones y registro, sin checkpoints `.pt`/`.ckpt`. Con los resultados de v1 (173 archivos, 35 MB), cada exportación tomó entre 1.7 y 3.3 s en este PC; la escritura en Drive no se ha medido. Si una exportación intermedia falla, la corrida sigue y el fallo queda anotado. En Colab, `files.download` solo se llama al final. Antes de esa última exportación, la celda «Resumen de la sesión» lista cada etapa con su estado y duración. Termina con «TODAS LAS ETAPAS TERMINARON BIEN» o con las etapas que fallaron y sus últimas líneas de error, y lo guarda en `reports/colab_sessions/resumen_<sesión>.json`, dentro del ZIP.

**Errores legibles.** Si un script falla, la celda repite sus últimas 40 líneas de salida bajo «ÚLTIMAS LÍNEAS DEL ERROR en «etapa» (cópialas en el chat)». Esas líneas también quedan en `reports/colab_sessions/<sesión>.json`. Basta con copiar ese bloque o el final de la celda.

Notas:
- Primero ejecuta con `SMOKE = True` (minutos) para validar todo de punta a punta. Después pon `SMOKE = False`. En SMOKE, el notebook borra antes las carpetas desechables `secuencias_v2_smoke/<modelo>` y `chart_cnn_w48_smoke` para que un manifiesto viejo no bloquee `--resume`.
- Si te faltan unidades, apaga `cnn`; probablemente sigue siendo la etapa más larga.
- Si una fuente externa está bloqueada en la región de la VM (por ejemplo Deribit desde EE. UU.), la descarga avisa y continúa. Sus columnas quedan vacías y el experimento lo reporta en la cobertura.
- Ningún modelo v2 se registra como activo. Las variantes con características `ta_` y `ext_` todavía no tienen ruta de cálculo en vivo en `execution/runner.py`, así que un ganador v2 no se puede operar en paper hasta implementarla.
- El holdout final (desde 2025-11-01) sigue sellado.

## Pruebas en la nube

La etapa `pruebas` (sección 3b, activa por defecto) corre después de la instalación y
antes de entrenar:

```bash
python -m pytest tests/test_sequence.py tests/test_chart_cnn.py tests/test_technical.py \
  tests/test_experiment_v2.py tests/test_backtest.py -p no:cacheprovider -rs
```

`-q` ya viene en `addopts` de `pyproject.toml`; repetirlo ocultaría la línea de totales.
`tests/test_rl_env.py` solo se agrega si la etapa `rl` está activa, y se omite sola si
gymnasium aún no está instalado. Si falta pytest, se instala solo en la nube. La salida
completa queda en `reports/colab_sessions/pytest_<sesión>.txt`, dentro del ZIP. Un fallo
no detiene la corrida: queda en el resumen final. En la nube ninguna prueba de torch
debería aparecer como `SKIPPED`.

## Kaggle (alternativa cuando Colab no asigna GPU)

El camino principal es Colab. El mismo notebook detecta Kaggle; en Colab nada cambia. La plataforma se detecta por las variables de entorno (`KAGGLE_KERNEL_RUN_TYPE` en Kaggle, `COLAB_*` en Colab) y no por carpetas: Colab también trae `/kaggle/input`, y con la detección anterior se confundía con Kaggle. La primera celda imprime el entorno detectado; si se equivoca, fija `PLATAFORMA = "colab"` o `"kaggle"` en la celda de parámetros.

1. En el PC: `python scripts/package_for_colab.py --kaggle`. Genera `dist/quant-ai-kaggle.zip`, con el mismo contenido que el ZIP de Colab: el código y una carpeta `estado_previo/`, que trae:
   - Los datos limpios de BTC y ETH, los mismos del PC.
   - El historial de intentos (22 al 2026-11-09). Sin él, el Deflated Sharpe contaría solo los intentos de la corrida nueva y sería optimista.

   El paquete no lleva secretos, estado de paper, checkpoints ni predicciones.
2. En Kaggle, sube ese ZIP como dataset privado (*Create → New Dataset*). Kaggle puede descomprimirlo, y el notebook acepta tanto el ZIP como la carpeta.
3. Importa `notebooks/colab_training.ipynb` (*Create → New Notebook → File → Import Notebook*) y agrega el dataset con *Add Input*.
4. En *Settings* del notebook:
   - *Accelerator*: **GPU T4 x2**. No elijas P100: el PyTorch actual de Kaggle no la soporta ([Kaggle/docker-python#1546](https://github.com/Kaggle/docker-python/issues/1546)). El código usa una sola GPU y, en la T4, fp16 con GradScaler en lugar de bf16.
   - *Internet*: activado. Lo necesitan los datos externos y la instalación. Si el interruptor aparece bloqueado, sigue lo que indique Kaggle. Sin Internet, el notebook se detiene en la celda 3 con un aviso y no gasta GPU.
5. Corre una prueba mínima en la sesión interactiva con `SMOKE = True` (*Run All*). Revisa que:
   - La celda 3 diga `Internet: disponible` y `estado_previo: 14 archivos copiados`.
   - La etapa `pruebas` termine con una línea de totales sin fallos y sin pruebas de torch omitidas.
   - La etapa `datos_externos` termine con fuentes descargadas y no con un fallo general.
   - El experimento v2 imprima `Intentos acumulados para el DSR`. Debe ser un número cercano o mayor a 22, no 8 a 12.
   - La CNN imprima `imágenes pre-renderizadas`, y tras cada etapa v2 aparezca `ZIP intermedio actualizado`.
   - La celda «Resumen de la sesión» diga `TODAS LAS ETAPAS TERMINARON BIEN`.
6. Corrida completa:
   - Pon `SMOKE = False`, guarda y usa **Save Version → Save & Run All (Commit)**.
   - Corre en segundo plano, sin la pestaña abierta.
   - Cuando termine, descarga `quant-ai-resultados_*.zip` desde la pestaña *Output* de esa versión. Al final del log está el resumen de la sesión; el mismo resumen va en el ZIP (`reports/colab_sessions/resumen_<sesión>.json`).

Kaggle muestra la cuota semanal de GPU y el límite por sesión en su panel. Este proyecto no supone cifras. Los tiempos de la tabla anterior se estimaron para una L4 y no se han medido en una T4. En Kaggle los datos y modelos se guardan en `/tmp/qa_home` y se pierden al cerrar la sesión. A `/kaggle/working` (la salida) va solo el ZIP de resultados, que se reemplaza tras cada etapa v2. Si la sesión se corta, ahí queda el último ZIP intermedio. No se ha comprobado qué conserva Kaggle de una versión que se corta por tiempo. No hay reanudación entre sesiones.

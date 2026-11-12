# Especificación: CNN sobre imágenes de velas ("análisis de gráficos" con IA)

Estado: especificación para implementar (2026-11-09). Inspirada en Jiang, Kelly y Xiu (2023), "(Re-)Imag(in)ing Price Trends", *Journal of Finance*: una CNN aprende patrones directamente de imágenes OHLC, sin indicadores predefinidos.

## Hipótesis falsable

Una CNN entrenada con imágenes de las últimas N velas predice la etiqueta de 24 h mejor que el ensemble LR+LightGBM. La comparación se hace en el mismo tramo walk-forward, con AUC y con backtest con costes. Si no la supera, se descarta: es un resultado válido.

## Restricciones

- No instalar paquetes en el PC. torch NO está instalado localmente, así que las pruebas usan `pytest.importorskip("torch")`. El entrenamiento corre en Colab (GPU L4).
- Crear solo los archivos indicados. Se pueden importar, pero no modificar: `features/`, `labeling/`, `validation/`, `backtesting/`, `training/dataset.py`, `training/experiment*.py`, `training/sequence_data.py`, `training/trainer.py` y `configs/`.
- Comentarios y textos en español (Colombia).

## 1. `src/quant_ai/training/chart_images.py`

- `render_window(o, h, l, c, v, height=64, vol_height=16) -> np.ndarray uint8 (1, height+vol_height, 3*N)`.
  - Cada vela ocupa 3 píxeles de ancho: apertura a la izquierda, barra máx-mín en el centro y cierre a la derecha (como en el paper).
  - Escala de precios min-max de la propia ventana.
  - Superpone la media móvil de N velas, calculada solo con datos de la ventana.
  - Barras de volumen abajo, escaladas al máximo de la ventana.
  - Sin información posterior a la última vela de la ventana.
- `ChartDataset(torch Dataset)`: renderiza bajo demanda sin materializar todas las imágenes.
  - La ventana de la muestra t son las velas t−N+1 … t (incluida t).
  - Excluye ventanas con huecos temporales.
  - Etiqueta: la misma que el experimento v2, horizonte 24 h, entrada en `open[t+1]`, salida en `open[t+25]`, y = 1 si el retorno supera 26 bps. Usar `quant_ai.labeling.labels.make_labels` con `horizon_bars=24`.
- Particiones: `quant_ai.validation.splits.purged_walk_forward` con la configuración de `configs/experiments_v2.yaml` (`test_start: 2022-01-01`, trimestres) y la opción `last_folds`. Último 20 % del ajuste para early stopping y otra porción separada para la calibración Platt (`quant_ai.models.ml.PlattCalibrator`): NO usar el mismo tramo para ambas cosas.
- Nunca usar datos desde `cfg.final_holdout_start`.

## 2. `src/quant_ai/models/chart_cnn.py`

`ChartCNN`: 3 bloques conv (kernel 5×3, BatchNorm, LeakyReLU y MaxPool 2×1, con canales 32, 64 y 128), dropout 0.5 y una capa lineal que produce el logit. Tamaño objetivo menor a 1 M de parámetros. Admite N = 48 o N = 96.

## 3. `scripts/train_chart_cnn.py`

- Reutiliza `quant_ai.training.trainer.SequenceTrainer` si encaja (AMP, checkpoints, reanudación y metrics.csv). Si no encaja, implementa un bucle equivalente con las mismas columnas de `metrics.csv` y el mismo `run_info.json`.
- Argumentos: `--window 48|96`, `--epochs 20`, `--batch-size 256`, `--lr 1e-3`, `--last-folds` (por defecto 16 = todo el tramo 2022-2025), `--symbols`, `--out` (por defecto `<models_dir>/dl/chart_cnn_w<N>`), `--resume`, `--num-workers 4` y `--smoke` (1 símbolo, 1 fold, 1 época, 512 muestras).
- Guarda `predictions.parquet` en el mismo formato que `scripts/train_sequence.py` (index open_time; columnas symbol, fold, p_seq, p_seq_raw, y, fwd_ret).
- Al final llama a `scripts/evaluate_sequence.py` (función `evaluate`) con el nombre `dl_chart_cnn`, que compara contra los baselines y contra `ml_ensemble` en el mismo tramo.
- Guarda 16 imágenes de ejemplo (PNG o `.npy`) en `<out>/samples/` para inspección visual; si no hay librería de imágenes, guarda `.npy`.

## 4. `tests/test_chart_cnn.py`

Con `pytest.importorskip("torch")`, salvo la prueba de renderizado, que solo usa numpy:
- Alterar velas posteriores a t no cambia la imagen de t.
- La forma de la imagen es la esperada.
- La vela de mayor máximo de la ventana toca la fila superior.
- El modelo produce un logit por muestra.
- Un entrenamiento smoke con imágenes sintéticas separables reduce la pérdida.

## 5. Notebook

Agregar al final de `notebooks/colab_training.ipynb`, antes de la celda de exportación de resultados, una sección "Análisis de gráficos (CNN)". Debe ejecutar `train_chart_cnn.py` con `--resume` y salida en Drive, y mostrar la tabla de `backtest_summary.json`. Mantener el notebook válido con nbformat y con `id` en cada celda nueva.

## Criterio de terminado

`py_compile` de todo; `pytest -q` pasando, aunque las pruebas que requieren torch se omitan localmente; notebook válido.

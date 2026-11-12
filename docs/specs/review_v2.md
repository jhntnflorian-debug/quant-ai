# Encargo: revisión adversarial del diseño v2 (solo lectura)

Objetivo: encontrar errores que invaliden los resultados v2 ANTES de entrenar en Colab. La prioridad son los fallos que hagan parecer rentable algo que no lo es.

## Restricciones

- No modificar archivos ni instalar nada.
- Nada de búsquedas web largas. Presupuesto: unos 15 minutos.
- Responder en español (Colombia).

## Archivos a revisar

- `src/quant_ai/features/technical.py`: fuga temporal en pivotes y soportes/resistencias, Supertrend, Ichimoku y marcos de 4 h y diario.
- `src/quant_ai/training/experiment_v2.py`: meta-etiquetado, pesos, escenarios de costes, DSR y puerta.
- `src/quant_ai/decision_engine/policy.py` (`vol_target_weights` y `weights_from_probs`).
- `src/quant_ai/validation/trials.py` y `src/quant_ai/backtesting/metrics.py` (`probabilistic_sharpe` y `deflated_sharpe`).
- `src/quant_ai/training/experiment.py` (`walk_forward_predictions` y `run_strategy_backtests` con `stop_names`).
- `configs/experiments_v2.yaml`.
- `scripts/train_sequence.py` (opciones `--feature-groups`, `--horizon` y `--test-start`).
- Las celdas con `ETAPAS` de `notebooks/colab_training.ipynb`.

## Qué buscar

1. Cualquier uso de información posterior al cierre de la vela t en características, etiquetas, pesos o backtest.
2. Errores de comparación: estrategias evaluadas en tramos o con costes o stops distintos.
3. Errores estadísticos: fórmula del PSR/DSR (Bailey y López de Prado), conteo de intentos, anualización y supuestos.
4. Errores que harían fallar la corrida en Colab (pandas 2.2.3, scikit-learn 1.6.1, LightGBM 4.6.0, Python 3.13 y torch 2.14).
5. Las 3 mejoras de mayor valor que falten en las variantes pre-registradas, con su justificación. Deben ser pocas: cada variante extra sube la vara del DSR.

## Formato de salida

Lista ordenada por gravedad. Cada hallazgo debe indicar archivo y línea, el escenario concreto en que falla y la corrección propuesta. Si algo está bien, decirlo en una línea. Máximo unas 900 palabras.

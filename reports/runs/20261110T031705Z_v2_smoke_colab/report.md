# Experimento v2 `20261110T031705Z_v2_smoke_colab` (v2)

- Tramo de prueba (walk-forward): 2022-01-01 00:00:00+00:00 → 2022-03-31 23:00:00+00:00 — **PRUEBA MÍNIMA, no válida**
- Holdout final (2025-11-01 00:00:00+00:00): sin tocar
- Intentos contados en el DSR: 27 acumulados sobre el periodo de desarrollo
- Stop-loss ATR y regla de señal nueva tras un stop: iguales para ML, DL y reglas de tendencia; no aplican a comprar y mantener

| Estrategia | Retorno | Sharpe | Máx. DD | Operaciones | PSR | DSR | ΔSharpe vs buy_and_hold (IC95 %) | ΔSharpe vs ema_cross_vol_target (IC95 %) | Estrés ×2 | BNB 7.5 bps | Puerta |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ml_v0_base_h12 | -3.6% | -2.84 | -4.0% | 3 | 0.009 | 0.000 | -2.83 (-8.02, 1.31) | -0.03 (-4.07, 4.20) | -4.0% | -3.5% | no |
| ml_v1_base_h24 | +0.0% | 0.00 | +0.0% | 0 | — | — | 0.02 (-3.98, 4.05) | 2.82 (-0.62, 6.42) | +0.0% | +0.0% | no |
| ml_v2_ta_h24 | +1.4% | 1.22 | -2.3% | 1 | 0.728 | 0.202 | 1.23 (-3.37, 5.23) | 4.03 (-0.26, 8.19) | +1.2% | +1.4% | no |
| ml_v2b_ext_h24 | +0.0% | 0.00 | +0.0% | 0 | — | — | 0.02 (-3.98, 4.05) | 2.82 (-0.62, 6.42) | +0.0% | +0.0% | no |
| ml_v3_ta_ext_h24 | +0.0% | 0.00 | +0.0% | 0 | — | — | 0.02 (-3.98, 4.05) | 2.82 (-0.62, 6.42) | +0.0% | +0.0% | no |
| ml_v4_meta_ema_h24 | +0.0% | 0.00 | +0.0% | 0 | — | — | 0.02 (-3.98, 4.05) | 2.82 (-0.62, 6.42) | +0.0% | +0.0% | no |
| ml_v5_meta_ema_h24_vt | +0.0% | 0.00 | +0.0% | 0 | — | — | 0.02 (-3.98, 4.05) | 2.82 (-0.62, 6.42) | +0.0% | +0.0% | no |
| ml_v6_meta_ema_tb_vt | +0.0% | 0.00 | +0.0% | 0 | — | — | 0.02 (-3.98, 4.05) | 2.82 (-0.62, 6.42) | +0.0% | +0.0% | no |
| buy_and_hold | -1.3% | -0.02 | -15.5% | 0 | 0.497 | 0.074 | — | — | -1.4% | -1.3% | ref. |
| ema_cross_24_168 | -10.9% | -2.72 | -12.7% | 10 | 0.086 | 0.002 | — | — | -12.0% | -10.7% | ref. |
| ema_cross_vol_target | -8.4% | -2.82 | -9.9% | 10 | 0.078 | 0.002 | — | — | -9.3% | -8.2% | ref. |

## Calidad predictiva fuera de muestra (variantes ML)

| Variante:símbolo | n | Tasa base | AUC | Brier | Brier base | Precisión@entrada | Cobertura |
|---|---|---|---|---|---|---|---|
| v0_base_h12:BTCUSDT | 2160 | 0.414 | 0.559 | 0.2418 | 0.2427 | 0.333 | +0.4% |
| v1_base_h24:BTCUSDT | 2160 | 0.455 | 0.519 | 0.2488 | 0.2480 | — | +0.0% |
| v2_ta_h24:BTCUSDT | 2160 | 0.455 | 0.491 | 0.2488 | 0.2480 | 1.000 | +0.0% |
| v2b_ext_h24:BTCUSDT | 2160 | 0.455 | 0.518 | 0.2488 | 0.2480 | — | +0.0% |
| v3_ta_ext_h24:BTCUSDT | 2160 | 0.455 | 0.491 | 0.2488 | 0.2480 | — | +0.0% |
| v4_meta_ema_h24:BTCUSDT | 989 | 0.405 | 0.480 | 0.2430 | 0.2411 | — | +0.0% |
| v5_meta_ema_h24_vt:BTCUSDT | 989 | 0.405 | 0.480 | 0.2430 | 0.2411 | — | +0.0% |
| v6_meta_ema_tb_vt:BTCUSDT | 989 | 0.408 | 0.499 | 0.2429 | 0.2416 | — | +0.0% |

## Motivos de rechazo de la puerta

- **ml_v0_base_h12**: evaluación incompleta (prueba mínima o folds limitados); DSR 0.0 < 0.95; Sharpe -2.84 <= buy_and_hold -0.02; IC95 % de la ventaja de Sharpe frente a buy_and_hold incluye 0 (inferior -8.02); Sharpe -2.84 <= ema_cross_vol_target -2.82; IC95 % de la ventaja de Sharpe frente a ema_cross_vol_target incluye 0 (inferior -4.07); retorno con estrés de costes <= 0
- **ml_v1_base_h24**: evaluación incompleta (prueba mínima o folds limitados); DSR no disponible < 0.95; IC95 % de la ventaja de Sharpe frente a buy_and_hold incluye 0 (inferior -3.98); IC95 % de la ventaja de Sharpe frente a ema_cross_vol_target incluye 0 (inferior -0.62); retorno con estrés de costes <= 0
- **ml_v2_ta_h24**: evaluación incompleta (prueba mínima o folds limitados); DSR 0.202 < 0.95; IC95 % de la ventaja de Sharpe frente a buy_and_hold incluye 0 (inferior -3.37); IC95 % de la ventaja de Sharpe frente a ema_cross_vol_target incluye 0 (inferior -0.26)
- **ml_v2b_ext_h24**: evaluación incompleta (prueba mínima o folds limitados); DSR no disponible < 0.95; IC95 % de la ventaja de Sharpe frente a buy_and_hold incluye 0 (inferior -3.98); IC95 % de la ventaja de Sharpe frente a ema_cross_vol_target incluye 0 (inferior -0.62); retorno con estrés de costes <= 0
- **ml_v3_ta_ext_h24**: evaluación incompleta (prueba mínima o folds limitados); DSR no disponible < 0.95; IC95 % de la ventaja de Sharpe frente a buy_and_hold incluye 0 (inferior -3.98); IC95 % de la ventaja de Sharpe frente a ema_cross_vol_target incluye 0 (inferior -0.62); retorno con estrés de costes <= 0
- **ml_v4_meta_ema_h24**: evaluación incompleta (prueba mínima o folds limitados); DSR no disponible < 0.95; IC95 % de la ventaja de Sharpe frente a buy_and_hold incluye 0 (inferior -3.98); IC95 % de la ventaja de Sharpe frente a ema_cross_vol_target incluye 0 (inferior -0.62); retorno con estrés de costes <= 0
- **ml_v5_meta_ema_h24_vt**: evaluación incompleta (prueba mínima o folds limitados); DSR no disponible < 0.95; IC95 % de la ventaja de Sharpe frente a buy_and_hold incluye 0 (inferior -3.98); IC95 % de la ventaja de Sharpe frente a ema_cross_vol_target incluye 0 (inferior -0.62); retorno con estrés de costes <= 0
- **ml_v6_meta_ema_tb_vt**: evaluación incompleta (prueba mínima o folds limitados); DSR no disponible < 0.95; IC95 % de la ventaja de Sharpe frente a buy_and_hold incluye 0 (inferior -3.98); IC95 % de la ventaja de Sharpe frente a ema_cross_vol_target incluye 0 (inferior -0.62); retorno con estrés de costes <= 0

## Lectura

- Ablación: v1 → v2 (técnico) / v2b (externo) → v3 (ambos), mismo horizonte y tamaño.
- Meta-etiquetado: v4, v5 y v6 frente a ema_cross_24_168 y ema_cross_vol_target (misma señal, sin ML).
- DSR < 0.95 o IC95 % que incluye 0: no hay evidencia de ventaja frente a la suerte de haber probado muchas ideas.
- Resultados simulados; no son beneficios reales.

# Experimento v2 `20261110T120156Z_v2final` (v2_final)

- Tramo de prueba (walk-forward): 2022-01-01 00:00:00+00:00 → 2025-10-31 23:00:00+00:00
- Holdout final (2025-11-01 00:00:00+00:00): sin tocar
- Intentos contados en el DSR: 30 acumulados sobre el periodo de desarrollo
- Stop-loss ATR y regla de señal nueva tras un stop: iguales para ML, DL y reglas de tendencia; no aplican a comprar y mantener

| Estrategia | Retorno | Sharpe | Máx. DD | Operaciones | PSR | DSR | ΔSharpe vs buy_and_hold (IC95 %) | ΔSharpe vs ema_cross_vol_target (IC95 %) | Estrés ×2 | BNB 7.5 bps | Puerta |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ml_v0_base_h12 | -5.7% | -0.12 | -21.5% | 57 | 0.410 | 0.000 | -0.64 (-1.85, 0.65) | -0.11 (-1.56, 1.49) | -12.5% | -4.4% | no |
| ml_v1_base_h24 | +18.3% | 0.65 | -9.8% | 25 | 0.903 | 0.008 | 0.13 (-1.31, 1.76) | 0.66 (-0.79, 2.27) | +14.5% | +19.0% | no |
| ml_v2_ta_h24 | +22.3% | 0.50 | -13.4% | 39 | 0.838 | 0.003 | -0.03 (-1.36, 1.16) | 0.50 (-0.93, 1.91) | +16.2% | +23.5% | no |
| ml_v2b_ext_h24 | +5.5% | 0.19 | -18.3% | 30 | 0.648 | 0.001 | -0.33 (-1.63, 1.01) | 0.20 (-1.21, 1.62) | +1.4% | +6.3% | no |
| ml_v3_ta_ext_h24 | +28.9% | 0.69 | -15.5% | 35 | 0.912 | 0.010 | 0.17 (-1.26, 1.45) | 0.70 (-0.90, 2.21) | +23.1% | +30.0% | no |
| ml_v4_meta_ema_h24 | +1.1% | 0.13 | -3.3% | 9 | 0.601 | 0.000 | -0.39 (-1.65, 0.84) | 0.14 (-1.29, 1.40) | -0.1% | +1.3% | no |
| ml_v5_meta_ema_h24_vt | +1.3% | 0.21 | -2.0% | 9 | 0.657 | 0.001 | -0.32 (-1.59, 0.94) | 0.22 (-1.19, 1.50) | +0.3% | +1.4% | no |
| ml_v6_meta_ema_tb_vt | -2.1% | -0.11 | -8.6% | 21 | 0.411 | 0.000 | -0.64 (-2.11, 0.71) | -0.11 (-1.78, 1.34) | -4.2% | -1.7% | no |
| dl_v2_gru | -3.2% | -0.11 | -9.4% | 31 | 0.416 | 0.000 | -0.63 (-2.12, 0.73) | -0.10 (-1.55, 1.48) | -7.0% | -2.4% | no |
| dl_v2_tcn | +4.7% | 0.20 | -10.4% | 84 | 0.649 | 0.001 | -0.33 (-1.62, 0.97) | 0.20 (-1.15, 1.61) | -6.1% | +6.9% | no |
| dl_chart_cnn_w48 | -10.1% | -1.20 | -12.9% | 20 | 0.006 | 0.000 | -1.73 (-3.19, -0.15) | -1.20 (-2.68, 0.46) | -12.4% | -9.6% | no |
| buy_and_hold | +69.0% | 0.52 | -69.6% | 0 | 0.848 | 0.004 | — | — | +68.9% | +69.0% | ref. |
| ema_cross_24_168 | -21.3% | -0.05 | -42.3% | 318 | 0.458 | 0.000 | — | — | -47.8% | -15.0% | ref. |
| ema_cross_vol_target | -13.0% | -0.01 | -37.1% | 318 | 0.494 | 0.000 | — | — | -41.3% | -6.1% | ref. |

## Calidad predictiva fuera de muestra (variantes ML)

| Variante:símbolo | n | Tasa base | AUC | Brier | Brier base | Precisión@entrada | Cobertura |
|---|---|---|---|---|---|---|---|
| v0_base_h12:BTCUSDT | 33573 | 0.404 | 0.538 | 0.2398 | 0.2409 | 0.620 | +0.2% |
| v0_base_h12:ETHUSDT | 33573 | 0.431 | 0.532 | 0.2448 | 0.2453 | 0.452 | +0.5% |
| v1_base_h24:BTCUSDT | 33549 | 0.441 | 0.520 | 0.2463 | 0.2466 | 0.500 | +0.4% |
| v1_base_h24:ETHUSDT | 33549 | 0.456 | 0.490 | 0.2490 | 0.2481 | 0.636 | +0.2% |
| v2_ta_h24:BTCUSDT | 33549 | 0.441 | 0.524 | 0.2464 | 0.2466 | 0.584 | +0.3% |
| v2_ta_h24:ETHUSDT | 33549 | 0.456 | 0.501 | 0.2488 | 0.2481 | 0.586 | +0.4% |
| v2b_ext_h24:BTCUSDT | 33549 | 0.441 | 0.517 | 0.2469 | 0.2466 | 0.547 | +1.6% |
| v2b_ext_h24:ETHUSDT | 33549 | 0.456 | 0.506 | 0.2485 | 0.2481 | 0.517 | +0.8% |
| v3_ta_ext_h24:BTCUSDT | 33549 | 0.441 | 0.527 | 0.2463 | 0.2466 | 0.537 | +0.6% |
| v3_ta_ext_h24:ETHUSDT | 33549 | 0.456 | 0.499 | 0.2486 | 0.2481 | 0.677 | +0.8% |
| v4_meta_ema_h24:BTCUSDT | 17366 | 0.429 | 0.503 | 0.2468 | 0.2449 | 0.750 | +0.1% |
| v4_meta_ema_h24:ETHUSDT | 16614 | 0.431 | 0.494 | 0.2475 | 0.2453 | 0.545 | +0.1% |
| v5_meta_ema_h24_vt:BTCUSDT | 17366 | 0.429 | 0.503 | 0.2468 | 0.2449 | 0.750 | +0.1% |
| v5_meta_ema_h24_vt:ETHUSDT | 16614 | 0.431 | 0.494 | 0.2475 | 0.2453 | 0.545 | +0.1% |
| v6_meta_ema_tb_vt:BTCUSDT | 17391 | 0.455 | 0.510 | 0.2489 | 0.2479 | 0.712 | +0.3% |
| v6_meta_ema_tb_vt:ETHUSDT | 16639 | 0.449 | 0.508 | 0.2481 | 0.2474 | 0.463 | +2.9% |

## Motivos de rechazo de la puerta

- **ml_v0_base_h12**: DSR 0.0 < 0.95; Sharpe -0.12 <= buy_and_hold 0.52; IC95 % de la ventaja de Sharpe frente a buy_and_hold incluye 0 (inferior -1.85); Sharpe -0.12 <= ema_cross_vol_target -0.01; IC95 % de la ventaja de Sharpe frente a ema_cross_vol_target incluye 0 (inferior -1.56); retorno con estrés de costes <= 0
- **ml_v1_base_h24**: DSR 0.008 < 0.95; IC95 % de la ventaja de Sharpe frente a buy_and_hold incluye 0 (inferior -1.31); IC95 % de la ventaja de Sharpe frente a ema_cross_vol_target incluye 0 (inferior -0.79)
- **ml_v2_ta_h24**: DSR 0.003 < 0.95; Sharpe 0.50 <= buy_and_hold 0.52; IC95 % de la ventaja de Sharpe frente a buy_and_hold incluye 0 (inferior -1.36); IC95 % de la ventaja de Sharpe frente a ema_cross_vol_target incluye 0 (inferior -0.93)
- **ml_v2b_ext_h24**: DSR 0.001 < 0.95; Sharpe 0.19 <= buy_and_hold 0.52; IC95 % de la ventaja de Sharpe frente a buy_and_hold incluye 0 (inferior -1.63); IC95 % de la ventaja de Sharpe frente a ema_cross_vol_target incluye 0 (inferior -1.21)
- **ml_v3_ta_ext_h24**: DSR 0.01 < 0.95; IC95 % de la ventaja de Sharpe frente a buy_and_hold incluye 0 (inferior -1.26); IC95 % de la ventaja de Sharpe frente a ema_cross_vol_target incluye 0 (inferior -0.9)
- **ml_v4_meta_ema_h24**: DSR 0.0 < 0.95; Sharpe 0.13 <= buy_and_hold 0.52; IC95 % de la ventaja de Sharpe frente a buy_and_hold incluye 0 (inferior -1.65); IC95 % de la ventaja de Sharpe frente a ema_cross_vol_target incluye 0 (inferior -1.29); retorno con estrés de costes <= 0
- **ml_v5_meta_ema_h24_vt**: DSR 0.001 < 0.95; Sharpe 0.21 <= buy_and_hold 0.52; IC95 % de la ventaja de Sharpe frente a buy_and_hold incluye 0 (inferior -1.59); IC95 % de la ventaja de Sharpe frente a ema_cross_vol_target incluye 0 (inferior -1.19)
- **ml_v6_meta_ema_tb_vt**: DSR 0.0 < 0.95; Sharpe -0.11 <= buy_and_hold 0.52; IC95 % de la ventaja de Sharpe frente a buy_and_hold incluye 0 (inferior -2.11); Sharpe -0.11 <= ema_cross_vol_target -0.01; IC95 % de la ventaja de Sharpe frente a ema_cross_vol_target incluye 0 (inferior -1.78); retorno con estrés de costes <= 0
- **dl_v2_gru**: DSR 0.0 < 0.95; Sharpe -0.11 <= buy_and_hold 0.52; IC95 % de la ventaja de Sharpe frente a buy_and_hold incluye 0 (inferior -2.12); Sharpe -0.11 <= ema_cross_vol_target -0.01; IC95 % de la ventaja de Sharpe frente a ema_cross_vol_target incluye 0 (inferior -1.55); retorno con estrés de costes <= 0
- **dl_v2_tcn**: DSR 0.001 < 0.95; Sharpe 0.20 <= buy_and_hold 0.52; IC95 % de la ventaja de Sharpe frente a buy_and_hold incluye 0 (inferior -1.62); IC95 % de la ventaja de Sharpe frente a ema_cross_vol_target incluye 0 (inferior -1.15); retorno con estrés de costes <= 0
- **dl_chart_cnn_w48**: DSR 0.0 < 0.95; Sharpe -1.20 <= buy_and_hold 0.52; IC95 % de la ventaja de Sharpe frente a buy_and_hold incluye 0 (inferior -3.19); Sharpe -1.20 <= ema_cross_vol_target -0.01; IC95 % de la ventaja de Sharpe frente a ema_cross_vol_target incluye 0 (inferior -2.68); retorno con estrés de costes <= 0

## Lectura

- Ablación: v1 → v2 (técnico) / v2b (externo) → v3 (ambos), mismo horizonte y tamaño.
- Meta-etiquetado: v4, v5 y v6 frente a ema_cross_24_168 y ema_cross_vol_target (misma señal, sin ML).
- DSR < 0.95 o IC95 % que incluye 0: no hay evidencia de ventaja frente a la suerte de haber probado muchas ideas.
- Resultados simulados; no son beneficios reales.

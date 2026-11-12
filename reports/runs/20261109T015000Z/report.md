# Reporte de experimento `20261109T015000Z`

- Tipo: **full**
- Periodo evaluado (fuera de muestra): 2020-01-01 00:00:00+00:00 → 2025-10-31 23:00:00+00:00
- Holdout final (2025-11-01 00:00:00+00:00 en adelante): sin tocar
- Costes: comisión 10.0 bps + medio spread 1.0 bps + slippage 2.0 bps por lado

## Estrategias (backtest con costes, señal al cierre → ejecución en la siguiente apertura)

| Estrategia | Retorno | CAGR | Sharpe (IC95 %) | Sortino | Máx. DD | Operaciones | Win rate | Profit factor | Exposición | Costes/capital |
|---|---|---|---|---|---|---|---|---|---|---|
| buy_and_hold | +2152.8% | +70.5% | 1.10 (0.33, 1.93) | 1.52 | -79.9% | 0 | — | — | +100.0% | +0.1% |
| price_above_ema_fast | -95.7% | -41.7% | -1.30 (-2.18, -0.54) | -1.86 | -96.9% | 4404 | +17.5% | 0.88 | +39.8% | +288.8% |
| ema_cross_24_168 | +650.2% | +41.3% | 1.01 (0.09, 1.82) | 1.40 | -55.6% | 488 | +25.6% | 1.25 | +54.4% | +306.1% |
| tsmom_168 | +136.1% | +15.9% | 0.56 (-0.43, 1.41) | 0.78 | -69.0% | 1725 | +22.7% | 1.06 | +54.8% | +670.0% |
| rsi_reversion | -62.5% | -15.5% | -0.24 (-0.86, 0.39) | -0.32 | -70.3% | 628 | +62.4% | 0.81 | +21.0% | +50.3% |
| ml_logistic | +176.2% | +19.0% | 0.76 (-0.04, 1.68) | 1.03 | -43.6% | 510 | +53.1% | 1.27 | +10.4% | +107.5% |
| ml_lightgbm | +193.8% | +20.3% | 0.85 (0.05, 1.65) | 1.17 | -45.1% | 716 | +58.2% | 1.16 | +11.3% | +231.9% |
| ml_ensemble | +146.4% | +16.7% | 0.72 (-0.09, 1.43) | 0.97 | -49.1% | 465 | +57.6% | 1.20 | +9.8% | +115.1% |
| ml_ensemble_risk | +154.4% | +17.4% | 0.88 (0.01, 1.70) | 1.20 | -35.1% | 316 | +58.5% | 1.38 | +6.7% | +78.2% |
| ml_ensemble_cost2x | +34.7% | +5.2% | 0.33 (-0.50, 1.06) | 0.44 | -53.4% | 465 | +53.3% | 1.06 | +9.8% | +166.1% |
| ml_ensemble_cost3x | -26.3% | -5.1% | -0.06 (-0.88, 0.68) | -0.08 | -58.1% | 465 | +48.6% | 0.94 | +9.8% | +184.8% |
| ema_cross_24_168_cost2x | +295.9% | +26.6% | 0.76 (-0.18, 1.58) | 1.05 | -61.5% | 488 | +24.6% | 1.15 | +54.4% | +427.8% |

## Clasificación fuera de muestra (y=1: comprar supera costes en el horizonte)

| Símbolo:modelo | Tasa base | AUC | Brier | Brier base | Bal. acc. | Precisión@entrada | Cobertura@entrada |
|---|---|---|---|---|---|---|---|
| BTCUSDT:p_logistic | 0.425 | 0.553 | 0.2431 | 0.2444 | 0.521 | 0.544 | +4.4% |
| BTCUSDT:p_lightgbm | 0.425 | 0.547 | 0.2433 | 0.2444 | 0.524 | 0.553 | +4.4% |
| BTCUSDT:p_ensemble | 0.425 | 0.554 | 0.2424 | 0.2444 | 0.525 | 0.564 | +3.7% |
| ETHUSDT:p_logistic | 0.453 | 0.544 | 0.2464 | 0.2478 | 0.525 | 0.538 | +4.2% |
| ETHUSDT:p_lightgbm | 0.453 | 0.544 | 0.2468 | 0.2478 | 0.523 | 0.577 | +6.3% |
| ETHUSDT:p_ensemble | 0.453 | 0.548 | 0.2460 | 0.2478 | 0.526 | 0.595 | +3.6% |

## Retorno por año

| Estrategia | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|
| buy_and_hold | +386.5% | +258.9% | -67.3% | +104.0% | +64.1% | +15.0% |
| ema_cross_24_168 | +213.4% | +96.2% | -41.1% | +27.7% | +66.8% | -2.6% |
| ema_cross_24_168_cost2x | +180.6% | +75.4% | -47.3% | +13.6% | +50.8% | -10.8% |
| ml_ensemble | +51.2% | +59.1% | -13.4% | +0.0% | +8.9% | +7.7% |
| ml_ensemble_cost2x | +28.6% | +11.0% | -16.8% | +0.0% | +7.6% | +4.7% |
| ml_ensemble_cost3x | +9.3% | -22.5% | -20.2% | +0.0% | +6.4% | +1.8% |
| ml_ensemble_risk | +67.5% | +43.3% | -10.3% | +0.0% | +8.9% | +7.7% |
| ml_lightgbm | +101.4% | +78.7% | -26.0% | +0.0% | +0.0% | +10.2% |
| ml_logistic | +29.8% | +43.1% | -13.6% | -0.5% | +30.0% | +32.9% |
| price_above_ema_fast | -9.9% | -24.3% | -52.6% | -50.2% | -42.7% | -53.2% |
| rsi_reversion | -23.2% | +14.0% | -41.2% | +1.9% | -21.9% | -9.6% |
| tsmom_168 | +188.8% | +51.0% | -49.2% | +13.9% | +8.3% | -13.9% |

## Por régimen de mercado (BTC; definido de forma causal)

| Estrategia | Dimensión | Régimen | Velas | Retorno anualizado | Sharpe |
|---|---|---|---|---|---|
| buy_and_hold | trend | alcista | 17710 | +335.7% | 4.84 |
| buy_and_hold | trend | bajista | 9748 | -334.3% | -3.25 |
| buy_and_hold | trend | lateral | 23654 | +60.2% | 1.00 |
| buy_and_hold | vol | alta_vol | 3993 | +278.7% | 1.89 |
| buy_and_hold | vol | normal_vol | 47119 | +63.6% | 1.01 |
| price_above_ema_fast | trend | alcista | 17710 | +8.5% | 0.19 |
| price_above_ema_fast | trend | bajista | 9748 | -115.0% | -3.77 |
| price_above_ema_fast | trend | lateral | 23654 | -61.0% | -2.00 |
| price_above_ema_fast | vol | alta_vol | 3993 | -103.9% | -1.91 |
| price_above_ema_fast | vol | normal_vol | 47119 | -42.4% | -1.23 |
| ema_cross_24_168 | trend | alcista | 17710 | +206.5% | 3.66 |
| ema_cross_24_168 | trend | bajista | 9748 | -127.3% | -3.76 |
| ema_cross_24_168 | trend | lateral | 23654 | -6.7% | -0.19 |
| ema_cross_24_168 | vol | alta_vol | 3993 | +28.0% | 0.42 |
| ema_cross_24_168 | vol | normal_vol | 47119 | +45.5% | 1.10 |
| tsmom_168 | trend | alcista | 17710 | +184.4% | 3.34 |
| tsmom_168 | trend | bajista | 9748 | -129.1% | -3.53 |
| tsmom_168 | trend | lateral | 23654 | -32.3% | -0.90 |
| tsmom_168 | vol | alta_vol | 3993 | +38.9% | 0.58 |
| tsmom_168 | vol | normal_vol | 47119 | +23.1% | 0.56 |
| rsi_reversion | trend | alcista | 17710 | +62.0% | 2.30 |
| rsi_reversion | trend | bajista | 9748 | -133.4% | -1.98 |
| rsi_reversion | trend | lateral | 23654 | -11.3% | -0.39 |
| rsi_reversion | vol | alta_vol | 3993 | +205.8% | 2.23 |
| rsi_reversion | vol | normal_vol | 47119 | -27.4% | -0.91 |
| ml_logistic | trend | alcista | 17710 | +71.6% | 2.28 |
| ml_logistic | trend | bajista | 9748 | -81.7% | -2.17 |
| ml_logistic | trend | lateral | 23654 | +26.2% | 1.37 |
| ml_logistic | vol | alta_vol | 3993 | +41.9% | 0.60 |
| ml_logistic | vol | normal_vol | 47119 | +19.6% | 0.95 |
| ml_lightgbm | trend | alcista | 17710 | +86.1% | 3.17 |
| ml_lightgbm | trend | bajista | 9748 | -66.0% | -1.80 |
| ml_lightgbm | trend | lateral | 23654 | +9.8% | 0.55 |
| ml_lightgbm | vol | alta_vol | 3993 | +45.8% | 0.82 |
| ml_lightgbm | vol | normal_vol | 47119 | +19.7% | 0.93 |
| ml_ensemble | trend | alcista | 17710 | +64.4% | 2.12 |
| ml_ensemble | trend | bajista | 9748 | -63.1% | -1.81 |
| ml_ensemble | trend | lateral | 23654 | +18.8% | 1.04 |
| ml_ensemble | vol | alta_vol | 3993 | +58.7% | 0.92 |
| ml_ensemble | vol | normal_vol | 47119 | +15.6% | 0.77 |
| ml_ensemble_risk | trend | alcista | 17710 | +47.0% | 1.92 |
| ml_ensemble_risk | trend | bajista | 9748 | -27.9% | -1.23 |
| ml_ensemble_risk | trend | lateral | 23654 | +15.5% | 0.96 |
| ml_ensemble_risk | vol | alta_vol | 3993 | +59.9% | 1.50 |
| ml_ensemble_risk | vol | normal_vol | 47119 | +14.6% | 0.81 |
| ml_ensemble_cost2x | trend | alcista | 17710 | +50.9% | 1.67 |
| ml_ensemble_cost2x | trend | bajista | 9748 | -77.9% | -2.23 |
| ml_ensemble_cost2x | trend | lateral | 23654 | +12.7% | 0.70 |
| ml_ensemble_cost2x | vol | alta_vol | 3993 | +23.9% | 0.37 |
| ml_ensemble_cost2x | vol | normal_vol | 47119 | +7.4% | 0.36 |
| ml_ensemble_cost3x | trend | alcista | 17710 | +37.4% | 1.22 |
| ml_ensemble_cost3x | trend | bajista | 9748 | -92.4% | -2.64 |
| ml_ensemble_cost3x | trend | lateral | 23654 | +6.6% | 0.36 |
| ml_ensemble_cost3x | vol | alta_vol | 3993 | -10.2% | -0.16 |
| ml_ensemble_cost3x | vol | normal_vol | 47119 | -0.9% | -0.04 |
| ema_cross_24_168_cost2x | trend | alcista | 17710 | +196.4% | 3.48 |
| ema_cross_24_168_cost2x | trend | bajista | 9748 | -138.3% | -4.08 |
| ema_cross_24_168_cost2x | trend | lateral | 23654 | -18.3% | -0.51 |
| ema_cross_24_168_cost2x | vol | alta_vol | 3993 | +17.7% | 0.27 |
| ema_cross_24_168_cost2x | vol | normal_vol | 47119 | +34.5% | 0.83 |

## Advertencias

- Resultados simulados: no equivalen a beneficios reales.
- Los IC95 % del Sharpe usan bootstrap de bloques de 168 h; son aproximados.
- Los umbrales de la estrategia están pre-registrados en configs/base.yaml (no optimizados sobre estos folds).

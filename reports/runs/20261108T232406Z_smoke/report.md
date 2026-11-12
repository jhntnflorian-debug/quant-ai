# Reporte de experimento `20261108T232406Z_smoke`

- Tipo: **smoke** (prueba mínima: NO son resultados válidos)
- Periodo evaluado (fuera de muestra): 2020-01-01 00:00:00+00:00 → 2020-06-30 23:00:00+00:00
- Holdout final (2025-11-01 00:00:00+00:00 en adelante): sin tocar
- Costes: comisión 10.0 bps + medio spread 1.0 bps + slippage 2.0 bps por lado

## Estrategias (backtest con costes, señal al cierre → ejecución en la siguiente apertura)

| Estrategia | Retorno | CAGR | Sharpe (IC95 %) | Sortino | Máx. DD | Operaciones | Win rate | Profit factor | Exposición | Costes/capital |
|---|---|---|---|---|---|---|---|---|---|---|
| buy_and_hold | +13.6% | +29.2% | 0.81 (-1.47, 3.80) | 1.08 | -35.9% | 0 | — | — | +53.7% | +0.1% |
| price_above_ema_fast | -1.0% | -2.1% | 0.01 (-2.29, 2.07) | 0.02 | -14.2% | 176 | +20.5% | 0.98 | +21.6% | +23.9% |
| ema_cross_24_168 | +9.7% | +20.5% | 0.85 (-1.50, 2.96) | 1.15 | -14.4% | 18 | +38.9% | 1.51 | +30.3% | +2.5% |
| tsmom_168 | +9.9% | +21.0% | 0.85 (-1.37, 3.03) | 1.20 | -12.9% | 71 | +25.4% | 1.29 | +29.7% | +9.8% |
| rsi_reversion | -12.8% | -24.1% | -0.72 (-2.60, 1.56) | -0.92 | -30.9% | 25 | +52.0% | 0.53 | +10.9% | +3.0% |
| ml_logistic | +5.2% | +10.8% | 0.78 (-1.24, 3.34) | 1.00 | -7.5% | 16 | +81.2% | 1.44 | +1.0% | +2.1% |
| ml_lightgbm | +4.2% | +8.6% | 1.71 (-2.61, 4.71) | 2.79 | -3.8% | 7 | +71.4% | 2.42 | +0.8% | +0.9% |
| ml_ensemble | +2.4% | +4.9% | 1.59 (0.00, 2.54) | 2.86 | -1.5% | 3 | +100.0% | — | +0.3% | +0.4% |
| ml_ensemble_risk | +2.4% | +4.9% | 1.59 (0.00, 2.54) | 2.86 | -1.5% | 3 | +100.0% | — | +0.3% | +0.4% |
| ml_ensemble_cost2x | +2.0% | +4.0% | 1.35 (0.00, 2.28) | 2.35 | -1.5% | 3 | +100.0% | — | +0.3% | +0.8% |
| ml_ensemble_cost3x | +1.6% | +3.2% | 1.10 (0.00, 1.95) | 1.85 | -1.5% | 3 | +100.0% | — | +0.3% | +1.2% |
| ema_cross_24_168_cost2x | +7.2% | +15.0% | 0.67 (-1.70, 2.80) | 0.91 | -15.3% | 18 | +38.9% | 1.35 | +30.3% | +5.0% |

## Clasificación fuera de muestra (y=1: comprar supera costes en el horizonte)

| Símbolo:modelo | Tasa base | AUC | Brier | Brier base | Bal. acc. | Precisión@entrada | Cobertura@entrada |
|---|---|---|---|---|---|---|---|
| BTCUSDT:p_logistic | 0.442 | 0.558 | 0.2501 | 0.2466 | 0.515 | 0.694 | +2.0% |
| BTCUSDT:p_lightgbm | 0.442 | 0.543 | 0.2524 | 0.2466 | 0.503 | 0.641 | +0.9% |
| BTCUSDT:p_ensemble | 0.442 | 0.558 | 0.2497 | 0.2466 | 0.512 | 0.639 | +0.8% |

## Retorno por año

| Estrategia | 2020 |
|---|---|
| buy_and_hold | +13.6% |
| ema_cross_24_168 | +9.7% |
| ema_cross_24_168_cost2x | +7.2% |
| ml_ensemble | +2.4% |
| ml_ensemble_cost2x | +2.0% |
| ml_ensemble_cost3x | +1.6% |
| ml_ensemble_risk | +2.4% |
| ml_lightgbm | +4.2% |
| ml_logistic | +5.2% |
| price_above_ema_fast | -1.0% |
| rsi_reversion | -12.8% |
| tsmom_168 | +9.9% |

## Por régimen de mercado (BTC; definido de forma causal)

| Estrategia | Dimensión | Régimen | Velas | Retorno anualizado | Sharpe |
|---|---|---|---|---|---|
| buy_and_hold | trend | alcista | 2262 | +146.0% | 4.28 |
| buy_and_hold | trend | bajista | 784 | -130.3% | -1.67 |
| buy_and_hold | trend | lateral | 1309 | -56.9% | -2.14 |
| buy_and_hold | vol | alta_vol | 568 | +15.9% | 0.18 |
| buy_and_hold | vol | normal_vol | 3787 | +38.2% | 1.20 |
| price_above_ema_fast | trend | alcista | 2262 | +21.8% | 0.96 |
| price_above_ema_fast | trend | bajista | 784 | +45.5% | 1.56 |
| price_above_ema_fast | trend | lateral | 1309 | -64.0% | -4.46 |
| price_above_ema_fast | vol | alta_vol | 568 | +3.0% | 0.10 |
| price_above_ema_fast | vol | normal_vol | 3787 | -0.1% | -0.01 |
| ema_cross_24_168 | trend | alcista | 2262 | +87.2% | 3.25 |
| ema_cross_24_168 | trend | bajista | 784 | -64.1% | -1.88 |
| ema_cross_24_168 | trend | lateral | 1309 | -38.9% | -2.18 |
| ema_cross_24_168 | vol | alta_vol | 568 | -37.4% | -1.04 |
| ema_cross_24_168 | vol | normal_vol | 3787 | +31.0% | 1.28 |
| tsmom_168 | trend | alcista | 2262 | +69.4% | 2.67 |
| tsmom_168 | trend | bajista | 784 | -3.4% | -0.09 |
| tsmom_168 | trend | lateral | 1309 | -42.8% | -2.50 |
| tsmom_168 | vol | alta_vol | 568 | -0.9% | -0.02 |
| tsmom_168 | vol | normal_vol | 3787 | +26.1% | 1.11 |
| rsi_reversion | trend | alcista | 2262 | +16.6% | 1.26 |
| rsi_reversion | trend | bajista | 784 | -122.3% | -1.80 |
| rsi_reversion | trend | lateral | 1309 | -30.4% | -2.02 |
| rsi_reversion | vol | alta_vol | 568 | -44.7% | -0.56 |
| rsi_reversion | vol | normal_vol | 3787 | -19.2% | -1.37 |
| ml_logistic | trend | alcista | 2262 | +14.6% | 3.67 |
| ml_logistic | trend | bajista | 784 | +26.3% | 0.79 |
| ml_logistic | trend | lateral | 1309 | -3.3% | -1.92 |
| ml_logistic | vol | alta_vol | 568 | +34.9% | 0.89 |
| ml_logistic | vol | normal_vol | 3787 | +7.8% | 2.40 |
| ml_lightgbm | trend | alcista | 2262 | +10.7% | 1.64 |
| ml_lightgbm | trend | bajista | 784 | +11.4% | 5.47 |
| ml_lightgbm | trend | lateral | 1309 | +2.4% | 1.38 |
| ml_lightgbm | vol | alta_vol | 568 | +0.0% | 0.00 |
| ml_lightgbm | vol | normal_vol | 3787 | +9.6% | 1.84 |
| ml_ensemble | trend | alcista | 2262 | +3.6% | 1.09 |
| ml_ensemble | trend | bajista | 784 | +11.9% | 3.16 |
| ml_ensemble | trend | lateral | 1309 | +2.6% | 1.51 |
| ml_ensemble | vol | alta_vol | 568 | +16.4% | 3.72 |
| ml_ensemble | vol | normal_vol | 3787 | +3.0% | 1.11 |
| ml_ensemble_risk | trend | alcista | 2262 | +3.6% | 1.09 |
| ml_ensemble_risk | trend | bajista | 784 | +11.9% | 3.16 |
| ml_ensemble_risk | trend | lateral | 1309 | +2.6% | 1.51 |
| ml_ensemble_risk | vol | alta_vol | 568 | +16.4% | 3.72 |
| ml_ensemble_risk | vol | normal_vol | 3787 | +3.0% | 1.11 |
| ml_ensemble_cost2x | trend | alcista | 2262 | +3.1% | 0.94 |
| ml_ensemble_cost2x | trend | bajista | 784 | +10.4% | 2.93 |
| ml_ensemble_cost2x | trend | lateral | 1309 | +1.7% | 1.03 |
| ml_ensemble_cost2x | vol | alta_vol | 568 | +14.3% | 3.44 |
| ml_ensemble_cost2x | vol | normal_vol | 3787 | +2.4% | 0.90 |
| ml_ensemble_cost3x | trend | alcista | 2262 | +2.6% | 0.79 |
| ml_ensemble_cost3x | trend | bajista | 784 | +8.9% | 2.65 |
| ml_ensemble_cost3x | trend | lateral | 1309 | +0.9% | 0.51 |
| ml_ensemble_cost3x | vol | alta_vol | 568 | +12.3% | 3.11 |
| ml_ensemble_cost3x | vol | normal_vol | 3787 | +1.8% | 0.68 |
| ema_cross_24_168_cost2x | trend | alcista | 2262 | +82.6% | 3.08 |
| ema_cross_24_168_cost2x | trend | bajista | 784 | -70.6% | -2.07 |
| ema_cross_24_168_cost2x | trend | lateral | 1309 | -42.8% | -2.40 |
| ema_cross_24_168_cost2x | vol | alta_vol | 568 | -42.4% | -1.17 |
| ema_cross_24_168_cost2x | vol | normal_vol | 3787 | +26.3% | 1.09 |

## Advertencias

- Resultados simulados: no equivalen a beneficios reales.
- Los IC95 % del Sharpe usan bootstrap de bloques de 168 h; son aproximados.
- Los umbrales de la estrategia están pre-registrados en configs/base.yaml (no optimizados sobre estos folds).

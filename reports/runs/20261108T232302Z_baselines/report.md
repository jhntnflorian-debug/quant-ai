# Reporte de experimento `20261108T232302Z_baselines`

- Tipo: **baselines_only**
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
| ema_cross_24_168_cost2x | +295.9% | +26.6% | 0.76 (-0.18, 1.58) | 1.05 | -61.5% | 488 | +24.6% | 1.15 | +54.4% | +427.8% |

## Retorno por año

| Estrategia | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|
| buy_and_hold | +386.5% | +258.9% | -67.3% | +104.0% | +64.1% | +15.0% |
| ema_cross_24_168 | +213.4% | +96.2% | -41.1% | +27.7% | +66.8% | -2.6% |
| ema_cross_24_168_cost2x | +180.6% | +75.4% | -47.3% | +13.6% | +50.8% | -10.8% |
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
| ema_cross_24_168_cost2x | trend | alcista | 17710 | +196.4% | 3.48 |
| ema_cross_24_168_cost2x | trend | bajista | 9748 | -138.3% | -4.08 |
| ema_cross_24_168_cost2x | trend | lateral | 23654 | -18.3% | -0.51 |
| ema_cross_24_168_cost2x | vol | alta_vol | 3993 | +17.7% | 0.27 |
| ema_cross_24_168_cost2x | vol | normal_vol | 47119 | +34.5% | 0.83 |

## Advertencias

- Resultados simulados: no equivalen a beneficios reales.
- Los IC95 % del Sharpe usan bootstrap de bloques de 168 h; son aproximados.
- Los umbrales de la estrategia están pre-registrados en configs/base.yaml (no optimizados sobre estos folds).

# Reporte de experimento `20261108T232220Z_baselines`

- Tipo: **baselines_only**
- Periodo evaluado (fuera de muestra): 2020-01-01 00:00:00+00:00 → 2025-10-31 23:00:00+00:00
- Holdout final (2025-11-01 00:00:00+00:00 en adelante): sin tocar
- Costes: comisión 10.0 bps + medio spread 1.0 bps + slippage 2.0 bps por lado

## Estrategias (backtest con costes, señal al cierre → ejecución en la siguiente apertura)

| Estrategia | Retorno | CAGR | Sharpe (IC95 %) | Sortino | Máx. DD | Operaciones | Win rate | Profit factor | Exposición | Costes/capital |
|---|---|---|---|---|---|---|---|---|---|---|
| buy_and_hold | +2444.1% | +74.1% | 1.15 (0.37, 1.96) | 1.60 | -77.7% | 0 | — | — | +100.0% | +5.1% |
| trend_ema | -95.7% | -41.7% | -1.30 (-2.18, -0.54) | -1.86 | -96.9% | 4404 | +17.5% | 0.88 | +39.8% | +288.8% |
| tsmom_168 | +136.1% | +15.9% | 0.56 (-0.43, 1.41) | 0.78 | -69.0% | 1725 | +22.7% | 1.06 | +54.8% | +670.0% |
| rsi_reversion | -62.5% | -15.5% | -0.24 (-0.86, 0.39) | -0.32 | -70.3% | 628 | +62.4% | 0.81 | +21.0% | +50.3% |
| trend_ema_cost2x | -99.9% | -69.4% | -3.28 (-4.25, -2.38) | -4.62 | -99.9% | 3595 | +15.3% | 0.70 | +32.5% | +188.2% |

## Retorno por año

| Estrategia | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|
| buy_and_hold | +397.7% | +201.5% | -65.5% | +123.2% | +80.8% | +18.9% |
| rsi_reversion | -23.2% | +14.0% | -41.2% | +1.9% | -21.9% | -9.6% |
| trend_ema | -9.9% | -24.3% | -52.6% | -50.2% | -42.7% | -53.2% |
| trend_ema_cost2x | -69.8% | -72.7% | -77.7% | -82.3% | -69.6% | +0.0% |
| tsmom_168 | +188.8% | +51.0% | -49.2% | +13.9% | +8.3% | -13.9% |

## Por régimen de mercado (BTC; definido de forma causal)

| Estrategia | Dimensión | Régimen | Velas | Retorno anualizado | Sharpe |
|---|---|---|---|---|---|
| buy_and_hold | trend | alcista | 17710 | +326.0% | 4.90 |
| buy_and_hold | trend | bajista | 9748 | -316.4% | -3.28 |
| buy_and_hold | trend | lateral | 23654 | +58.0% | 1.03 |
| buy_and_hold | vol | alta_vol | 3993 | +265.1% | 1.89 |
| buy_and_hold | vol | normal_vol | 47119 | +63.7% | 1.07 |
| trend_ema | trend | alcista | 17710 | +8.5% | 0.19 |
| trend_ema | trend | bajista | 9748 | -115.0% | -3.77 |
| trend_ema | trend | lateral | 23654 | -61.0% | -2.00 |
| trend_ema | vol | alta_vol | 3993 | -103.9% | -1.91 |
| trend_ema | vol | normal_vol | 47119 | -42.4% | -1.23 |
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
| trend_ema_cost2x | trend | alcista | 17710 | -101.8% | -2.35 |
| trend_ema_cost2x | trend | bajista | 9748 | -155.9% | -5.19 |
| trend_ema_cost2x | trend | lateral | 23654 | -102.7% | -3.68 |
| trend_ema_cost2x | vol | alta_vol | 3993 | -149.5% | -2.80 |
| trend_ema_cost2x | vol | normal_vol | 47119 | -109.4% | -3.39 |

## Advertencias

- Resultados simulados: no equivalen a beneficios reales.
- Los IC95 % del Sharpe usan bootstrap de bloques de 168 h; son aproximados.
- Los umbrales de la estrategia están pre-registrados en configs/base.yaml (no optimizados sobre estos folds).

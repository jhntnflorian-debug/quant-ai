# Experimento v2 `20261109T133748Z_v2smoke_ta`

- Tramo de prueba (walk-forward): 2022-01-01 00:00:00+00:00 → 2022-03-31 23:00:00+00:00 — **PRUEBA MÍNIMA, no válida**
- Holdout final (2025-11-01 00:00:00+00:00): sin tocar
- Intentos contados en el DSR: 5

| Estrategia | Retorno | Sharpe | Máx. DD | Operaciones | PSR | DSR | Retorno estrés ×2 | BNB 7.5 bps | Puerta |
|---|---|---|---|---|---|---|---|---|---|
| ml_v1_base_h24 | +0.0% | 0.00 | +0.0% | 0 | — | — | +0.0% | +0.0% | no |
| ml_v2_ta_h24 | +1.4% | 1.22 | -2.3% | 1 | 0.728 | 0.529 | +1.2% | +1.4% | no |
| buy_and_hold | -1.3% | -0.02 | -15.5% | 0 | 0.497 | 0.295 | -1.4% | -1.3% | ref. |
| ema_cross_24_168 | -6.0% | -1.16 | -14.6% | 9 | 0.282 | 0.134 | -7.2% | -5.8% | ref. |
| ema_cross_vol_target | -1.7% | -0.61 | -7.0% | 9 | 0.382 | 0.203 | -2.5% | -1.6% | ref. |

## Calidad predictiva fuera de muestra

| Variante:símbolo | n | Tasa base | AUC | Brier | Brier base | Precisión@entrada | Cobertura |
|---|---|---|---|---|---|---|---|
| v1_base_h24:BTCUSDT | 2160 | 0.455 | 0.519 | 0.2488 | 0.2480 | — | +0.0% |
| v2_ta_h24:BTCUSDT | 2160 | 0.455 | 0.491 | 0.2488 | 0.2480 | 1.000 | +0.0% |

## Lectura

- Contribución incremental: comparar v1 → v2 (análisis técnico) → v3 (datos externos) con el mismo horizonte.
- Meta-etiquetado: v4 y v5 frente a ema_cross_24_168 y ema_cross_vol_target (misma señal, sin ML).
- DSR < 0.95: el resultado puede explicarse por haber probado varias variantes.
- Resultados simulados; no son beneficios reales.

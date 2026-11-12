# Especificación: datos de otras plataformas con alineación point-in-time

Estado: especificación para implementar (2026-11-09). Fuentes verificadas manualmente ese día.

## Convención temporal

La fila con `open_time = t` describe la vela [t, t+1h) y se conoce al cierre: `decision_time = t + 1h`.
Cada serie externa normalizada lleva la columna `available_at` (UTC): el primer instante en que el valor pudo conocerse. Un valor solo se usa en la fila t si `available_at <= decision_time`.

## Restricciones

- No instalar paquetes. Usar requests, pandas, numpy y pyarrow.
- No descargar datos masivos en el PC local: allí solo peticiones mínimas de prueba. Las descargas completas corren en Colab.
- Las pruebas no usan red: fixtures y strings embebidos.
- Crear solo los archivos indicados. No modificar `configs/base.yaml`, `src/quant_ai/features/engineering.py`, `src/quant_ai/features/technical.py`, `src/quant_ai/training/*`, ni el resto del código existente.

## 1. `src/quant_ai/data/external.py`

Descargadores incrementales y reanudables:
- Caché de originales en `<data_dir>/external/<fuente>/`.
- Parquet normalizado en `<data_dir>/external/<fuente>[_<simbolo>].parquet`.
- Reintentos con backoff; respetar HTTP 429/418 y Retry-After.
- `max_workers` configurable: 2 por defecto, 8 en Colab.
- Manifiesto en `<data_dir>/external/manifest.json` con URL base, filas, rango, fecha de descarga y SHA256 de cada Parquet.

| Fuente | Origen y formato | Desde | available_at |
|---|---|---|---|
| Perpetuos Binance USD-M, klines 1h | `data.binance.vision/data/futures/um/monthly/klines/<SYM>/1h/<SYM>-1h-YYYY-MM.zip`; mismo formato que spot (ms, sin encabezado, incluye taker buy) | BTC 2019-09, ETH 2019-11 | open_time + 1h |
| Premium index 1h | `.../futures/um/monthly/premiumIndexKlines/<SYM>/1h/<SYM>-1h-YYYY-MM.zip`; formato klines | 2019-12 / 2020-01 | open_time + 1h |
| Funding | `.../futures/um/monthly/fundingRate/<SYM>/<SYM>-fundingRate-YYYY-MM.zip`; encabezado `calc_time,funding_interval_hours,last_funding_rate` | 2020-01 | calc_time + 1 min (conservador) |
| Open interest y ratios (5 min) | `.../futures/um/daily/metrics/<SYM>/<SYM>-metrics-YYYY-MM-DD.zip`; encabezado `create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio`; create_time texto UTC `YYYY-MM-DD HH:MM:SS` | BTC 2020-09-01; ETH entre 2021-09 y 2021-12 (detectar el primer día con HTTP 200) | create_time + 5 min; agregar a 1h con el último snapshot disponible |
| Coinbase BTC-USD, ETH-USD 1h | `api.exchange.coinbase.com/products/{BTC-USD,ETH-USD}/candles?granularity=3600&start=ISO&end=ISO`, máximo 300 velas por petición; columnas **[time, low, high, open, close, volume]**, time en segundos = inicio de la hora | 2018-01 | time + 1h |
| Deribit DVOL 1h | `www.deribit.com/api/v2/public/get_volatility_index_data?currency={BTC,ETH}&start_timestamp=ms&end_timestamp=ms&resolution=3600`; `result.data = [[ts, open, high, low, close]]`; paginar por ventanas o con `result.continuation` | 2021-03-24 | ts + 1h |
| Fear & Greed | `api.alternative.me/fng/?limit=0&format=json`; `data[].{value, value_classification, timestamp}` (segundos, 00:00 UTC del día) | 2018-02-01 | timestamp + 2h |
| Stablecoins (DefiLlama) | `stablecoins.llama.fi/stablecoincharts/all`; `[{date, totalCirculatingUSD.peggedUSD}]` | 2017-11-29 | date + 1 día + 1h |
| FRED (sin API key) | `fred.stlouisfed.org/graph/fredgraph.csv?id=SERIE`; faltantes como `.` | según serie | VIXCLS, SP500 y DGS10: fecha + 2 días. DTWEXBGS: fecha + 8 días (publicación semanal) |
| Spot de alts (amplitud) | BNBUSDT, SOLUSDT, XRPUSDT, ADAUSDT, DOGEUSDT 1h desde data.binance.vision; reutilizar `quant_ai.data.binance` (`fetch_vision_file`, `read_vision_zip`, `parse_klines`; maneja µs desde 2025-01) | 2018 / según par | open_time + 1h |
| CoinMetrics community (on-chain) | `community-api.coinmetrics.io/v4/timeseries/asset-metrics?assets=btc,eth&metrics=AdrActCnt,TxCnt&frequency=1d&start_time=2018-01-01&page_size=10000`; paginar con `next_page_url` | 2018 | time + 1 día + 2h |

Para el mes o el día en curso de Binance, completar con la API REST pública:
- `fapi.binance.com/fapi/v1/klines`, `/fapi/v1/premiumIndexKlines` y `/fapi/v1/fundingRate` (con `startTime` > 0; con 0 la API devuelve los últimos valores).
- `/futures/data/openInterestHist`, `/futures/data/topLongShortAccountRatio`, `/futures/data/globalLongShortAccountRatio` y `/futures/data/takerlongshortRatio` con `period=5m` (solo últimos 30 días).
- Descartar velas no cerradas.

## 2. `src/quant_ai/features/external.py`

- `align_point_in_time(series, decision_times)`: merge_asof hacia atrás sobre `available_at <= decision_time`. Es la ÚNICA forma permitida de unir datos externos.
- `build_external_features(data_dir, symbol, bar_index)`: index = bar_index (open_time) y decision_time = bar_index + 1h.
  - Columnas con prefijo `ext_`.
  - NaN donde la fuente aún no existe; nunca rellenar hacia atrás.
  - Ventanas móviles solo hacia atrás.
  - Debe funcionar aunque falten algunas fuentes: las omite con un aviso.
- `external_coverage(df)`: primera fecha no nula y porcentaje de cobertura por columna.

Características (prefijo `ext_` omitido):

| Grupo | Características |
|---|---|
| Derivados | funding_rate; funding_mean_3d (media de los últimos 9 cobros); funding_z_30d; premium_close; premium_mean_24h; basis_perp_spot (close perp / close spot − 1, con `data/clean/<SYM>_1h.parquet`); perp_spot_volume_ratio_24h; perp_taker_buy_ratio_24h; oi_change_1h y oi_change_24h (log); oi_value_z_30d; toptrader_ls_ratio; global_ls_ratio; taker_ls_vol_ratio_24h |
| Coinbase | coinbase_premium (close Coinbase USD / close Binance USDT − 1, en la misma hora); coinbase_premium_mean_24h |
| Volatilidad implícita | dvol (BTC para BTCUSDT, ETH para ETHUSDT); dvol_change_24h; dvol_minus_realized (DVOL/100 − volatilidad realizada anualizada de 24 h del spot) |
| Sentimiento y liquidez | fng (/100); fng_change_7d; stablecoin_growth_7d; stablecoin_growth_30d |
| Macro | vix; vix_change_5d; spx_ret_5d; dgs10_change_5d; dxy_change_20d |
| Amplitud | breadth_above_ema168 (fracción de las 5 alts sobre su EMA de 168 h); alt_minus_btc_ret_24h |
| On-chain | active_addr_z_30d; tx_count_z_30d |

## 3. `scripts/download_extended.py`

CLI con `--sources all|perp,coinbase,deribit,fng,stable,fred,alts,onchain`, `--symbols`, `--workers` y `--since`. Imprime el progreso y un resumen de cobertura.

## 4. `tests/test_external.py` (sin red)

- Parseo de cada formato con strings embebidos: el orden de columnas de Coinbase, el CSV de FRED con `.` y los encabezados de metrics y fundingRate.
- `align_point_in_time` nunca usa una fila con `available_at > decision_time`.
- Alterar valores con `available_at` posterior a T no cambia las características de las filas con `decision_time <= T`.
- El funding publicado exactamente al cierre de una vela NO se usa en esa vela (margen de 1 minuto).
- `build_external_features` funciona con un directorio que tiene solo algunas fuentes.

## Criterio de terminado

- `py_compile` de todo y `.venv\Scripts\python.exe -m pytest -q` pasando completo.
- Una prueba manual mínima con red (1 mes de funding y 2 días de Coinbase) guardada en un directorio temporal, no en `data/`.

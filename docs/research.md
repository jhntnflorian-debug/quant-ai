# Investigación y selección del stack

Fecha de verificación: 2026-11-08. Convención de este documento:

- **Verificado**: leído en una fuente oficial. Se cita el enlace.
- **Secundario**: solo en fuentes de terceros. Úsese con cautela.
- **No verificado**: no se encontró fuente oficial. Queda como pendiente.
- **Medido**: resultado ejecutado en este proyecto con datos reales.

## 1. Selección de mercado

| Criterio | Cripto spot (BTC/ETH) | Acciones y ETF EE. UU. | Forex | Futuros CME micro |
|---|---|---|---|---|
| Horario | 24/7 | 9:30 a 16:00 ET en sesión regular ([NYSE](https://www.nyse.com/markets/hours-calendars), verificado) | 24/5 (no verificado) | Casi 24/5 |
| Histórico gratis en 1h | Completo desde 2017 en data.binance.vision (medido: 77,495 velas por par desde 2018-01) | Alpaca Basic: solo IEX, desde 2016 ([Alpaca](https://docs.alpaca.markets/us/docs/about-market-data-api), verificado) | Depende del broker | De pago (Databento u otros) |
| Coste de negociación | 0.10 % maker/taker en el nivel regular de Binance ([tabla](https://www.binance.com/en/fee/exchange-makertaker-url), verificado) | Comisión cero en muchos brokers, más spread | Spread | Comisión más margen |
| Acceso desde Colombia | Colombia no figura en la lista de países prohibidos de Binance ([lista, fechada 2024-02-27](https://www.binance.com/en/legal/list-of-prohibited-countries/), verificado) | Alpaca: país no confirmado. IBKR: secundario | Según broker | Vía IBKR (secundario) |
| Paper o testnet sin coste | Binance Spot Testnet ([testnet.binance.vision](https://testnet.binance.vision/), verificado y accesible desde este equipo) | Alpaca paper | Cuentas demo | IBKR paper |
| Apalancamiento | No necesario (spot) | No necesario | Inherente | Inherente |
| Regla PDT | No aplica | El mínimo de 25,000 USD fue reemplazado por un estándar de margen intradía, vigente desde 2026-06-04 ([FINRA 26-10](https://www.finra.org/rules-guidance/notices/26-10), verificado) | No aplica | No aplica |

**Decisión**: BTCUSDT y ETHUSDT spot, velas de 1h, sin apalancamiento. Se confirma la hipótesis inicial. Las razones son el histórico gratuito y completo, la operación 24/7 (más muestras por año), la API pública sin claves, el testnet gratuito y que el acceso desde Colombia no está prohibido. Las acciones de EE. UU. quedan como segunda fase: Alpaca no confirmó acceso para Colombia y su plan gratuito solo trae datos IEX.

## 2. Proveedores de datos

| Proveedor | Cobertura y frecuencia | Coste y límites | Licencia | Recomendación |
|---|---|---|---|---|
| Binance (data.binance.vision + REST) | Spot, klines desde 1s hasta 1M, trades | Gratis. REST limitado por peso y por IP. HTTP 429 al exceder y 418 en caso de ban ([límites](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/limits), verificado) | Términos de Binance; no redistribuir | **Elegido** |
| CCXT | Capa unificada para más de 100 exchanges; versión 4.5.92 en PyPI (verificado) | Gratis (MIT) | Hereda las del exchange | Útil para multi-exchange; no hace falta en el MVP |
| Alpaca Market Data | Acciones EE. UU. Basic: IEX gratis; Algo Trader Plus: 99 USD/mes con todos los exchanges ([Alpaca](https://docs.alpaca.markets/us/docs/about-market-data-api), verificado) | 200 solicitudes/min en la API de trading (secundario) | Términos de Alpaca | Fase 2 (acciones) |
| Interactive Brokers | Multiactivo | Datos con suscripción mensual (no verificado) | Términos de IBKR | Fase 3 |
| Databento | Futuros y acciones, tick y MBO | Pago por uso; 125 USD de crédito inicial (secundario) | Comercial | Cuando haga falta microestructura |
| Polygon / Massive | Acciones, opciones y cripto | Stocks Advanced: 199 USD/mes ([massive.com/pricing](https://massive.com/pricing), verificado). Plan gratuito: no verificado | Comercial | No necesario por ahora |
| Yahoo Finance / yfinance | Diario e intradía limitado | Los términos prohíben la recolección automatizada sin permiso y el uso comercial ([Yahoo ToS](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html), verificado) | Uso personal | **Descartado** para el sistema |
| FRED | Macroeconomía de EE. UU. | API key obligatoria ([FRED](https://fred.stlouisfed.org/docs/api/api_key.html), verificado) | Términos de FRED | Fase 2 (contexto macro) |
| Binance USD-M (funding, open interest) | Funding con historial; `openInterestHist` solo cubre el último mes ([docs](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest-Statistics), verificado) | Gratis | Términos de Binance | Funding sí; el open interest exige almacenar hacia adelante |
| Fear & Greed (alternative.me) | Diario | Gratis con atribución ([alternative.me](https://alternative.me/crypto/fear-and-greed-index/), verificado) | Atribución obligatoria | Candidato, con desfase de publicación |

Estimación de volumen antes de descargar: 2 pares × 8.9 años × 8,760 velas ≈ 155,000 filas. Medido en disco: 21 MB entre zips originales y Parquet.

## 3. Variables de entrenamiento

Implementadas en el MVP (39 por activo, todas causales):

- Retornos logarítmicos (1, 6, 12, 24, 72 y 168 h) y volatilidad realizada (12, 24, 72 y 168 h), más el ratio de volatilidad 24/168.
- Distancia a las EMA de 12, 24, 72 y 168 h, MACD normalizado e histograma, ADX(14), RSI(14) y ATR(14) relativo.
- Bandas de Bollinger (%b y ancho), rango de la vela, posición del cierre dentro del rango, distancia al máximo y al mínimo de 168 h.
- Volumen (z-score de 168 h), **volumen agresor comprador** (`taker_buy_base / volume`, instantáneo y media de 24 h), tamaño medio de operación y distancia a un VWAP de 24 h.
- Calendario (hora y día en seno/coseno).
- Del otro activo: retorno de 1 h y de 24 h, correlación móvil de 168 h y retorno relativo de 24 h.

Importancia medida (ganancia media de LightGBM entre folds, BTC): correlación BTC–ETH de 168 h 7.9 %, volatilidad de 168 h 5.5 %, volumen agresor de 24 h 5.1 %, volatilidad de 72 h 5.1 %, distancia al máximo de 168 h 4.4 %. La importancia es asociativa y no prueba causalidad.

Pendientes, cuando haya datos fiables con su hora real de publicación: libro de órdenes y desequilibrio (exige capturar hacia adelante), funding rate, dominancia de BTC, calendario macro y sentimiento. El estudio de contribución incremental (ablación por grupos con walk-forward) está pendiente. No se asume que más variables mejoren el modelo.

## 4. Frameworks

| Categoría | Evaluado | Estado | Decisión |
|---|---|---|---|
| ML tradicional | Regresión logística, LightGBM | **Medido** en walk-forward (ver README) | LightGBM 4.7: rápido en CPU, admite NaN nativo y trae TreeSHAP integrado (`pred_contrib`) sin dependencias extra |
| ML tradicional | XGBoost, CatBoost, Random Forest | No probados (fuera del alcance del MVP) | Pendiente. XGBoost 3.4.1 exige Python 3.12 o superior; en Python 3.11 se resuelve XGBoost 3.2.0 (verificado con `uv pip install --dry-run`) |
| Deep learning | PyTorch: GRU, TCN, Transformer | Código listo; entrenamiento pendiente en Colab | PyTorch: Colab lo trae preinstalado con CUDA |
| RL | Gymnasium + Stable-Baselines3 | Pendiente (fase 4) | SB3 2.9 activo (verificado en PyPI). FinRL descartado: último release en 2024-04 |
| Backtesting | Motor propio | **Medido**: 31 pruebas, incluida la coincidencia exacta con comprar y mantener | Backtrader (último release en 2023-04) y vectorbt descartados para el núcleo: un motor pequeño propio es auditable |
| Seguimiento de experimentos | Artefactos JSON, Parquet y CSV versionados por corrida | En uso | MLflow 3.18 queda como opción cuando haya varios usuarios o modelos |
| Hiperparámetros | Optuna | Opcional (`.[tune]`) | Solo en Colab y siempre dentro del periodo de desarrollo |
| Dashboard | Streamlit + Plotly | En uso | React + FastAPI solo si hacen falta multiusuario o autenticación |

Versiones verificadas en PyPI el 2026-11-08: torch 2.15.0, lightgbm 4.7.0, scikit-learn 1.9.1, pandas 3.1.1, streamlit 1.67.0, stable-baselines3 2.9.2, gymnasium 1.4.0, optuna 5.0.0, mlflow 3.18.0.

## 5. Hardware de Google Colab

| Acelerador | GPU | VRAM | Compute capability | BF16 | Fuente |
|---|---|---|---|---|---|
| G4 | NVIDIA RTX PRO 6000 Blackwell (secundario) | ~96 GB | 12.0 | Sí | [McCormick](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/), [Google Cloud G4](https://cloud.google.com/blog/products/compute/g4-vms-powered-by-nvidia-rtx-6000-blackwell-gpus-are-ga) |
| A100 | Ampere | 40 u 80 GB | 8.0 | Sí | [NVIDIA](https://www.nvidia.com/en-us/data-center/a100/) |
| L4 | Ada Lovelace | 24 GB | 8.9 | Sí | [NVIDIA](https://www.nvidia.com/en-us/data-center/l4/) |
| T4 | Turing | 16 GB | 7.5 | No (usar FP16) | [NVIDIA](https://www.nvidia.com/en-us/data-center/tesla-t4/) |
| H100 | Hopper | 80 GB | 9.0 | Sí | Deshabilitada en la cuenta del usuario |
| TPU v5e / v6e | — | — | — | — | PyTorch en TPU requiere TorchTPU o torch_xla; no se usa en el MVP |

- Consumo de unidades por hora: Google no lo publica de forma oficial (no verificado). Las cifras de foros no coinciden entre sí. El notebook mide el tiempo real y calcula el coste con el valor que el usuario lee en su propia interfaz de Colab.
- Límites de sesión según la FAQ oficial ([Colab FAQ](https://research.google.com/colaboratory/faq.html)): hasta 12 h en la versión gratuita, hasta 24 h de ejecución continua en Pro+ si hay unidades, y corte por inactividad.
- **Medido en Colab con L4 (2026-11-09)**: matmul de 64.6 TFLOPS en BF16 y 12.4 en FP32. GRU de 257,000 muestras/s con batch 1024 y un pico de 0.27 GB de VRAM. Cada modelo tardó 71 a 88 s en 8 folds, y la etapa más lenta de la sesión fue el RL en CPU (~18 min por activo). La hipótesis se confirma: la L4 sobra y una G4 o A100 no acortaría el entrenamiento, porque el cuello de botella es la carga de datos.
- La ingeniería de características, el ML tradicional y el backtest corren en CPU. Medido: el walk-forward completo de LR + LightGBM (48 folds) tardó 111 s con 2 hilos.

## 6. Plataformas de ejecución (Colombia)

| Plataforma | Colombia | Paper o testnet | Comisión spot base | Estado |
|---|---|---|---|---|
| Binance | No prohibida (verificado, lista de 2024-02-27) | Spot Testnet gratuito | 0.10 % / 0.10 % (verificado) | **Elegida**. Adaptador testnet listo; falta que el usuario cree las claves |
| Bybit | Probable; hay conflicto entre fuentes (no verificado) | Demo trading | 0.1 % (secundario) | Alternativa |
| OKX | Atiende Colombia vía OKX El Salvador según sus términos (verificado) | Demo con la cabecera `x-simulated-trading` | No verificado | Alternativa |
| Kraken | Hay clientes colombianos (secundario) | Sin sandbox spot de autoservicio | 0.40 % / 0.80 % en el nivel 1 de Kraken Pro (verificado por un agente; no revisado manualmente) | Cara para alta rotación |
| Alpaca | No confirmado para cuentas minoristas | Paper | Acciones con comisión cero | Fase 2 |
| Interactive Brokers | En la lista de países según búsqueda (secundario; la página respondió 403) | Cuenta paper | No verificado | Fase 3 |
| TradingView | — | — | — | Los webhooks exigen 2FA ([TradingView](https://www.tradingview.com/support/solutions/43000529348-about-webhooks/), verificado); el plan requerido no se verificó |
| MetaTrader 5 | Depende del broker | Demo | — | Paquete Python solo para Windows y con el terminal abierto (verificado en PyPI) |
| QuantConnect LEAN | — | — | — | Licencia Apache 2.0 (verificado). No se adopta: el motor propio es más simple de auditar |

## Pendientes de verificación

- Identidad oficial de la G4 de Colab y tasas oficiales de unidades por hora.
- Lista vigente de países restringidos de Binance (la página oficial está fechada en 2024-02-27).
- Acceso de Alpaca e Interactive Brokers para residentes en Colombia, y comisiones de OKX e IBKR.
- Plan de TradingView requerido para webhooks.

## 7. Evidencia de las técnicas de la ronda v2 (verificada el 2026-11-09)

| Técnica en el código | Evidencia | Lectura honesta |
|---|---|---|
| Señal primaria de tendencia (cruce EMA) y momentum | Liu y Tsyvinski (2021), *Review of Financial Studies* 34(6): momentum de serie temporal significativo en BTC, ETH y XRP a frecuencia diaria y semanal ([NBER w24877](https://nber.org/papers/w24877)) | Respalda usar la tendencia como dirección; a 1 h la evidencia es más débil |
| Meta-etiquetado | López de Prado (2018), *Advances in Financial Machine Learning*; Joubert (2022), "Meta-Labeling: Theory and Framework", *Journal of Financial Data Science* 4(3) ([Hudson & Thames](https://hudsonthames.org/?p=3454)) | Marco establecido; su valor depende de que la señal primaria tenga ventaja |
| Volatility targeting | Moreira y Muir (2017), *Journal of Finance* 72(4): mayor Sharpe al escalar por la varianza reciente ([NBER w22208](https://www.nber.org/papers/w22208)) | Hay réplicas en contra: fuera de muestra (Cederburg et al.) y con costes (Barroso y Detzel), citadas en DeMiguel et al. (2024, *JF*). Por eso se cuantiza el peso y todo se mide neto de costes |
| CNN sobre imágenes de velas | Jiang, Kelly y Xiu (2023), *Journal of Finance* 78(6) ([PDF Yale](https://economics.yale.edu/sites/default/files/2023-11/The%20Journal%20of%20Finance%20-%202023%20-%20JIANG%20-%20Re%25E2%2580%2590%20Imag%20in%20ing%20Price%20Trends_0.pdf)) | Una réplica independiente en 20 acciones de EE. UU. obtuvo un AUC de 0.517 ([BSIC](https://bsic.it/technical-analysis-on-steroids/)). Expectativa baja: se prueba como hipótesis falsable |
| Funding y prima de perpetuos | He, Manela, Ross y von Wachter, "Fundamentals of Perpetual Futures" ([arXiv 2212.06888](https://arxiv.org/pdf/2212.06888v5)): explica la desviación perpetuo-spot | La capacidad predictiva para retornos solo tiene evidencia de practicantes y resultados mixtos ([Brave New Coin](https://bravenewcoin.com/insights/testing-theory-is-perpetual-swap-funding-rate-predictive)); se trata como hipótesis |
| Deflated Sharpe y PSR | Bailey y López de Prado (2012, 2014) | Corrige la selección entre muchos intentos; aquí se cuentan todos los intentos acumulados sobre el periodo de desarrollo |

Conclusión previa al entrenamiento: la evidencia más sólida respalda la tendencia o el momentum y una gestión prudente del tamaño. Las fuentes alternativas y la CNN son hipótesis con expectativas modestas. El diseño v2 está hecho para descartarlas si no aportan.

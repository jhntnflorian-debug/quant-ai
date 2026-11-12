# Recursos para análisis y trading

Revisión de fuentes: **9 de noviembre de 2026**.

Esta selección reúne **33 recursos principales** para estudiar mercados, trabajar con velas, calcular indicadores, probar estrategias, medir riesgo y entrenar modelos. Incluye repositorios públicos, documentación y formación de bolsas, brokers y proyectos técnicos. Para comenzar, propongo **yfinance + mplfinance + ta + backtesting.py + QuantStats**: cubren datos, gráficos, indicadores, simulación e informes. Para cripto, conviene explorar CCXT y Freqtrade; para sistemas más complejos, LEAN o NautilusTrader. Qlib y FinRL corresponden a una etapa posterior de investigación con inteligencia artificial.

La selección se basa en documentación oficial consultada; **no se instalaron ni auditaron los proyectos**, y no se comprobó la rentabilidad de sus estrategias. Los niveles y combinaciones sugeridos son una valoración práctica según sus funciones. La fecha indicada es de consulta, no de actualización de cada recurso.

**Repositorios para obtener datos, probar ideas y operar sistemas**

Estos proyectos cumplen funciones distintas. Un conector entrega datos o envía órdenes; un motor simula reglas; una biblioteca de métricas interpreta resultados. Elegirlos por esa función evita instalar herramientas redundantes.

| Repositorio | Qué permite hacer | Encaje y límite principal |
|---|---|---|
| [yfinance](https://github.com/ranaroussi/yfinance) | Descargar precios e información financiera de Yahoo Finance desde Python. | Útil para aprender con datos históricos. No está afiliado a Yahoo; revisar sus condiciones de datos, especialmente para redistribuirlos. |
| [OpenBB](https://github.com/openbq-org/OpenBB) | Integrar fuentes públicas, licenciadas y propias mediante Python y API REST. | Investigación y tableros con diversas fuentes. La plataforma abierta y OpenBB Workspace son productos diferenciados; no todos los datos son gratuitos. |
| [CCXT](https://github.com/ccxt/ccxt) | Consultar OHLCV, libros de órdenes y conectar operaciones con exchanges. | Cripto; interfaces para Python y JavaScript/TypeScript, entre otras. Cada exchange tiene capacidades y parámetros propios. |
| [backtesting.py](https://github.com/kernc/backtesting.py) | Programar reglas, simular comisiones y revisar gráficos y estadísticas. | Buena entrada al backtesting en Python. Su licencia es AGPL-3.0. |
| [vectorbt](https://github.com/polakowo/vectorbt) | Explorar señales, activos y combinaciones de parámetros; analizar operaciones. | Investigación vectorizada en Python. Distinguir la versión pública de PRO y revisar Commons Clause en su licencia. |
| [LEAN](https://github.com/QuantConnect/Lean) | Investigar, simular, optimizar y ejecutar estrategias con C# y Python. | Motor amplio para proyectos algorítmicos. La CLI local usa Docker; datos y servicios asociados tienen condiciones propias. |
| [NautilusTrader](https://github.com/nautechsystems/nautilus_trader) | Simulación y ejecución por eventos con núcleo Rust e interfaz Python. | Para arquitectura avanzada. Reutilizar lógica entre simulación y vivo no elimina diferencias de transporte, tiempos o reconciliación. |
| [Freqtrade](https://github.com/freqtrade/freqtrade) | Backtesting, optimización, dry-run, interfaz web y modelos mediante FreqAI. | Bots de cripto en Python. Comenzar con simulación y comprobar compatibilidad del exchange. |
| [Hummingbot](https://github.com/hummingbot/hummingbot) | Desarrollar bots de market making y arbitraje con scripts y controladores. | Microestructura y ejecución cripto. Revisar el conector concreto y sus funciones. |
| [QuantStats](https://github.com/ranaroussi/quantstats) | Generar estadísticas de rendimiento, riesgo, drawdown e informes HTML. | Evaluación de resultados en Python; no reemplaza el motor de operaciones. |
| [PyPortfolioOpt](https://github.com/PyPortfolio/PyPortfolioOpt) | Optimización media-varianza, Black-Litterman, covarianza y carteras jerárquicas. | Asignación entre activos; responde cuánto asignar a cada uno. |
| [Riskfolio-Lib](https://github.com/dcajasn/Riskfolio-Lib) | Optimizar carteras con CVaR, risk parity, drawdown y métodos jerárquicos. | Para comparar distintas definiciones de riesgo y objetivos de cartera. |
| [Backtrader](https://github.com/mementum/backtrader) | Backtesting por eventos, varias temporalidades, comisiones, slippage y órdenes. | Amplio material en Python. El README conserva integraciones deprecadas o antiguas: revisar dependencias antes de adoptar ejemplos. |

No usé estrellas de GitHub como evidencia de calidad financiera. Tampoco equiparé repositorio público con servicio gratuito: el código, la licencia de datos y la infraestructura son aspectos separados. En particular, vectorbt incorpora **Apache-2.0 con Commons Clause**, y la licencia de yfinance no concede automáticamente derechos sobre los datos de Yahoo. ([Licencia de vectorbt](https://github.com/polakowo/vectorbt/blob/master/LICENSE.md), [documentación de yfinance](https://github.com/ranaroussi/yfinance)).

**Herramientas para velas, indicadores y gráficos**

Conviene elegir una biblioteca de indicadores y otra de visualización. Para notebooks y exportación, mplfinance; para exploración interactiva, Plotly; para desarrollar una interfaz web, Lightweight Charts. Son propuestas de encaje técnico, no una comparación de rendimiento medida.

| Recurso | Uso práctico | Qué tener presente |
|---|---|---|
| [TA-Lib para Python](https://github.com/TA-Lib/ta-lib-python) | Indicadores y reconocimiento automático de patrones de velas. | Wrapper de biblioteca C; revisar compatibilidad de versiones e instalación. |
| [ta](https://github.com/bukosabino/ta) | Indicadores de tendencia, momentum, volatilidad y volumen sobre Pandas/NumPy. | Adecuado para construir variables de análisis; necesita datos de entrada. |
| [pandas-ta-classic](https://github.com/xgboosted/pandas-ta-classic) | Extender DataFrames con indicadores y patrones. | Proyecto y paquete propios; no descarga precios. |
| [mplfinance](https://github.com/matplotlib/mplfinance) | Velas, OHLC, Renko, Point & Figure, volumen y subgráficos. | Recibe datos con índice temporal; útil para notebooks y gráficos exportables. |
| [Plotly.py](https://github.com/plotly/plotly.py) | Gráficos interactivos, velas y paneles de exploración en Python. | Las reglas e indicadores se calculan aparte; tiene [ejemplos oficiales de velas](https://plotly.com/python/candlestick-charts/). |
| [TradingView Lightweight Charts](https://github.com/tradingview/lightweight-charts) | Crear gráficos financieros web con JavaScript/TypeScript. | No equivale a la plataforma completa de TradingView; revisar atribución y enlace exigidos por el proyecto. |
| [lightweight-charts-python](https://github.com/louisnw01/lightweight-charts-python) | Usar gráficos interactivos, paneles y herramientas de dibujo desde Python. | Wrapper comunitario distinto del proyecto oficial JavaScript. |
| [finplot](https://github.com/highfestiva/finplot) | Gráficos de escritorio, volumen por precio y mapas de calor. | El README indica que no es una aplicación web y no funciona con Jupyter Labs. |
| [Manual de estrategias Pine Script](https://www.tradingview.com/pine-script-docs/concepts/strategies/) | Aprender reglas, órdenes, emulador, comisiones y evaluación en TradingView. | Documentación y ejemplos, no repositorio Python. |

**Técnicas que vale la pena estudiar y convertir en experimentos**

Una vela resume apertura, máximo, mínimo y cierre de un intervalo. El cuerpo representa apertura y cierre; las mechas muestran los extremos. Cambiar la temporalidad cambia esa agregación: una vela diaria no conserva toda la secuencia intradía. El primer ejercicio útil es graficar OHLCV —precios y volumen— y comprobar sesiones, zona horaria, faltantes y ajustes. ([Gráficos de velas de Plotly](https://plotly.com/python/candlestick-charts/)).

| Familia | Conceptos y herramientas | Ejercicio propuesto |
|---|---|---|
| Tendencia y estructura | Máximos/mínimos, soportes, resistencias, SMA/EMA, MACD y ADX. | Formular una regla de tendencia y medirla en periodos diferentes. |
| Momentum | RSI y otros osciladores. | Examinar resultados por régimen; no interpretar un umbral como orden automática. |
| Volatilidad | ATR, bandas de Bollinger y dispersión de retornos. | Comparar riesgo y tamaño de posición en entornos tranquilos y volátiles. |
| Volumen | OBV y VWAP. | Estudiar cómo cambia una señal cuando se acompaña de mayor o menor actividad. |
| Patrones de velas | Formaciones reconocibles y etiquetas automáticas. | Medir frecuencia y resultado posterior según activo, tendencia y volatilidad. |
| Liquidez y microestructura | Spread bid-ask, profundidad y costo según tamaño de orden. | Comparar el precio observado con el precio al que sería ejecutable la operación. |

La documentación de **ta** permite revisar entradas, ventanas y cálculos de las principales familias de indicadores; mplfinance aporta ejemplos de líneas de tendencia y soportes. TA-Lib ofrece detección de patrones, pero su presencia en una biblioteca no demuestra capacidad predictiva. Para liquidez, CME distingue spread, profundidad y costo de negociar: son dimensiones que las velas por sí solas no reconstruyen. ([API de ta](https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html), [mplfinance](https://github.com/matplotlib/mplfinance), [TA-Lib](https://github.com/TA-Lib/ta-lib-python), [guía de liquidez CME](https://www.cmegroup.com/education/demos-and-tutorials/cme-liquidity-tool-user-guide)).

Tendencia, rupturas y reversión a la media pueden plantearse como hipótesis de investigación. Cada hipótesis debe definir entradas, salidas, tamaño de posición y costos. Esta búsqueda no encontró evidencia que permita declarar rentables de forma universal patrones concretos o escuelas como Wyckoff, Elliott o SMC.

**Métricas para evaluar una estrategia**

| Métrica | Qué responde | Precaución al interpretarla |
|---|---|---|
| Retorno neto y benchmark | ¿Qué produjo después de costos y frente a una alternativa? | Comparar el mismo periodo y documentar exposición. |
| Máximo drawdown | ¿Cuál fue la mayor caída desde un máximo del capital? | Revisar también duración y recuperación; aclarar cálculo intrabar o al cierre. |
| Sharpe | ¿Qué exceso de retorno obtuvo respecto a su volatilidad? | Alinear periodicidad, anualización y referencia libre de riesgo. |
| Sortino | ¿Cómo se compara el retorno con la desviación a la baja? | Documentar el objetivo de retorno usado. |
| Profit factor | ¿Cuánto ganó en operaciones cerradas por cada unidad perdida? | Depende de la muestra y del tratamiento de costos. |
| Expectativa por operación | ¿Cuál fue el resultado neto promedio por operación cerrada? | Examinar también distribución y tamaño de pérdidas extremas. |
| Tasa de acierto | ¿Qué proporción de operaciones terminó ganando? | Necesita acompañarse de ganancia y pérdida promedio. |
| Operaciones y estabilidad | ¿Cuánta evidencia existe y cómo cambia entre periodos? | Un agregado favorable puede esconder deterioro o concentración. |

Estas definiciones y distinciones se apoyan en el [reporte de estrategias](https://www.tradingview.com/support/solutions/43000764138-tradingview-strategy-report-how-to-start/), y las explicaciones oficiales de [Sharpe](https://www.tradingview.com/support/solutions/43000681694-sharpe-ratio/), [Sortino](https://www.tradingview.com/support/solutions/43000681697-sortino-ratio/), [profit factor](https://www.tradingview.com/support/solutions/43000681698-profit-factor/) y [expectativa](https://www.tradingview.com/support/solutions/43000772770-expected-payoff/).

Un ejemplo didáctico muestra por qué acertar mucho no basta: ganar 80% de las veces una unidad y perder 20% de las veces cinco unidades produce **0,8 × 1 − 0,2 × 5 = −0,2 unidades por operación**, antes de costos. No existe un umbral único de una métrica que certifique una estrategia para operar con dinero real.

**Formación y entrenamiento de modelos**

Aquí “entrenamiento” tiene dos sentidos: aprender a analizar mercados y entrenar algoritmos. La progresión propuesta comienza por instrumentos, datos y simulación; después incorpora aprendizaje automático. No se verificó disponibilidad en español ni gratuidad de todos los certificados, datos o servicios asociados.

| Recurso | Contenido | Nivel sugerido |
|---|---|---|
| [IBKR: Introduction to Technical Analysis](https://www.interactivebrokers.com/campus/trading-course/introduction-to-technical-analysis/) | Gráficos, medias, bandas, indicadores y volatilidad. | Inicial; sin programación. |
| [IBKR: Trading and Analysis](https://www.interactivebrokers.com/campus/trading-course/trading-and-analysis/) | Liquidez, participantes, fundamentos, tendencia y soportes. | Inicial. |
| [CME: Introduction to Futures](https://www.cmegroup.com/education/courses/introduction-to-futures) | Contratos, vencimientos, ticks, nominal, margen y liquidación. | Inicial, especialmente para futuros. |
| [CME: Technical Analysis](https://www.cmegroup.com/education/courses/technical-analysis) | Patrones, tendencias, soportes, resistencias y osciladores. | Inicial-intermedio. |
| [QuantConnect Boot Camp](https://www.quantconnect.com/bootcamp/) | Lecciones guiadas de trading algorítmico y su API. | Intermedio; Python o C#. |
| [QuantConnect Research Guide](https://www.quantconnect.com/docs/v2/writing-algorithms/key-concepts/research-guide) | Investigación, sobreajuste y sesgos de datos. | Intermedio; útil con cualquier motor. |
| [Machine Learning for Trading](https://github.com/stefan-jansen/machine-learning-for-trading) | Código sobre datos, microestructura, modelos y backtesting; el README consultado corresponde a la tercera edición. | Intermedio-avanzado; Python, estadística y ML. |
| [Microsoft Qlib](https://github.com/microsoft/qlib) | Plataforma de investigación cuantitativa con modelos y flujos de datos. | Avanzado; el README advierte indisponibilidad temporal del dataset oficial. |
| [FinRL](https://github.com/AI4Finance-Foundation/FinRL) | Aprendizaje por refuerzo financiero. | Avanzado y experimental; el proyecto conserva foco educativo e investigador. |
| [FinRL-Tutorials](https://github.com/AI4Finance-Foundation/FinRL-Tutorials) | Ejemplos introductorios, avanzados y de optimización. | Requiere Python y fundamentos de aprendizaje por refuerzo. |
| [scikit-learn: TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) | Particiones cronológicas con ventanas de prueba y separación temporal. | Intermedio; metodología de evaluación, no estrategia. |

Para modelos, comenzaría con una referencia simple y después regresión, clasificación o árboles. Cada aumento de complejidad debería justificarse fuera de muestra y después de costos. FinRL resulta útil para investigar estados, acciones y recompensas, pero el repositorio consultado remite el desarrollo de producción a otros proyectos; no debe interpretarse como un bot listo y validado para operar. ([FinRL](https://github.com/AI4Finance-Foundation/FinRL)).

**Ruta práctica y combinaciones iniciales**

| Etapa | Trabajo concreto | Entregable |
|---|---|---|
| 1. Instrumentos | Estudiar órdenes, sesiones, spread, margen y vencimientos pertinentes. | Ficha del mercado y sus costos. |
| 2. Datos | Revisar OHLCV, horarios, faltantes y ajustes; graficar retornos y volumen. | Notebook de exploración. |
| 3. Reglas | Formular una hipótesis con entradas, salidas y tamaño explícitos. | Estrategia sencilla y benchmark. |
| 4. Simulación | Incluir comisiones, spread, deslizamiento y financiación cuando corresponda. | Operaciones y resultados netos. |
| 5. Validación | Separar cronológicamente entrenamiento, validación y prueba final; aplicar walk-forward. | Resultados por ventanas y registro de experimentos. |
| 6. Seguimiento | Hacer paper trading y comparar ejecuciones con los supuestos históricos. | Diario de desviaciones y límites del modelo. |

Los transformadores y escaladores deben ajustarse únicamente con entrenamiento para evitar fuga de información. La evaluación walk-forward usa historia previa para reajustar y evaluar después. Si las etiquetas se solapan temporalmente, hay que diseñar una separación o purga adecuada: usar TimeSeriesSplit no resuelve automáticamente todos los casos financieros. También deben controlarse sesgos de supervivencia, selección de periodos y pruebas múltiples. ([Errores comunes de scikit-learn](https://scikit-learn.org/stable/common_pitfalls.html), [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html), [walk-forward de QuantConnect](https://www.quantconnect.com/docs/v2/writing-algorithms/optimization/walk-forward-optimization), [Research Guide](https://www.quantconnect.com/docs/v2/writing-algorithms/key-concepts/research-guide)).

| Objetivo | Combinación propuesta | Primer proyecto |
|---|---|---|
| Aprender con acciones o ETF | yfinance + mplfinance + ta + backtesting.py + QuantStats. | Graficar un activo, probar una regla simple y compararla con comprar y mantener. |
| Investigar cripto | CCXT + Freqtrade en dry-run; Plotly para explorar resultados. | Evaluar una estrategia con costos del exchange y seguimiento simulado. |
| Desarrollar investigación avanzada | LEAN o NautilusTrader, datos adecuados al instrumento y QuantStats; Qlib después. | Crear un experimento reproducible por eventos, con prueba temporal reservada. |

Estas combinaciones son propuestas de arquitectura; su integración y compatibilidad no se probaron. El primer objetivo útil es obtener un experimento que otra persona pueda reproducir: datos identificados, reglas explícitas, costos documentados y evaluación reservada. Esa base permite decidir con evidencia qué gráficos, modelos o herramientas adicionales aportan valor.

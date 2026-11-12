# Dashboard QUANT-AI

Panel de solo lectura sobre los artefactos reales del proyecto (modo PAPER). No genera datos de ejemplo.

```powershell
.venv\Scripts\python.exe -m streamlit run dashboard/app.py --server.address localhost
```

`--server.address localhost` evita que el panel quede expuesto en la red (por defecto Streamlit escucha en todas las interfaces y no tiene autenticación).

Páginas: Estado del sistema, Entrenamiento, Decisiones, Rendimiento, Datos, Arquitectura.
Las rutas salen de `quant_ai.config.load_config()` (respeta `QUANT_AI_HOME`). Si falta un artefacto, la página
muestra el comando que lo genera.

Pruebas: `.venv\Scripts\python.exe -m pytest tests/test_dashboard.py -q`

"""Punto de entrada del dashboard QUANT-AI (solo lectura de artefactos). Ejecutar:

    streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dash_common import FOOTER  # noqa: E402

st.set_page_config(page_title="QUANT-AI · Paper trading", page_icon=":material/monitoring:", layout="wide")

PAGES = Path(__file__).resolve().parent / "app_pages"
pg = st.navigation([
    st.Page(PAGES / "principal.py", title="Estado del sistema", icon=":material/dashboard:", default=True),
    st.Page(PAGES / "entrenamiento.py", title="Entrenamiento", icon=":material/model_training:"),
    st.Page(PAGES / "decisiones.py", title="Decisiones", icon=":material/rule:"),
    st.Page(PAGES / "rendimiento.py", title="Rendimiento", icon=":material/show_chart:"),
    st.Page(PAGES / "datos.py", title="Datos", icon=":material/database:"),
    st.Page(PAGES / "arquitectura.py", title="Arquitectura", icon=":material/account_tree:"),
])
pg.run()  # las páginas que se detienen con C.stop_page() dibujan su propio pie

st.divider()
st.caption(FOOTER)

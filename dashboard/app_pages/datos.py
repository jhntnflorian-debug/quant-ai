"""Datos: calidad por símbolo, huecos y procedencia (hashes) desde data/manifest.json."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dash_common as C  # noqa: E402

P = C.paths()
st.title("Datos")

manifest = C.read_json(str(P["data"] / "manifest.json"))
if not manifest or not manifest.get("latest"):
    C.missing("manifiesto de datos (data/manifest.json)", C.CMD_DATA)
else:
    latest = manifest["latest"]
    st.subheader("Calidad por símbolo")
    rows = []
    for s, m in latest.items():
        q = m.get("quality") or {}
        rows.append({"símbolo": s, "filas": m.get("rows"), "inicio": q.get("start"), "fin": q.get("end"),
                     "huecos": q.get("gap_count"), "barras faltantes": q.get("missing_bars"),
                     "hueco más largo (barras)": q.get("largest_gap_bars"),
                     "duplicados eliminados": q.get("duplicates_removed"),
                     "barras fuera de rejilla": q.get("off_grid_bars"),
                     "barras con volumen 0": q.get("zero_volume_bars"), "actualizado": m.get("updated_at")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.subheader("Huecos")
    for s, m in latest.items():
        gaps = (m.get("quality") or {}).get("gaps") or []
        with st.expander(f"{s}: {len(gaps)} huecos registrados", expanded=False):
            if gaps:
                st.dataframe(pd.DataFrame(gaps), hide_index=True, width="stretch")
            else:
                st.caption("Sin huecos registrados en el manifiesto.")

    st.subheader("Procedencia e integridad")
    src = manifest.get("source") or {}
    if src:
        st.write("**Fuente:** " + str(src.get("provider", "—")))
        if src.get("license_note"):
            st.caption(str(src["license_note"]))
    prov = []
    for s, m in latest.items():
        f = P["data"] / "clean" / f"{s}_1h.parquet"
        real = C.sha256_file(str(f))
        exp = m.get("clean_sha256")
        estado = "archivo ausente" if real is None else ("coincide" if real == exp else "NO coincide")
        prov.append({"símbolo": s, "sha256 del manifiesto": exp, "sha256 del archivo en disco": real,
                     "verificación": estado, "filas añadidas por REST": m.get("rest_rows_added")})
    pdf = pd.DataFrame(prov)
    st.dataframe(pdf, hide_index=True, width="stretch")
    if (pdf["verificación"] == "NO coincide").any():
        st.error("Algún archivo limpio ha cambiado desde que se escribió el manifiesto.")
    with st.expander("Archivos nuevos incorporados en la última actualización"):
        for s, m in latest.items():
            nf = m.get("new_files") or []
            st.write(f"**{s}**")
            if nf:
                st.dataframe(pd.DataFrame(nf), hide_index=True, width="stretch")
            else:
                st.caption("Ninguno.")
    hist = manifest.get("history") or []
    with st.expander(f"Historial de versiones del manifiesto ({len(hist)})"):
        flat = [{"símbolo": s, **v} for h in hist for s, v in h.items() if isinstance(v, dict)]
        if flat:
            st.dataframe(pd.DataFrame(flat), hide_index=True, width="stretch")
        else:
            st.caption("Sin historial.")

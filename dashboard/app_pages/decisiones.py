"""Decisiones: inspección de cada decisión del motor paper junto al precio."""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dash_common as C  # noqa: E402

P = C.paths()
st.title("Decisiones")
st.info(C.ASSOC_NOTE, icon=":material/info:")

decisions = C.read_jsonl(str(P["paper"] / "decisions.jsonl"))
if decisions is None:
    C.missing("decisiones (state/paper/decisions.jsonl)", C.CMD_PAPER,
              note="Las decisiones las escribe el motor paper en cada ciclo.")
    C.stop_page()
if not decisions:
    st.info("decisions.jsonl existe pero no contiene decisiones.")
    C.stop_page()

decisions = sorted(decisions, key=lambda d: str(d.get("bar_time")), reverse=True)
c1, c2 = st.columns([1, 3])
syms = sorted({d.get("symbol", "?") for d in decisions})
sym_f = c1.segmented_control("Activo", ["Todos", *syms], default="Todos", key="dec_sym") or "Todos"
acts = sorted({d.get("action", "?") for d in decisions})
act_f = c1.pills("Acción", acts, selection_mode="multi", default=acts, key="dec_act")
pool = [d for d in decisions if (sym_f == "Todos" or d.get("symbol") == sym_f) and d.get("action") in (act_f or acts)]
if not pool:
    st.info("Ningún resultado con esos filtros.")
    C.stop_page()
labels = {d.get("decision_id", str(i)): f"{str(d.get('bar_time'))[:16]} · {d.get('symbol')} · {d.get('action')}"
          for i, d in enumerate(pool)}
pick = c2.selectbox("Decisión", list(labels), format_func=labels.get, key="dec_pick")
d = next(x for i, x in enumerate(pool) if x.get("decision_id", str(i)) == pick)

# ---- resumen
a = st.columns(5)
a[0].metric("Acción", str(d.get("action")))
a[1].metric("Dirección", str(d.get("direction", "—")))
a[2].metric("Peso objetivo", C.pct(d.get("target_weight")), help=f"Peso actual: {C.pct(d.get('current_weight'))}")
a[3].metric("Precio de referencia", C.num(d.get("ref_price")))
a[4].metric("Ejecutada", "sí" if d.get("executed") else "no")

# ---- velas
sym = d.get("symbol")
bt = C.ts(d.get("bar_time"))
px = C.load_ohlcv(P["data"], sym) if sym else None
horizon = int(d.get("horizon_bars") or 0)
if px is None or bt is None:
    C.missing(f"velas de {sym} (data/clean)", C.CMD_DATA)
else:
    px = px.copy()
    px["ema24"] = px["close"].ewm(span=24, adjust=False).mean()
    px["ema168"] = px["close"].ewm(span=168, adjust=False).mean()
    pos = px.index.searchsorted(bt)
    if pos >= len(px):
        st.warning("El bar_time de la decisión es posterior a la última vela disponible en data/clean.")
        pos = len(px) - 1
    lo, hi = max(pos - 120, 0), min(pos + max(horizon, 24) + 24, len(px))
    w = px.iloc[lo:hi]
    fig = go.Figure(go.Candlestick(x=w.index, open=w["open"], high=w["high"], low=w["low"], close=w["close"],
                                   name="OHLC", increasing_line_color="#2fa37a", decreasing_line_color="#d6457b"))
    fig.add_scatter(x=w.index, y=w["ema24"], name="EMA24", mode="lines", line=dict(color=C.CAT[0], width=2))
    fig.add_scatter(x=w.index, y=w["ema168"], name="EMA168", mode="lines", line=dict(color=C.CAT[1], width=2))
    fig.add_vline(x=bt, line_dash="dash", line_color="gray")
    if horizon and pos + horizon < len(px):
        fig.add_vline(x=px.index[pos + horizon], line_dash="dot", line_color="gray")
    if d.get("ref_price"):
        fig.add_scatter(x=[bt], y=[d["ref_price"]], mode="markers", name=f"{d.get('action')} (ref.)",
                        marker=dict(size=12, color="#111", symbol="diamond", line=dict(color="white", width=2)))
    if d.get("stop_loss"):
        fig.add_hline(y=d["stop_loss"], line_dash="dot", line_color="#d6457b", annotation_text="stop")
    fig.update_layout(xaxis_rangeslider_visible=False)
    C.show(C.style_fig(fig, 460, f"{sym} 1h alrededor de la decisión"),
           "Velas 1h de data/clean con EMA24 y EMA168. Línea discontinua: barra de la decisión; "
           "punteada: horizonte.", key="dec_candles")

    st.write("**Resultado posterior**")
    if horizon and pos + horizon < len(px) and d.get("ref_price"):
        fut = float(px["close"].iloc[pos + horizon])
        r = fut / d["ref_price"] - 1
        o = st.columns(3)
        o[0].metric(f"Cierre a {horizon} barras", C.num(fut))
        o[1].metric("Retorno desde el precio de referencia", C.pct(r))
        o[2].metric("Fecha del cierre", f"{px.index[pos + horizon]:%Y-%m-%d %H:%M}")
        st.caption("Retorno bruto del activo (sin costes) entre ref_price y el cierre de la barra a horizon_bars. "
                   "No es el PnL de la estrategia.")
    else:
        st.info("Aún no hay datos suficientes en data/clean para medir el resultado a horizon_bars "
                f"(horizonte: {horizon or 'no definido'}).")

# ---- probabilidades y modelo
st.subheader("Señal del modelo")
l, r = st.columns(2)
probs = d.get("probabilities") or {}
with l:
    if probs:
        fig = go.Figure(go.Bar(x=list(probs), y=[probs[k] for k in probs], marker_color=C.CAT[0],
                               text=[f"{v:.3f}" for v in probs.values()], textposition="outside"))
        fig.update_yaxes(range=[0, 1], title="probabilidad")
        fig.update_layout(hovermode="closest")
        C.show(C.style_fig(fig, 300, "Probabilidades"), "Probabilidad estimada de cada modelo.", key="dec_probs")
    else:
        st.caption("Sin probabilidades (decisión por reglas).")
    k = st.columns(3)
    k[0].metric("Incertidumbre", C.num(d.get("uncertainty"), 3))
    k[1].metric("Retorno esperado", C.pct(d.get("exp_ret")))
    k[2].metric("Volatilidad esperada", C.pct(d.get("exp_vol")))
    reg = d.get("regime") or {}
    st.caption(f"Régimen: tendencia={reg.get('trend', '—')}, volatilidad={reg.get('vol', '—')} · "
               f"modelo: {d.get('model_version', '—')}")
with r:
    tf = d.get("top_features") or []
    if tf:
        df = pd.DataFrame(tf)
        fig = go.Figure(go.Bar(y=df["feature"], x=df["z"], orientation="h", marker_color=C.CAT[2],
                               customdata=df["value"], hovertemplate="%{y}<br>z=%{x:.2f}<br>valor=%{customdata}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(hovermode="closest")
        C.show(C.style_fig(fig, 300, "Características más influyentes (z)"),
               "Desviación z del valor de cada característica en esta barra. " + C.ASSOC_NOTE, key="dec_feats")
        st.dataframe(df, hide_index=True, width="stretch")
    else:
        st.caption("Sin top_features en esta decisión.")

# ---- riesgo
st.subheader("Motor de riesgo")
risk = d.get("risk") or {}
if risk:
    (st.success if risk.get("approved") else st.error)("Aprobada" if risk.get("approved") else "Rechazada / reducida")
    for x in risk.get("reasons", []) or []:
        st.write(f"- {x}")
    if risk.get("checks"):
        st.dataframe(pd.DataFrame([{"chequeo": k, "resultado": str(v)} for k, v in risk["checks"].items()]),
                     hide_index=True, width="stretch")
else:
    st.caption("Sin información del motor de riesgo.")
if d.get("reasons"):
    st.write("**Motivos de la decisión**")
    for x in d["reasons"]:
        st.write(f"- {x}")
if d.get("fill"):
    st.write("**Ejecución simulada:** " + ", ".join(f"{k}={v}" for k, v in d["fill"].items()))
with st.expander("Decisión completa (JSON)"):
    st.json(d)

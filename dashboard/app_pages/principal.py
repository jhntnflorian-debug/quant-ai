"""Página principal: estado del sistema paper."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dash_common as C  # noqa: E402

P = C.paths()
st.title("QUANT-AI · Estado del sistema")
C.mode_banner()

portfolio = C.read_json(str(P["paper"] / "portfolio.json"))
heartbeat = C.read_json(str(P["state"] / "heartbeat.json"))
decisions = C.read_jsonl(str(P["paper"] / "decisions.jsonl"))
active = C.read_text(str(P["models"] / "registry" / "ACTIVE"))

# ---- salud: kill switch, heartbeat, antigüedad de datos
st.subheader("Salud del sistema")
c1, c2, c3 = st.columns(3)
with c1.container(border=True):
    if P["kill"].exists():
        st.error("KILL SWITCH ACTIVO: no se emiten órdenes.", icon=":material/emergency:")
    else:
        st.success("Kill switch inactivo", icon=":material/check_circle:")
    st.caption(f"Archivo vigilado: {P['kill']}")
with c2.container(border=True):
    if heartbeat is None:
        C.missing("heartbeat (state/heartbeat.json)", C.CMD_PAPER)
    else:
        status = heartbeat.get("status", "?")
        t = C.ts(heartbeat.get("last_cycle_at"))
        st.metric("Último ciclo", C.age_text(t), help=str(heartbeat.get("last_cycle_at")))
        (st.success if status == "ok" else st.error)(f"Estado: {status}")
        if heartbeat.get("message"):
            st.caption(str(heartbeat["message"]))
        if heartbeat.get("cycle_seconds") is not None:
            st.caption(f"Duración del ciclo: {C.num(heartbeat['cycle_seconds'], 1)} s")
with c3.container(border=True):
    syms = C.symbols_in_clean(P["data"])
    if not syms:
        C.missing("datos limpios (data/clean)", C.CMD_DATA)
    else:
        st.write("**Antigüedad de los datos**")
        for s in syms:
            df = C.load_ohlcv(P["data"], s, ("open_time", "close"))
            last = df.index[-1] if df is not None and len(df) else None
            st.write(f"{s}: última vela {last:%Y-%m-%d %H:%M} UTC ({C.age_text(last)})" if last is not None
                     else f"{s}: sin velas")
        if heartbeat and heartbeat.get("last_bar"):
            st.caption("Última barra procesada por el ciclo: " + ", ".join(
                f"{k}: {str(v)[:16]}" for k, v in heartbeat["last_bar"].items()))

# ---- modelo
st.subheader("Modelo en uso")
with st.container(border=True):
    version = (portfolio or {}).get("model_version") or active
    mstatus = (portfolio or {}).get("model_status")
    if not version:
        st.info("Sin modelo registrado: el sistema paper solo puede operar con reglas hasta que exista una "
                "versión en models/registry.", icon=":material/info:")
    else:
        meta = C.read_json(str(P["models"] / "registry" / str(version) / "meta.json"))
        a, b = st.columns(2)
        a.metric("Versión", str(version))
        b.metric("Estado", str(mstatus or (meta or {}).get("status") or "—"))
        if meta and meta.get("gate"):
            gate = meta["gate"]
            if gate.get("passed"):
                st.success("Puerta de validación: superada")
            else:
                st.warning("Puerta de validación: NO superada")
            for r in gate.get("reasons", []) or []:
                st.write(f"- {r}")
        elif meta is None:
            st.caption("No se encontró models/registry/<versión>/meta.json para esta versión.")

# ---- precios
st.subheader("Precios (último cierre en data/clean)")
syms = C.symbols_in_clean(P["data"])
if syms:
    cols = st.columns(len(syms))
    for col, s in zip(cols, syms):
        df = C.load_ohlcv(P["data"], s, ("open_time", "close"))
        if df is None or len(df) < 25:
            col.write(f"{s}: sin datos")
            continue
        last = float(df["close"].iloc[-1])
        d24 = last / float(df["close"].iloc[-25]) - 1
        col.metric(s, f"{last:,.2f}", f"{d24 * 100:+.2f}% (24 h)")
else:
    C.missing("precios", C.CMD_DATA)

# ---- paper
st.subheader("Paper trading")
if portfolio is None:
    C.missing("estado paper (state/paper/portfolio.json)", C.CMD_PAPER,
              note="El motor paper aún no ha corrido, así que no hay posiciones, balance ni señales.")
else:
    equity = portfolio.get("equity")
    init = portfolio.get("initial_cash")
    peak = portfolio.get("peak_equity")
    day0 = portfolio.get("day_start_equity")
    ret = equity / init - 1 if equity is not None and init else None
    dd = equity / peak - 1 if equity is not None and peak else None
    dday = equity / day0 - 1 if equity is not None and day0 else None
    positions = portfolio.get("positions") or {}
    mark = {}
    for s in positions:
        df = C.load_ohlcv(P["data"], s, ("open_time", "close"))
        if df is not None and len(df):
            mark[s] = float(df["close"].iloc[-1])
    gross = sum(p.get("qty", 0) * mark.get(s, p.get("avg_price", 0)) for s, p in positions.items())
    expo = gross / equity if equity else None

    m = st.columns(5)
    m[0].metric("Equity", C.num(equity))
    m[1].metric("Caja", C.num(portfolio.get("cash")))
    m[2].metric("Rentabilidad acumulada", C.pct(ret))
    m[3].metric("Drawdown vs. máximo", C.pct(dd))
    m[4].metric("Exposición bruta", C.pct(expo))
    with st.container(border=True):
        st.write("**Nivel de riesgo**")
        r = st.columns(4)
        r[0].metric("Resultado del día", C.pct(dday))
        r[1].metric("Pico de equity", C.num(peak))
        r[2].metric("Posiciones abiertas", len(positions))
        r[3].metric("Kill switch", "ACTIVO" if P["kill"].exists() else "inactivo")
        st.caption("Valores calculados desde portfolio.json; la exposición valora las posiciones con el último "
                   "cierre de data/clean.")

    st.write("**Posiciones simuladas**")
    if positions:
        rows = [{"símbolo": s, "cantidad": p.get("qty"), "precio medio": p.get("avg_price"),
                 "último cierre": mark.get(s), "stop": p.get("stop_price"), "abierta": p.get("opened_at")}
                for s, p in positions.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.caption("Sin posiciones abiertas (plano).")

# ---- decisiones / señales
st.subheader("Señales y decisiones")
if decisions is None:
    C.missing("decisiones (state/paper/decisions.jsonl)", C.CMD_PAPER)
elif not decisions:
    st.info("decisions.jsonl existe pero está vacío.")
else:
    df = pd.DataFrame(decisions).reindex(columns=[
        "bar_time", "symbol", "action", "target_weight", "ref_price", "executed", "risk", "probabilities"
    ]).sort_values("bar_time")
    last = df.groupby("symbol").tail(1)
    st.write("**Última señal por activo**")
    cols = st.columns(max(len(last), 1))
    for col, (_, d) in zip(cols, last.iterrows()):
        pr = d.get("probabilities") or {}
        col.metric(str(d["symbol"]), str(d.get("action")),
                   f"p_ensemble={C.num(pr.get('p_ensemble'), 3)} · {str(d['bar_time'])[:16]}", delta_color="off")
    st.write("**Últimas 10 decisiones**")
    tail = df.tail(10).iloc[::-1]
    view = pd.DataFrame({
        "barra": tail["bar_time"], "símbolo": tail["symbol"], "acción": tail["action"],
        "peso objetivo": tail.get("target_weight"), "precio ref.": tail.get("ref_price"),
        "ejecutada": tail.get("executed"),
        "riesgo aprobó": tail["risk"].map(lambda r: (r or {}).get("approved") if isinstance(r, dict) else None),
        "p_ensemble": tail["probabilities"].map(lambda r: (r or {}).get("p_ensemble") if isinstance(r, dict) else None),
    })
    st.dataframe(view, hide_index=True, width="stretch")

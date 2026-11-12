"""Rendimiento: métricas, equity, drawdown, operaciones, ML y comparación backtest vs paper."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dash_common as C  # noqa: E402

P = C.paths()
st.title("Rendimiento")
st.caption("Resultados de backtest y paper: simulados, con costes modelados. No son beneficios reales.")

runs = C.list_runs(P["reports"])
if not runs:
    C.missing("corridas de experimentos (reports/runs/<run_id>/)", C.CMD_EXPERIMENTS)
    C.stop_page()

run_id = st.selectbox("Corrida", runs, index=runs.index(C.default_run(P["reports"], runs)), key="perf_run")
rdir = P["reports"] / "runs" / run_id
summary = C.read_json(str(rdir / "summary.json"))
if summary is None:
    C.missing(f"summary.json de {run_id}", C.CMD_EXPERIMENTS)
    C.stop_page()
C.run_banner(summary)
per = summary.get("period") or {}
st.caption(f"Tipo: {summary.get('kind')} · periodo {str(per.get('start'))[:10]} → {str(per.get('end'))[:10]} · "
           f"símbolos: {', '.join(summary.get('symbols', []))} · creada {str(summary.get('created_at'))[:19]}")

strat = C.strategies_frame(summary)
order = list(strat["estrategia"]) if not strat.empty else []

# ---- métricas
st.header("Métricas por estrategia")
if strat.empty:
    st.info("La corrida no contiene estrategias.")
else:
    ratio = ["total_return", "cagr", "ann_volatility", "max_drawdown", "exposure_time", "avg_gross_exposure",
             "win_rate"]
    cfgcols = {c: st.column_config.NumberColumn(c, format="percent") for c in ratio if c in strat.columns}
    for c in strat.columns:
        if c not in cfgcols and c != "estrategia" and pd.api.types.is_float_dtype(strat[c]):
            cfgcols[c] = st.column_config.NumberColumn(c, format="%.3f")
    st.dataframe(strat.drop(columns=[c for c in ("start", "end") if c in strat.columns]), hide_index=True,
                 width="stretch", column_config=cfgcols)
    st.caption("Sharpe con IC95 por bootstrap (sharpe_ic95_*). Un IC que cruza 0 indica que no se distingue de ruido.")

# ---- equity y drawdown
st.header("Equity y drawdown")
eq = C.read_parquet(str(rdir / "equity.parquet"))
if eq is None or eq.empty:
    C.missing(f"equity.parquet de {run_id}", C.CMD_EXPERIMENTS)
else:
    eq = eq.copy()
    eq.index = pd.to_datetime(eq.index, utc=True)
    sel = st.multiselect("Estrategias", list(eq.columns), default=list(eq.columns)[:6], key="perf_eq_sel")
    logy = st.toggle("Escala logarítmica", value=False, key="perf_log")
    if sel:
        daily = eq[sel].resample("1D").last().dropna(how="all")
        dd = eq[sel].apply(C.drawdown).resample("1D").min().dropna(how="all")
        f1, f2 = go.Figure(), go.Figure()
        for s in sel:
            col = C.strategy_color(s, order)
            f1.add_scatter(x=daily.index, y=daily[s], name=s, mode="lines", line=dict(color=col, width=2))
            f2.add_scatter(x=dd.index, y=dd[s], name=s, mode="lines", line=dict(color=col, width=1.5))
        f1.update_yaxes(type="log" if logy else "linear", title="capital")
        f2.update_yaxes(tickformat=".0%", title="drawdown")
        C.show(C.style_fig(f1, 420, "Curvas de equity"),
               "Capital por estrategia, resolución diaria (último valor del día) para ligereza.", key="perf_eq")
        C.show(C.style_fig(f2, 300, "Drawdown"),
               "Caída desde el máximo previo, calculada barra a barra y mostrada como mínimo diario.", key="perf_dd")
    else:
        st.info("Elige al menos una estrategia.")

# ---- operaciones
st.header("Operaciones")
trades = C.read_parquet(str(rdir / "trades.parquet"))
if trades is None or trades.empty:
    C.missing(f"trades.parquet de {run_id}", note="Puede no existir si la estrategia no cerró operaciones.")
else:
    trades = trades.copy()
    for c in ("entry_time", "exit_time"):
        trades[c] = pd.to_datetime(trades[c], utc=True)
    strats_t = sorted(trades["strategy"].unique())
    f = st.columns(4)
    s_t = f[0].selectbox("Estrategia", strats_t, key="perf_tr_strat")
    syms_t = sorted(trades["symbol"].unique())
    sy = f[1].multiselect("Activo", syms_t, default=syms_t, key="perf_tr_sym")
    reasons = sorted(trades["exit_reason"].dropna().unique())
    rs = f[2].multiselect("Motivo de salida", reasons, default=reasons, key="perf_tr_reason")
    res = f[3].segmented_control("Resultado", ["Todas", "Ganadoras", "Perdedoras"], default="Todas",
                                 key="perf_tr_res") or "Todas"
    t = trades[(trades["strategy"] == s_t) & trades["symbol"].isin(sy) & trades["exit_reason"].isin(rs)]
    if res == "Ganadoras":
        t = t[t["pnl"] > 0]
    elif res == "Perdedoras":
        t = t[t["pnl"] <= 0]
    st.caption(f"{len(t):,} operaciones" + (" (se muestran las 5.000 más recientes)" if len(t) > 5000 else ""))
    st.dataframe(t.sort_values("exit_time", ascending=False).head(5000), hide_index=True, width="stretch",
                 column_config={"return": st.column_config.NumberColumn("return", format="percent")})

    if not t.empty:
        l, r = st.columns(2)
        with l:
            fig = go.Figure(go.Histogram(x=t["pnl"], nbinsx=60, marker_color=C.CAT[0]))
            fig.update_layout(hovermode="closest", bargap=0.05)
            fig.update_xaxes(title="PnL por operación")
            fig.add_vline(x=0, line_color="gray")
            C.show(C.style_fig(fig, 320, "Distribución de PnL"), "Histograma del PnL de las operaciones filtradas.",
                   key="perf_hist")
        with r:
            tm = t.assign(mes=t["exit_time"].dt.month, anio=t["exit_time"].dt.year)
            grid = tm.pivot_table(index="anio", columns="mes", values="pnl", aggfunc="sum")
            grid = grid.reindex(columns=range(1, 13))
            lim = float(np.nanmax(np.abs(grid.values))) if grid.notna().any().any() else 1.0
            fig = go.Figure(go.Heatmap(z=grid.values, x=[f"{m:02d}" for m in grid.columns],
                                       y=[str(a) for a in grid.index], zmid=0, zmin=-lim, zmax=lim,
                                       colorscale=[[0, "#c0392b"], [0.5, "#eeeeee"], [1, "#2a6fdb"]],
                                       colorbar=dict(title="PnL"), hoverongaps=False,
                                       hovertemplate="%{y}-%{x}: %{z:,.2f}<extra></extra>"))
            fig.update_layout(hovermode="closest")
            fig.update_yaxes(autorange="reversed", type="category")
            C.show(C.style_fig(fig, 320, "PnL por mes (salida) y año"),
                   "Suma de PnL de operaciones cerradas en cada mes, para la estrategia y activos filtrados.",
                   key="perf_heat")

        by_sym = t.groupby("symbol").agg(operaciones=("pnl", "size"), pnl_total=("pnl", "sum"),
                                         retorno_medio=("return", "mean"),
                                         tasa_acierto=("pnl", lambda s: (s > 0).mean()))
        st.write("**Resumen por activo (operaciones filtradas)**")
        st.dataframe(by_sym.reset_index(), hide_index=True, width="stretch",
                     column_config={"retorno_medio": st.column_config.NumberColumn(format="percent"),
                                    "tasa_acierto": st.column_config.NumberColumn(format="percent")})

# ---- por año y por régimen
st.header("Retorno por año y por régimen")
l, r = st.columns(2)
with l:
    by_year = C.read_csv(str(rdir / "by_year.csv"))
    if by_year is None or by_year.empty:
        C.missing(f"by_year.csv de {run_id}")
    else:
        pv = by_year.pivot(index="year", columns="strategy", values="return")
        fig = go.Figure()
        for s in [x for x in order if x in pv.columns]:
            fig.add_bar(x=pv.index.astype(str), y=pv[s], name=s, marker_color=C.strategy_color(s, order))
        fig.update_layout(barmode="group")
        fig.update_yaxes(tickformat=".0%")
        C.show(C.style_fig(fig, 340, "Retorno por año"), "Retorno anual de cada estrategia.", key="perf_year")
with r:
    by_reg = C.read_csv(str(rdir / "by_regime.csv"))
    if by_reg is None or by_reg.empty:
        C.missing(f"by_regime.csv de {run_id}")
    else:
        dim = st.segmented_control("Dimensión", sorted(by_reg["dimension"].unique()),
                                   default=sorted(by_reg["dimension"].unique())[0], key="perf_dim")
        d = by_reg[by_reg["dimension"] == (dim or by_reg["dimension"].iloc[0])]
        st.dataframe(d.pivot(index="strategy", columns="regime", values="sharpe").reset_index(),
                     hide_index=True, width="stretch")
        st.caption("Sharpe anualizado por régimen (tabla). Cada celda es una estrategia en ese régimen.")
        with st.expander("Detalle (barras, retorno anualizado, sharpe)"):
            st.dataframe(d, hide_index=True, width="stretch")

# ---- ML
st.header("Clasificación y calibración (ML)")
clf = summary.get("classification") or {}
if not clf:
    st.info("Esta corrida no incluye modelos ML (solo reglas).")
else:
    cdf = pd.DataFrame([{"modelo": k, **v} for k, v in clf.items()])
    st.dataframe(cdf, hide_index=True, width="stretch")
    st.caption("brier por debajo de brier_baseline indica mejor calibración que predecir la tasa base; "
               "AUC cercano a 0,5 es ruido.")
    cal = C.read_csv(str(rdir / "calibration.csv"))
    fi = C.read_csv(str(rdir / "feature_importance.csv"))
    l, r = st.columns(2)
    with l:
        if cal is None or cal.empty:
            C.missing(f"calibration.csv de {run_id}")
        else:
            sy = st.selectbox("Activo (calibración)", sorted(cal["symbol"].unique()), key="perf_cal_sym")
            fig = go.Figure()
            fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="calibración perfecta",
                            line=dict(color="gray", dash="dash"))
            for i, m in enumerate(sorted(cal["model"].unique())):
                d = cal[(cal["symbol"] == sy) & (cal["model"] == m)].sort_values("mean_pred")
                fig.add_scatter(x=d["mean_pred"], y=d["frac_pos"], mode="lines+markers", name=m,
                                line=dict(color=C.CAT[i % len(C.CAT)], width=2), marker=dict(size=8),
                                customdata=d["n"], hovertemplate="pred=%{x:.3f}<br>frac. positivos=%{y:.3f}"
                                "<br>n=%{customdata}<extra></extra>")
            fig.update_xaxes(title="probabilidad media predicha", range=[0, 1])
            fig.update_yaxes(title="fracción de positivos", range=[0, 1])
            fig.update_layout(hovermode="closest")
            C.show(C.style_fig(fig, 340, "Curva de calibración"),
                   "Cuanto más cerca de la diagonal, mejor calibrado el modelo.", key="perf_cal")
    with r:
        if fi is None or fi.empty:
            C.missing(f"feature_importance.csv de {run_id}")
        else:
            sy2 = st.selectbox("Activo (importancia)", sorted(fi["symbol"].unique()), key="perf_fi_sym")
            d = fi[fi["symbol"] == sy2].sort_values("lgbm_gain_pct", ascending=False).head(20)
            fig = go.Figure(go.Bar(y=d["feature"], x=d["lgbm_gain_pct"], orientation="h", marker_color=C.CAT[2]))
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(hovermode="closest")
            fig.update_xaxes(title="ganancia LightGBM (%)")
            C.show(C.style_fig(fig, 340, "Importancia de características (top 20)"),
                   C.ASSOC_NOTE, key="perf_fi")
            st.info(C.ASSOC_NOTE, icon=":material/info:")

# ---- backtest vs paper
st.header("Comparación backtest vs paper")
peq = C.read_csv(str(P["paper"] / "equity.csv"))
if peq is None or peq.empty:
    C.missing("equity paper (state/paper/equity.csv)", C.CMD_PAPER,
              note="Sin paper trading no hay nada que comparar con el backtest.")
elif eq is None or eq.empty:
    st.info("Sin equity de backtest para comparar.")
else:
    peq["time"] = pd.to_datetime(peq["time"], utc=True)
    peq = peq.sort_values("time").set_index("time")
    st.caption(f"Paper: {peq.index[0]:%Y-%m-%d %H:%M} → {peq.index[-1]:%Y-%m-%d %H:%M} UTC · "
               f"backtest: {eq.index[0]:%Y-%m-%d} → {eq.index[-1]:%Y-%m-%d}")
    s_cmp = st.selectbox("Estrategia de backtest", list(eq.columns), key="perf_cmp")
    bt_slice = eq[s_cmp].loc[peq.index[0]:peq.index[-1]].dropna()
    if len(bt_slice) < 2:
        st.info("El backtest no se solapa en el tiempo con el paper (el backtest termina antes de que "
                "empiece el paper), así que no se puede comparar rendimiento sobre el mismo periodo.")
    else:
        fig = go.Figure()
        fig.add_scatter(x=peq.index, y=peq["equity"] / peq["equity"].iloc[0], name="paper",
                        line=dict(color=C.CAT[0], width=2))
        fig.add_scatter(x=bt_slice.index, y=bt_slice / bt_slice.iloc[0], name=f"backtest · {s_cmp}",
                        line=dict(color=C.CAT[1], width=2))
        fig.update_yaxes(title="equity normalizado (inicio = 1)")
        C.show(C.style_fig(fig, 340, "Paper vs backtest en el periodo común"),
               "Ambas series normalizadas al inicio del periodo común.", key="perf_cmp_fig")

"""Entrenamiento: corridas DL, benchmarks GPU, folds walk-forward y registro de modelos."""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dash_common as C  # noqa: E402

P = C.paths()
st.title("Entrenamiento")

# ======================================================================= DL
st.header("Corridas de deep learning")
dl_root = P["models"] / "dl"
# Una corrida es cualquier carpeta con run_info.json (p. ej. secuencias_v1/gru), salvo las de cada fold.
runs = sorted(p.parent.relative_to(dl_root).as_posix() for p in dl_root.rglob("run_info.json")
              if "folds" not in p.relative_to(dl_root).parts) if dl_root.exists() else []
if not runs:
    C.missing("corridas DL (models/dl/<run>/)", C.CMD_TRAIN_DL,
              note="El entrenamiento DL se hace en Colab; copia models/dl/<run>/ al proyecto para verlo aquí.")
else:
    run = st.selectbox("Corrida DL", runs, key="dl_run")
    rdir = dl_root / run
    info = C.read_json(str(rdir / "run_info.json"))
    metrics = C.read_csv(str(rdir / "metrics.csv"))
    if info:
        a = st.columns(4)
        a[0].metric("Modelo", str(info.get("model", "—")))
        a[1].metric("Estado", str(info.get("status", "—")))
        a[2].metric("Parámetros", f"{info['n_params']:,}" if info.get("n_params") else "—")
        a[3].metric("Duración total", f"{C.num(info.get('total_seconds'), 0)} s")
        with st.expander("Entorno e hiperparámetros"):
            env = {k: v for k, v in info.items() if k != "hyperparams"}
            st.json({"entorno": env, "hyperparams": info.get("hyperparams")})
    else:
        C.missing(f"run_info.json de {run}")

    if metrics is None or metrics.empty:
        C.missing(f"metrics.csv de {run}")
    else:
        if "fold" not in metrics.columns:
            metrics["fold"] = 0
        folds = sorted(metrics["fold"].unique())
        sel = st.pills("Folds", folds, selection_mode="multi", default=folds, key="dl_folds")
        m = metrics[metrics["fold"].isin(sel or folds)]

        def curves(cols: list[str], title: str, ytitle: str, key: str):
            fig = go.Figure()
            for fi, f in enumerate(sorted(m["fold"].unique())):
                d = m[m["fold"] == f].sort_values("epoch")
                for ci, col in enumerate(cols):
                    if col in d.columns:
                        fig.add_scatter(x=d["epoch"], y=d[col], mode="lines+markers",
                                        name=f"fold {f} · {col}", line=dict(color=C.CAT[fi % len(C.CAT)],
                                        dash="solid" if ci == 0 else "dash", width=2), marker=dict(size=5))
            fig.update_xaxes(title="época")
            fig.update_yaxes(title=ytitle)
            C.show(C.style_fig(fig, 320, title), f"{title}. Línea continua: {cols[0]}"
                   + (f"; discontinua: {cols[1]}." if len(cols) > 1 else "."), key=key)

        l, r = st.columns(2)
        with l:
            curves(["train_loss", "val_loss"], "Pérdida entrenamiento / validación", "loss", "dl_loss")
            curves(["epoch_seconds"], "Tiempo por época", "segundos", "dl_time")
        with r:
            curves(["val_auc"], "AUC de validación", "AUC", "dl_auc")
            curves(["gpu_mem_alloc_gb", "gpu_mem_peak_gb"], "VRAM (asignada / pico)", "GB", "dl_vram")
        with st.expander("Tabla de métricas"):
            st.dataframe(metrics, hide_index=True, width="stretch")

    st.write("**Checkpoints disponibles**")
    cks = [{"archivo": p.relative_to(rdir).as_posix(), "MB": round(p.stat().st_size / 1e6, 2)}
           for p in sorted(rdir.rglob("*.pt"))]
    if cks:
        st.dataframe(pd.DataFrame(cks), hide_index=True, width="stretch")
    else:
        st.caption("Sin checkpoints .pt en esta copia (la exportación desde Colab los omite; quedan en Drive).")

    bt = C.read_json(str(rdir / "backtest_summary.json"))
    if bt:
        st.write("**Backtest de esta corrida (backtest_summary.json)**")
        rows = [{"estrategia": k, **{kk: vv for kk, vv in v.items() if not isinstance(vv, (list, dict))}}
                for k, v in (bt.get("strategies", bt) if isinstance(bt, dict) else {}).items()
                if isinstance(v, dict)]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.json(bt)
    else:
        st.caption("Sin backtest_summary.json para esta corrida.")

# ======================================================================= RL
st.header("Aprendizaje por refuerzo")
rl_root = P["models"] / "rl"
rl_runs = sorted(p.parent.name for p in rl_root.glob("*/eval_summary.json")) if rl_root.exists() else []
if not rl_runs:
    C.missing("corridas RL (models/rl/<run>/eval_summary.json)", f"{C.PY} scripts/train_rl.py",
              note="El RL se entrena en Colab; importa models/rl/<run>/ al proyecto para verlo aquí.")
else:
    rl_run = st.selectbox("Corrida RL", rl_runs, key="rl_run")
    s = C.read_json(str(rl_root / rl_run / "eval_summary.json")) or {}
    per = s.get("period", {})
    st.caption(f"{s.get('algo', '—').upper()} · {s.get('symbol', '—')} · evaluación "
               f"{str(per.get('start', '—'))[:10]} → {str(per.get('end_exclusive', '—'))[:10]} (exclusivo)")
    rows = []
    bh = s.get("buy_and_hold", {})
    bh = bh.get("metrics", bh) if isinstance(bh, dict) else {}
    if bh:
        rows.append({"política": "comprar y mantener", **{k: bh.get(k) for k in
                     ("total_return", "sharpe", "max_drawdown", "n_trades")}})
    for seed, r in (s.get("seeds") or {}).items():
        m = r.get("metrics", {})
        rows.append({"política": f"PPO semilla {seed}", **{k: m.get(k) for k in
                     ("total_return", "sharpe", "max_drawdown", "n_trades")},
                     "costes/capital": m.get("costs_pct_initial")})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if s.get("stability"):
        with st.expander("Estabilidad entre semillas"):
            st.json(s["stability"])

# ======================================================================= GPU
st.header("Benchmarks de GPU")
bench = sorted((P["reports"] / "gpu_benchmark").glob("*.json")) if (P["reports"] / "gpu_benchmark").exists() else []
if not bench:
    C.missing("benchmarks (reports/gpu_benchmark/*.json)", C.CMD_BENCH)
else:
    def flatten(d, prefix=""):
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                yield from flatten(v, key + ".")
            else:
                yield key, v

    for f in bench:
        data = C.read_json(str(f))
        with st.expander(f.name, expanded=len(bench) == 1):
            if isinstance(data, dict):
                kv = pd.DataFrame([(k, str(v)) for k, v in flatten(data)], columns=["clave", "valor"])
                st.dataframe(kv, hide_index=True, width="stretch")
            else:
                st.json(data)

# ======================================================================= folds ML
st.header("Folds del walk-forward (ML)")
rr = C.list_runs(P["reports"])
if not rr:
    C.missing("corridas de experimentos (reports/runs)", C.CMD_EXPERIMENTS)
else:
    run_id = st.selectbox("Corrida de experimentos", rr, index=rr.index(C.default_run(P["reports"], rr)),
                          key="ml_run")
    summary = C.read_json(str(P["reports"] / "runs" / run_id / "summary.json"))
    if summary is None:
        C.missing(f"summary.json de {run_id}")
    else:
        C.run_banner(summary)
        folds = pd.DataFrame(summary.get("folds") or [])
        if folds.empty:
            st.info("Esta corrida no tiene folds ML (solo reglas).")
        else:
            st.dataframe(folds, hide_index=True, width="stretch")
            fig = go.Figure()
            for i, s in enumerate(sorted(folds["symbol"].unique())):
                d = folds[folds["symbol"] == s]
                lab = d["fold"].astype(str)
                for col, name in (("n_fit", "n_fit"), ("n_calib", "n_calib"), ("n_test", "n_test")):
                    if col in d.columns:
                        fig.add_bar(x=[f"{s}·{f}" for f in lab], y=d[col], name=f"{name}")
            fig.update_layout(barmode="group")
            C.show(C.style_fig(fig, 320, "Tamaños de ajuste / calibración / prueba por fold"),
                   "Barras agrupadas: n_fit, n_calib y n_test por símbolo y fold.", key="folds_sizes")
            if "seconds" in folds.columns:
                f2 = go.Figure(go.Bar(x=[f"{s}·{f}" for s, f in zip(folds["symbol"], folds["fold"])],
                                      y=folds["seconds"], marker_color=C.CAT[0]))
                C.show(C.style_fig(f2, 260, "Segundos por fold"), "Tiempo de entrenamiento por fold.", key="folds_secs")

# ======================================================================= registro
st.header("Registro de modelos")
reg = P["models"] / "registry"
versions = [v for v in C.list_dirs(str(reg))]
active = C.read_text(str(reg / "ACTIVE"))
if not versions:
    C.missing("registro de modelos (models/registry/<versión>/meta.json)",
              note="Aún no hay modelos registrados; el registro lo escribe el pipeline de entrenamiento/promoción.")
else:
    st.caption(f"Versión activa (ACTIVE): {active or 'ninguna'}")
    rows = []
    for v in reversed(versions):
        meta = C.read_json(str(reg / v / "meta.json"))
        if not meta:
            rows.append({"versión": v, "estado": "meta.json ausente"})
            continue
        gate = meta.get("gate") or {}
        rows.append({"versión": v, "activa": v == active, "creada": meta.get("created_at"),
                     "estado": meta.get("status"), "puerta superada": gate.get("passed"),
                     "motivos de la puerta": "; ".join(gate.get("reasons", []) or []),
                     "corrida de validación": meta.get("validation_run"),
                     "símbolos": ", ".join((meta.get("symbols") or {}).keys())})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    pick = st.selectbox("Detalle de versión", list(reversed(versions)), key="reg_pick")
    meta = C.read_json(str(reg / pick / "meta.json"))
    if meta and meta.get("symbols"):
        d = pd.DataFrame([{"símbolo": s, **{k: v for k, v in x.items() if k != "features"},
                           "n_features": len(x.get("features", []))} for s, x in meta["symbols"].items()])
        st.dataframe(d, hide_index=True, width="stretch")
        for s, x in meta["symbols"].items():
            with st.expander(f"Características de {s}"):
                st.write(", ".join(x.get("features", [])))

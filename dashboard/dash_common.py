"""Utilidades compartidas del dashboard: rutas, lectura de artefactos y componentes comunes.

Regla de oro: solo se muestran datos que existen en disco. Si un artefacto falta se
muestra un aviso con el comando que lo genera; jamás se inventan cifras.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quant_ai.config import load_config

FOOTER = ("Resultados simulados; no son beneficios reales. Modo PAPER. "
          "Operación real bloqueada sin autorización expresa.")
ASSOC_NOTE = "Las importancias y explicaciones son asociativas, no prueban causalidad."

PY = r".venv\Scripts\python.exe"
CMD_EXPERIMENTS = f"{PY} scripts/run_experiments.py --tag baselines --skip-ml"
CMD_DATA = f"{PY} scripts/download_data.py"
CMD_PAPER = f"{PY} scripts/run_paper.py"
CMD_BENCH = f"{PY} scripts/gpu_benchmark.py"
CMD_TRAIN_DL = f"{PY} scripts/train_sequence.py   # (normalmente en Colab; ver notebooks/)"

# Paleta categórica en orden fijo (la identidad de la estrategia no cambia al filtrar).
CAT = ["#2a6fdb", "#e8743b", "#2fa37a", "#9b59b6", "#c9a227", "#17a2b8", "#d6457b", "#6b7280",
       "#8c564b", "#4c78a8", "#59a14f", "#b07aa1"]


# --------------------------------------------------------------------------- rutas
def paths() -> dict[str, Path]:
    """Rutas derivadas de load_config() (respeta QUANT_AI_HOME). No crea carpetas."""
    cfg = load_config()

    def p(name: str) -> Path:
        q = Path(cfg.raw["paths"][name])
        return q if q.is_absolute() else cfg.home / q

    state = p("state_dir")
    return {
        "home": cfg.home, "reports": p("reports_dir"), "models": p("models_dir"),
        "state": state, "data": p("data_dir"), "paper": state / "paper",
        "kill": cfg.home / "KILL_SWITCH", "docs": cfg.home / "docs",
    }


def config_mode() -> str:
    return str(load_config().raw.get("project", {}).get("mode", "paper"))


# --------------------------------------------------------------------------- lectura (cacheada)
@st.cache_data(ttl=60, show_spinner=False)
def read_json(path: str):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@st.cache_data(ttl=60, show_spinner=False)
def read_text(path: str) -> str | None:
    p = Path(path)
    return p.read_text(encoding="utf-8").strip() if p.exists() else None


@st.cache_data(ttl=60, show_spinner=False)
def read_parquet(path: str, columns: tuple | None = None) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p, columns=list(columns) if columns else None)
    except Exception:  # noqa: BLE001 - archivo corrupto: se trata como ausente
        return None


@st.cache_data(ttl=60, show_spinner=False)
def read_csv(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(ttl=60, show_spinner=False)
def read_jsonl(path: str) -> list[dict] | None:
    p = Path(path)
    if not p.exists():
        return None
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


@st.cache_data(ttl=60, show_spinner=False)
def sha256_file(path: str) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@st.cache_data(ttl=60, show_spinner=False)
def list_dirs(path: str) -> list[str]:
    p = Path(path)
    return sorted(d.name for d in p.iterdir() if d.is_dir()) if p.exists() else []


@st.cache_data(ttl=60, show_spinner=False)
def list_files(path: str, pattern: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [{"archivo": f.name, "tamaño_MB": round(f.stat().st_size / 1e6, 2),
             "modificado": datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)}
            for f in sorted(p.glob(pattern)) if f.is_file()]


def symbols_in_clean(data_dir: Path) -> list[str]:
    d = data_dir / "clean"
    return sorted(f.name.split("_")[0] for f in d.glob("*_1h.parquet")) if d.exists() else []


def load_ohlcv(data_dir: Path, symbol: str, columns: tuple | None = None) -> pd.DataFrame | None:
    df = read_parquet(str(data_dir / "clean" / f"{symbol}_1h.parquet"), columns)
    if df is None:
        return None
    if "open_time" in df.columns:
        df = df.set_index("open_time")
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


# --------------------------------------------------------------------------- avisos y formato
def missing(what: str, command: str | None = None, note: str | None = None) -> None:
    """Aviso uniforme de artefacto ausente, con el comando que lo genera."""
    st.warning(f"Sin datos: {what}", icon=":material/folder_off:")
    if note:
        st.caption(note)
    if command:
        st.caption("Para generarlo, ejecuta:")
        st.code(command, language="powershell")


def utcnow() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def ts(x) -> pd.Timestamp | None:
    if x in (None, ""):
        return None
    try:
        t = pd.Timestamp(x)
        return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    except (ValueError, TypeError):
        return None


def age_text(t: pd.Timestamp | None) -> str:
    if t is None:
        return "—"
    s = (utcnow() - t).total_seconds()
    if s < 0:
        return "en el futuro"
    if s < 3600:
        return f"hace {int(s // 60)} min"
    if s < 172800:
        return f"hace {s / 3600:.1f} h"
    return f"hace {s / 86400:.1f} días"


def pct(x, d: int = 2) -> str:
    return "—" if x is None or pd.isna(x) else f"{x * 100:.{d}f}%"


def num(x, d: int = 2) -> str:
    return "—" if x is None or pd.isna(x) else f"{x:,.{d}f}"


def mode_banner(mode: str | None = None) -> None:
    mode = (mode or config_mode()).upper()
    if mode == "PAPER":
        st.success("MODO PAPER: operaciones simuladas, sin dinero real.", icon=":material/science:")
    else:
        st.error(f"Modo configurado: {mode}. Este dashboard solo es válido para PAPER.", icon=":material/block:")
    st.caption("LIVE: bloqueado. Requiere autorización expresa y no se habilita desde este dashboard.")


# --------------------------------------------------------------------------- corridas de experimentos
def list_runs(reports: Path) -> list[str]:
    return list(reversed(list_dirs(str(reports / "runs"))))


def default_run(reports: Path, runs: list[str]) -> str | None:
    latest = read_text(str(reports / "latest_run.txt"))
    if latest in runs:
        return latest
    return runs[0] if runs else None


def run_banner(summary: dict) -> None:
    if summary.get("is_smoke") or summary.get("kind") == "smoke":
        st.error("PRUEBA MÍNIMA (smoke): ML con datos mínimos. NO es un resultado válido; "
                 "no la uses para evaluar la estrategia.", icon=":material/warning:")
    if summary.get("final_holdout_used"):
        st.warning("Esta corrida usó el holdout final (kind=final_holdout). Se consume una sola vez.")
    else:
        st.caption(f"Holdout final bloqueado desde {str(summary.get('final_holdout_start', '—'))[:10]}; "
                   "esta corrida no lo usó.")


def strategy_color(name: str, order: list[str]) -> str:
    return CAT[order.index(name) % len(CAT)] if name in order else CAT[-1]


def strategies_frame(summary: dict) -> pd.DataFrame:
    rows = []
    for name, m in (summary.get("strategies") or {}).items():
        r = {"estrategia": name, **{k: v for k, v in m.items() if not isinstance(v, (list, dict))}}
        ci = m.get("sharpe_ci95")
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            r["sharpe_ic95_inf"], r["sharpe_ic95_sup"] = ci
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- gráficos
def style_fig(fig: go.Figure, height: int = 380, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height, title=dict(text=title, x=0, font=dict(size=14)) if title else None,
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        legend=dict(orientation="h", y=1.08, x=0), hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    return fig


def show(fig: go.Figure, alt: str, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False})
    st.caption(alt)


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def mermaid_blocks(text: str) -> list[str]:
    return [m.strip("\n") for m in re.findall(r"```mermaid\s*\n(.*?)```", text, flags=re.S)]


def stop_page() -> None:
    """Detiene la página mostrando antes el pie (st.stop() impide que app.py lo dibuje)."""
    st.divider()
    st.caption(FOOTER)
    st.stop()

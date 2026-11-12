"""Etiquetado financiero.

Ambos métodos asumen la convención de ejecución del sistema: señal al cierre de la vela t,
entrada en ``open[t+1]``. Cada etiqueta devuelve ``t1``: el instante en que la etiqueta queda
determinada. ``t1`` se usa para purgar el entrenamiento (evita solapamiento con el test).

Spot sin apalancamiento => el problema es long/flat: y=1 si comprar en open[t+1] supera costes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_net_return_labels(df: pd.DataFrame, horizon: int, cost_threshold: float,
                              freq: str = "1h") -> pd.DataFrame:
    """y=1 si open[t+1+h]/open[t+1]-1 > cost_threshold. NaN si hay huecos dentro del horizonte."""
    d = df.set_index("open_time").sort_index()
    o = d["open"]
    entry = o.shift(-1)
    exit_ = o.shift(-(1 + horizon))
    t_exit = pd.Series(d.index, index=d.index).shift(-(1 + horizon))
    expected = pd.Timedelta(freq) * (1 + horizon)
    contiguous = (t_exit - d.index.to_series()) == expected
    fwd = exit_ / entry - 1
    out = pd.DataFrame(index=d.index)
    out["fwd_ret"] = fwd.where(contiguous)
    out["y"] = (out["fwd_ret"] > cost_threshold).astype("float64").where(out["fwd_ret"].notna())
    out["t1"] = t_exit.where(contiguous)
    return out


def triple_barrier_labels(df: pd.DataFrame, atr_pct: pd.Series, pt_mult: float, sl_mult: float,
                          max_bars: int, cost_threshold: float) -> pd.DataFrame:
    """Triple barrera long-only con barreras proporcionales al ATR conocido en t.

    Si en la misma vela se tocan ambas barreras se asume, conservadoramente, que primero se tocó el stop.
    Barrera vertical: y=1 si el retorno a open[t+1+max_bars] supera los costes.
    """
    d = df.set_index("open_time").sort_index()
    o, h, low = d["open"].to_numpy(), d["high"].to_numpy(), d["low"].to_numpy()
    a = atr_pct.reindex(d.index).to_numpy()
    idx = d.index
    n = len(d)
    y = np.full(n, np.nan)
    ret = np.full(n, np.nan)
    t1 = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    idx_np = idx.tz_convert(None).to_numpy() if idx.tz is not None else idx.to_numpy()
    for i in range(n - max_bars - 1):
        if not np.isfinite(a[i]):
            continue
        entry = o[i + 1]
        up, dn = entry * (1 + pt_mult * a[i]), entry * (1 - sl_mult * a[i])
        label, r, tt = None, None, None
        for j in range(i + 1, i + 1 + max_bars):
            hit_dn, hit_up = low[j] <= dn, h[j] >= up
            if hit_dn:  # conservador: stop primero
                label, r, tt = 0.0, dn / entry - 1, idx_np[j]
                break
            if hit_up:
                label, r, tt = 1.0, up / entry - 1, idx_np[j]
                break
        if label is None:
            r = o[i + 1 + max_bars] / entry - 1
            label, tt = float(r > cost_threshold), idx_np[i + 1 + max_bars]
        y[i], ret[i], t1[i] = label, r, tt
    out = pd.DataFrame({"y": y, "fwd_ret": ret, "t1": pd.to_datetime(t1).tz_localize("UTC")}, index=idx)
    return out


def make_labels(df: pd.DataFrame, cfg_labeling: dict, cost_threshold: float, atr_pct: pd.Series | None = None,
                freq: str = "1h") -> pd.DataFrame:
    if cfg_labeling["method"] == "forward_net_return":
        return forward_net_return_labels(df, cfg_labeling["horizon_bars"], cost_threshold, freq)
    if cfg_labeling["method"] == "triple_barrier":
        tb = cfg_labeling["triple_barrier"]
        if atr_pct is None:
            raise ValueError("triple_barrier requiere atr_pct")
        return triple_barrier_labels(df, atr_pct, tb["pt_atr_mult"], tb["sl_atr_mult"], tb["max_bars"],
                                     cost_threshold)
    raise ValueError(f"Método de etiquetado desconocido: {cfg_labeling['method']}")

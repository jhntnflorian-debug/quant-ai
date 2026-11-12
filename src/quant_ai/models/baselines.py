"""Estrategias de referencia (baselines) expresadas como peso objetivo decidido al cierre de cada vela."""
from __future__ import annotations

import numpy as np
import pandas as pd


def buy_and_hold(f: pd.DataFrame, weight: float) -> pd.Series:
    return pd.Series(weight, index=f.index)


def price_above_ema_fast(f: pd.DataFrame, weight: float) -> pd.Series:
    """Largo cuando el precio está sobre su EMA de 24h y de 168h. Cambia de estado muy seguido
    (whipsaw): se conserva como ejemplo de regla que los costes destruyen."""
    sig = (f["ema_dist_168"] > 0) & (f["ema_dist_24"] > 0)
    return sig.astype(float) * weight


def ema_cross(f: pd.DataFrame, weight: float) -> pd.Series:
    """Cruce clásico: largo cuando EMA(24h) > EMA(168h).
    Como ema_dist_w = close/EMA_w - 1, EMA24 > EMA168  <=>  ema_dist_168 > ema_dist_24."""
    return (f["ema_dist_168"] > f["ema_dist_24"]).astype(float) * weight


def time_series_momentum(f: pd.DataFrame, weight: float) -> pd.Series:
    return (f["ret_168"] > 0).astype(float) * weight


def rsi_reversion(f: pd.DataFrame, weight: float, enter: float = 30, exit_: float = 55) -> pd.Series:
    """Reversión: entra con RSI < enter, sale con RSI > exit_ (con estado)."""
    rsi = f["rsi_14"].to_numpy()
    out = np.zeros(len(rsi))
    inpos = False
    for i, v in enumerate(rsi):
        if not np.isfinite(v):
            out[i] = weight if inpos else 0.0
            continue
        if not inpos and v < enter:
            inpos = True
        elif inpos and v > exit_:
            inpos = False
        out[i] = weight if inpos else 0.0
    return pd.Series(out, index=f.index)


RULE_STRATEGIES = {
    "buy_and_hold": buy_and_hold,
    "price_above_ema_fast": price_above_ema_fast,
    "ema_cross_24_168": ema_cross,
    "tsmom_168": time_series_momentum,
    "rsi_reversion": rsi_reversion,
}

"""Ingeniería de características CAUSAL.

Convención temporal (crítica):
- La fila con índice ``open_time = t`` describe la vela [t, t+1h). Sus OHLCV se conocen al CIERRE (t+1h).
- Toda característica de la fila t usa solo velas con open_time <= t (rolling/ewm hacia atrás).
- La decisión se toma al cierre de la vela t y se ejecuta en la APERTURA de la vela t+1.
- No hay ``shift(-k)``, ``center=True`` ni normalizaciones con estadísticos de toda la serie aquí.
  Los escaladores se ajustan dentro de cada fold de entrenamiento (ver models/).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / (down + EPS)
    return 100 - 100 / (1 + rs)


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(),
                      (df["low"] - prev_close).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return _true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = _true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / (tr + EPS)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / (tr + EPS)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPS)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def _zscore(s: pd.Series, n: int) -> pd.Series:
    m = s.rolling(n, min_periods=n).mean()
    sd = s.rolling(n, min_periods=n).std()
    return (s - m) / (sd + EPS)


def build_features(df: pd.DataFrame, windows: list[int] = (6, 12, 24, 72, 168),
                   include_calendar: bool = True) -> pd.DataFrame:
    """Calcula características de un solo activo. ``df`` debe tener open_time y OHLCV de Binance."""
    d = df.set_index("open_time").sort_index()
    c, h, low, v = d["close"], d["high"], d["low"], d["volume"]
    f = pd.DataFrame(index=d.index)

    logc = np.log(c)
    r1 = logc.diff()
    f["ret_1"] = r1
    for w in windows:
        f[f"ret_{w}"] = logc.diff(w)
        if w >= 12:
            f[f"vol_{w}"] = r1.rolling(w, min_periods=w).std()
    f["vol_ratio_24_168"] = f["vol_24"] / (f["vol_168"] + EPS)

    # Tendencia
    for w in (12, 24, 72, 168):
        f[f"ema_dist_{w}"] = c / c.ewm(span=w, adjust=False, min_periods=w).mean() - 1
    ema12 = c.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = c.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    f["macd"] = macd / c
    f["macd_hist"] = (macd - sig) / c
    f["adx_14"] = _adx(d, 14)

    # Osciladores / volatilidad
    f["rsi_14"] = _rsi(c, 14)
    a14 = atr(d, 14)
    f["atr_pct_14"] = a14 / c
    m20 = c.rolling(20, min_periods=20).mean()
    s20 = c.rolling(20, min_periods=20).std()
    f["bb_pctb"] = (c - (m20 - 2 * s20)) / (4 * s20 + EPS)
    f["bb_width"] = 4 * s20 / (m20 + EPS)
    f["range_pct"] = (h - low) / c
    f["close_in_range"] = (c - low) / (h - low + EPS)
    f["dist_max_168"] = c / c.rolling(168, min_periods=168).max() - 1
    f["dist_min_168"] = c / c.rolling(168, min_periods=168).min() - 1

    # Volumen y microestructura derivable de klines (volumen agresor comprador)
    f["vol_z_168"] = _zscore(np.log(v + 1), 168)
    taker_ratio = d["taker_buy_base"] / (v + EPS)
    f["taker_buy_ratio"] = taker_ratio - 0.5
    f["taker_buy_ratio_24"] = taker_ratio.rolling(24, min_periods=24).mean() - 0.5
    avg_trade = d["quote_volume"] / (d["trades"] + 1)
    f["avg_trade_z_168"] = _zscore(np.log(avg_trade + 1), 168)

    # Precio de referencia tipo VWAP de 24h
    vwap24 = (d["quote_volume"].rolling(24, min_periods=24).sum() / (v.rolling(24, min_periods=24).sum() + EPS))
    f["vwap24_dist"] = c / vwap24 - 1

    if include_calendar:
        hour = d.index.hour
        dow = d.index.day_of_week
        f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        f["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        f["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    return f.replace([np.inf, -np.inf], np.nan)


def add_cross_asset(features: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Añade a cada activo los retornos de los otros y la correlación móvil (alineado por open_time)."""
    out = {}
    for sym, f in features.items():
        g = f.copy()
        for other, fo in features.items():
            if other == sym:
                continue
            tag = other.replace("USDT", "").lower()
            o = fo.reindex(g.index)  # solo misma marca temporal; nunca valores futuros
            g[f"x_{tag}_ret_1"] = o["ret_1"]
            g[f"x_{tag}_ret_24"] = o["ret_24"]
            g[f"x_{tag}_corr_168"] = g["ret_1"].rolling(168, min_periods=120).corr(o["ret_1"])
            g[f"x_{tag}_rel_ret_24"] = g["ret_24"] - o["ret_24"]
        out[sym] = g
    return out


def build_feature_set(raw: dict[str, pd.DataFrame], cfg_features: dict) -> dict[str, pd.DataFrame]:
    feats = {s: build_features(df, cfg_features.get("windows", (6, 12, 24, 72, 168)),
                               cfg_features.get("include_calendar", True)) for s, df in raw.items()}
    if cfg_features.get("include_cross_asset", True) and len(feats) > 1:
        feats = add_cross_asset(feats)
    return feats


def feature_columns(f: pd.DataFrame) -> list[str]:
    return [c for c in f.columns if not c.startswith("_")]

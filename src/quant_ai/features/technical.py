"""Análisis técnico ampliado (grupo ``ta_``), 100 % causal.

Reglas que evitan fugas de información:
- Ventanas móviles y EWM solo hacia atrás.
- Pivotes (swing highs/lows): un máximo en i se CONFIRMA en i+k (necesita k velas posteriores), así que
  su nivel solo se usa desde i+k. Implementado con high.shift(k) == max(high[i-2k..i]).
- Marcos superiores (4 h, 1 d): una vela de 4 h que empieza en T cubre [T, T+4h) y solo se usa en filas
  cuya decisión ocurre en T+4h o después (t + 1h >= T + 4h). Barras incompletas (por huecos) -> NaN.
- Ichimoku: la nube que se ve en t se calculó en t-26 (desplazamiento hacia atrás, causal).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant_ai.features.engineering import EPS, _rsi, atr

H = pd.Timedelta(hours=1)


def _roll_max(s, n):
    return s.rolling(n, min_periods=n).max()


def _roll_min(s, n):
    return s.rolling(n, min_periods=n).min()


def _z(s, n):
    return (s - s.rolling(n, min_periods=n).mean()) / (s.rolling(n, min_periods=n).std() + EPS)


def _supertrend(h, low, c, a, mult=3.0):
    """Dirección de Supertrend (+1 alcista / -1 bajista) con estado; solo usa datos hasta t."""
    hl2 = ((h + low) / 2).to_numpy()
    av = a.to_numpy()
    cv = c.to_numpy()
    up, dn = hl2 - mult * av, hl2 + mult * av
    n = len(cv)
    dirn = np.full(n, np.nan)
    line = np.full(n, np.nan)  # banda activa EN CADA PASO (no el valor final: eso filtraría el futuro)
    fu, fd = np.nan, np.nan
    d = 1.0
    for i in range(1, n):
        if not np.isfinite(av[i]):
            continue
        fu = up[i] if not np.isfinite(fu) or cv[i - 1] <= fu else max(up[i], fu)
        fd = dn[i] if not np.isfinite(fd) or cv[i - 1] >= fd else min(dn[i], fd)
        if cv[i] > fd:
            d = 1.0
        elif cv[i] < fu:
            d = -1.0
        dirn[i] = d
        line[i] = fu if d > 0 else fd
    return pd.Series(dirn, index=c.index), pd.Series(line, index=c.index)


def _ffd_weights(d: float, thresh: float = 1e-4, max_len: int = 500) -> np.ndarray:
    w = [1.0]
    for k in range(1, max_len):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thresh:
            break
        w.append(w_k)
    return np.array(w)


def frac_diff(series: pd.Series, d: float = 0.4) -> pd.Series:
    """Diferenciación fraccional de ventana fija (López de Prado): estacionaria y con memoria. Causal."""
    w = _ffd_weights(d)
    x = series.to_numpy(dtype=float)
    out = np.convolve(x, w, mode="full")[:len(x)]  # out[i] = sum_j w_j * x[i-j]: solo pasado
    out[:len(w) - 1] = np.nan
    return pd.Series(out, index=series.index)


def _higher_tf(d: pd.DataFrame, rule: str, n_bars: int) -> pd.DataFrame:
    """Velas de marco superior indexadas por su instante de CIERRE; NaN si están incompletas."""
    agg = d.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    count = d["close"].resample(rule, label="left", closed="left").count()
    agg = agg[count == n_bars]
    agg.index = agg.index + pd.Timedelta(rule)  # disponible al cierre de la vela superior
    return agg


def _mtf_features(d: pd.DataFrame, rule: str, n_bars: int, tag: str) -> pd.DataFrame:
    hb = _higher_tf(d, rule, n_bars)
    c = hb["close"]
    f = pd.DataFrame(index=hb.index)
    f[f"ta_{tag}_rsi_14"] = _rsi(c, 14)
    f[f"ta_{tag}_ema20_dist"] = c / c.ewm(span=20, adjust=False, min_periods=20).mean() - 1
    f[f"ta_{tag}_ema50_dist"] = c / c.ewm(span=50, adjust=False, min_periods=50).mean() - 1
    f[f"ta_{tag}_ret_1"] = np.log(c).diff()
    f[f"ta_{tag}_atr_pct"] = atr(hb, 14) / c
    # alinear a filas horarias por instante de decisión (t + 1h) usando solo velas superiores ya cerradas
    decision = d.index + H
    aligned = f.reindex(f.index.union(decision)).sort_index().ffill().reindex(decision)
    aligned.index = d.index
    return aligned


def build_technical_features(df: pd.DataFrame, pivot_k: int = 5) -> pd.DataFrame:
    d = df.set_index("open_time").sort_index() if "open_time" in df.columns else df.sort_index()
    o, h, low, c, v = d["open"], d["high"], d["low"], d["close"], d["volume"]
    f = pd.DataFrame(index=d.index)
    logc = np.log(c)
    r1 = logc.diff()
    a14 = atr(d, 14)

    # --- Tendencia -----------------------------------------------------------
    tenkan = (_roll_max(h, 9) + _roll_min(low, 9)) / 2
    kijun = (_roll_max(h, 26) + _roll_min(low, 26)) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((_roll_max(h, 52) + _roll_min(low, 52)) / 2).shift(26)
    cloud_top, cloud_bot = np.maximum(span_a, span_b), np.minimum(span_a, span_b)
    f["ta_ichi_tk"] = tenkan / kijun - 1
    f["ta_ichi_close_kijun"] = c / kijun - 1
    f["ta_ichi_above_cloud"] = (c - cloud_top) / c
    f["ta_ichi_below_cloud"] = (cloud_bot - c) / c
    f["ta_ichi_cloud_width"] = (cloud_top - cloud_bot) / c
    for n in (20, 55, 168):
        hi_prev, lo_prev = _roll_max(h, n).shift(1), _roll_min(low, n).shift(1)
        f[f"ta_donchian_pos_{n}"] = (c - lo_prev) / (hi_prev - lo_prev + EPS)
        f[f"ta_breakout_up_{n}"] = (c > hi_prev).astype(float).where(hi_prev.notna())
        f[f"ta_breakout_dn_{n}"] = (c < lo_prev).astype(float).where(lo_prev.notna())
    ema20 = c.ewm(span=20, adjust=False, min_periods=20).mean()
    f["ta_keltner_pos"] = (c - ema20) / (2 * a14 + EPS)
    st_dir, st_line = _supertrend(h, low, c, atr(d, 10), 3.0)
    f["ta_supertrend_dir"] = st_dir
    f["ta_supertrend_dist"] = c / st_line - 1
    up_age = h.rolling(25, min_periods=25).apply(lambda x: 24 - np.argmax(x), raw=True)
    dn_age = low.rolling(25, min_periods=25).apply(lambda x: 24 - np.argmin(x), raw=True)
    f["ta_aroon_osc"] = ((25 - up_age) - (25 - dn_age)) / 25
    f["ta_efficiency_24"] = (c - c.shift(24)).abs() / (c.diff().abs().rolling(24, min_periods=24).sum() + EPS)
    f["ta_efficiency_72"] = (c - c.shift(72)).abs() / (c.diff().abs().rolling(72, min_periods=72).sum() + EPS)

    # --- Osciladores ---------------------------------------------------------
    ll14, hh14 = _roll_min(low, 14), _roll_max(h, 14)
    stoch_k = (c - ll14) / (hh14 - ll14 + EPS)
    f["ta_stoch_k"] = stoch_k
    f["ta_stoch_d"] = stoch_k.rolling(3, min_periods=3).mean()
    f["ta_williams_r"] = stoch_k - 1
    rsi = _rsi(c, 14)
    f["ta_stochrsi"] = (rsi - _roll_min(rsi, 14)) / (_roll_max(rsi, 14) - _roll_min(rsi, 14) + EPS)
    tp = (h + low + c) / 3
    md = tp.rolling(20, min_periods=20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    f["ta_cci_20"] = (tp - tp.rolling(20, min_periods=20).mean()) / (0.015 * md + EPS)
    for n in (3, 48, 336):
        f[f"ta_roc_{n}"] = c / c.shift(n) - 1
    # divergencia RSI: precio en máximo de 24 h pero RSI no (y viceversa)
    f["ta_bear_div"] = ((c >= _roll_max(c, 24)) & (rsi < _roll_max(rsi, 24) - 5)).astype(float).where(rsi.notna())
    f["ta_bull_div"] = ((c <= _roll_min(c, 24)) & (rsi > _roll_min(rsi, 24) + 5)).astype(float).where(rsi.notna())

    # --- Volumen y flujo -------------------------------------------------------
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    f["ta_obv_slope_z_24"] = _z(obv.diff(24), 168)
    mf = tp * v
    pos = mf.where(tp > tp.shift(1), 0).rolling(14, min_periods=14).sum()
    neg = mf.where(tp < tp.shift(1), 0).rolling(14, min_periods=14).sum()
    f["ta_mfi_14"] = 100 - 100 / (1 + pos / (neg + EPS))
    mfm = ((c - low) - (h - c)) / (h - low + EPS)
    f["ta_cmf_20"] = (mfm * v).rolling(20, min_periods=20).sum() / (v.rolling(20, min_periods=20).sum() + EPS)
    if "taker_buy_base" in d:
        delta = 2 * d["taker_buy_base"] - v  # volumen agresor comprador - vendedor
        for n in (24, 168):
            f[f"ta_cvd_{n}"] = delta.rolling(n, min_periods=n).sum() / (v.rolling(n, min_periods=n).sum() + EPS)
    f["ta_force_z"] = _z(c.diff() * v, 168)
    vwap168 = (tp * v).rolling(168, min_periods=168).sum() / (v.rolling(168, min_periods=168).sum() + EPS)
    f["ta_vwap168_dist"] = c / vwap168 - 1
    f["ta_amihud_z"] = _z((r1.abs() / (d.get("quote_volume", v * c) + EPS)).rolling(24, min_periods=24).mean(), 168)

    # --- Estimadores de volatilidad --------------------------------------------
    hl = np.log(h / low)
    co = np.log(c / o)
    oc_prev = np.log(o / c.shift(1))
    for n in (24, 168):
        park = np.sqrt((hl ** 2).rolling(n, min_periods=n).mean() / (4 * np.log(2)))
        gk = np.sqrt((0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2).rolling(n, min_periods=n).mean().clip(lower=0))
        rs = (np.log(h / c) * np.log(h / o) + np.log(low / c) * np.log(low / o)).rolling(n, min_periods=n).mean()
        k = 0.34 / (1.34 + (n + 1) / (n - 1))
        yz = np.sqrt((oc_prev.rolling(n, min_periods=n).var() + k * co.rolling(n, min_periods=n).var()
                      + (1 - k) * rs).clip(lower=0))
        f[f"ta_vol_park_{n}"] = park
        f[f"ta_vol_gk_{n}"] = gk
        f[f"ta_vol_yz_{n}"] = yz
    f["ta_vol_yz_ratio"] = f["ta_vol_yz_24"] / (f["ta_vol_yz_168"] + EPS)

    # --- Velas japonesas -------------------------------------------------------
    rng = (h - low).replace(0, np.nan)
    body = c - o
    f["ta_body_ratio"] = body / rng
    f["ta_upper_wick"] = (h - np.maximum(o, c)) / rng
    f["ta_lower_wick"] = (np.minimum(o, c) - low) / rng
    prev_body = body.shift(1)
    f["ta_bull_engulf"] = ((body > 0) & (prev_body < 0) & (c >= o.shift(1)) & (o <= c.shift(1))).astype(float)
    f["ta_bear_engulf"] = ((body < 0) & (prev_body > 0) & (c <= o.shift(1)) & (o >= c.shift(1))).astype(float)
    f["ta_hammer"] = ((f["ta_lower_wick"] > 0.6) & (f["ta_body_ratio"].abs() < 0.3)).astype(float)
    f["ta_shooting_star"] = ((f["ta_upper_wick"] > 0.6) & (f["ta_body_ratio"].abs() < 0.3)).astype(float)
    f["ta_doji"] = (f["ta_body_ratio"].abs() < 0.1).astype(float)
    f["ta_inside_bar"] = ((h < h.shift(1)) & (low > low.shift(1))).astype(float)
    f["ta_outside_bar"] = ((h > h.shift(1)) & (low < low.shift(1))).astype(float)
    up = (c > c.shift(1)).astype(int)
    f["ta_up_streak"] = up.groupby((up != up.shift()).cumsum()).cumsum() * up
    dn = 1 - up
    f["ta_down_streak"] = dn.groupby((dn != dn.shift()).cumsum()).cumsum() * dn

    # --- Soportes/resistencias y estructura de mercado (pivotes confirmados) -------------
    k = pivot_k
    win = 2 * k + 1
    piv_hi = (h.shift(k) == _roll_max(h, win)) & h.shift(k).notna()   # confirmado en t (el pivote está en t-k)
    piv_lo = (low.shift(k) == _roll_min(low, win)) & low.shift(k).notna()
    last_hi = h.shift(k).where(piv_hi).ffill()
    last_lo = low.shift(k).where(piv_lo).ffill()
    f["ta_dist_resistance"] = last_hi / c - 1
    f["ta_dist_support"] = c / last_lo - 1
    f["ta_resistance_atr"] = (last_hi - c) / (a14 + EPS)
    f["ta_support_atr"] = (c - last_lo) / (a14 + EPS)
    hi_vals = h.shift(k).where(piv_hi)
    lo_vals = low.shift(k).where(piv_lo)
    hh = (hi_vals > hi_vals.ffill().shift(1)).astype(float).where(piv_hi)
    hl_ = (lo_vals > lo_vals.ffill().shift(1)).astype(float).where(piv_lo)
    f["ta_struct_hh_ratio"] = hh.ffill().rolling(168, min_periods=24).mean()
    f["ta_struct_hl_ratio"] = hl_.ffill().rolling(168, min_periods=24).mean()

    # --- Estadística de la serie -----------------------------------------------
    r4 = logc.diff(4)
    f["ta_var_ratio_4"] = r4.rolling(168, min_periods=168).var() / (4 * r1.rolling(168, min_periods=168).var() + EPS)
    f["ta_autocorr_1_168"] = r1.rolling(168, min_periods=120).corr(r1.shift(1))
    f["ta_skew_168"] = r1.rolling(168, min_periods=168).skew()
    f["ta_kurt_168"] = r1.rolling(168, min_periods=168).kurt()
    f["ta_fracdiff_04"] = frac_diff(logc, 0.4)
    f["ta_fracdiff_04_z"] = _z(f["ta_fracdiff_04"], 720)

    # --- Calendario de mercado -------------------------------------------------
    hour = d.index.hour
    f["ta_hours_to_funding"] = ((8 - (hour + 1) % 8) % 8) / 8  # tras el cierre de la vela
    f["ta_us_session"] = ((hour >= 13) & (hour < 21)).astype(float)
    f["ta_asia_session"] = ((hour >= 0) & (hour < 8)).astype(float)

    # --- Multi-timeframe (solo velas superiores ya cerradas) ---------------------
    f = f.join(_mtf_features(d, "4h", 4, "4h")).join(_mtf_features(d, "1D", 24, "1d"))
    return f.replace([np.inf, -np.inf], np.nan)

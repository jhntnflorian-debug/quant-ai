"""Pruebas anti-fuga del análisis técnico ampliado (ta_)."""
import numpy as np
import pandas as pd

from conftest import make_ohlcv
from quant_ai.features.technical import build_technical_features, frac_diff


def test_technical_invariant_to_future_mutation():
    df = make_ohlcv(1200, seed=11)
    full = build_technical_features(df)
    assert full.shape[1] >= 70 and all(c.startswith("ta_") for c in full.columns)
    rng = np.random.default_rng(5)
    for cut in (400, 777, 1000):
        mut = df.copy()
        for col in ["open", "high", "low", "close", "volume", "taker_buy_base", "quote_volume"]:
            mut.loc[cut + 1:, col] = mut.loc[cut + 1:, col] * rng.uniform(0.6, 1.6, len(mut) - cut - 1)
        mut["high"] = mut[["open", "high", "close"]].max(axis=1)
        mut["low"] = mut[["open", "low", "close"]].min(axis=1)
        after = build_technical_features(mut)
        pd.testing.assert_frame_equal(full.iloc[: cut + 1], after.iloc[: cut + 1], check_exact=False, rtol=1e-9)


def test_technical_equal_on_truncated_history():
    df = make_ohlcv(1000, seed=12)
    full = build_technical_features(df)
    for t in (500, 731, 999):
        trunc = build_technical_features(df.iloc[: t + 1])
        pd.testing.assert_series_equal(full.iloc[t], trunc.iloc[-1], check_names=False, check_exact=False, rtol=1e-9)


def test_higher_timeframe_uses_only_closed_bars():
    df = make_ohlcv(400, seed=13, start="2024-01-01")
    f = build_technical_features(df)
    col = "ta_4h_ret_1"
    # la vela de 4h [00:00, 04:00) se conoce al cierre de la vela horaria de las 03:00 (decisión 04:00)
    idx = f.index
    t_open = pd.Timestamp("2024-01-05 00:00", tz="UTC")
    rows = f.loc[(idx >= t_open) & (idx < t_open + pd.Timedelta(hours=4)), col]
    assert rows.iloc[0] == rows.iloc[1] == rows.iloc[2]   # 00h, 01h, 02h: aún vela 4h anterior
    assert rows.iloc[3] != rows.iloc[2]                    # 03h: ya cerró la vela 4h que empezó a las 00h


def test_frac_diff_is_causal():
    s = pd.Series(np.cumsum(np.random.default_rng(1).normal(size=800)))
    a = frac_diff(s, 0.4)
    s2 = s.copy()
    s2.iloc[600:] += 100
    b = frac_diff(s2, 0.4)
    pd.testing.assert_series_equal(a.iloc[:600], b.iloc[:600])

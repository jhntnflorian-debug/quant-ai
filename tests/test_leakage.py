"""Pruebas anti-fuga temporal: las características en t no pueden depender de datos posteriores a t."""
import numpy as np
import pandas as pd

from conftest import make_ohlcv
from quant_ai.features.engineering import add_cross_asset, build_features
from quant_ai.labeling.labels import forward_net_return_labels


def test_features_invariant_to_future_mutation():
    df = make_ohlcv(800, seed=1)
    full = build_features(df)
    cut = 500
    mutated = df.copy()
    rng = np.random.default_rng(99)
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base"]:
        mutated.loc[cut + 1:, c] = mutated.loc[cut + 1:, c] * rng.uniform(0.5, 2.0, len(mutated) - cut - 1)
    after = build_features(mutated)
    pd.testing.assert_frame_equal(full.iloc[: cut + 1], after.iloc[: cut + 1])


def test_features_equal_on_truncated_history():
    """Calcular con historia completa o truncada en t debe dar lo mismo en t (sin normalización global)."""
    df = make_ohlcv(700, seed=2)
    full = build_features(df)
    for t in (300, 450, 699):
        trunc = build_features(df.iloc[: t + 1])
        pd.testing.assert_series_equal(full.iloc[t], trunc.iloc[-1], check_names=False)


def test_cross_asset_features_causal():
    a, b = make_ohlcv(600, seed=3), make_ohlcv(600, seed=4)
    base = add_cross_asset({"AAA": build_features(a), "BBB": build_features(b)})
    b2 = b.copy()
    b2.loc[401:, "close"] *= 3
    mut = add_cross_asset({"AAA": build_features(a), "BBB": build_features(b2)})
    pd.testing.assert_frame_equal(base["AAA"].iloc[:401], mut["AAA"].iloc[:401])


def test_labels_use_next_open_and_report_t1():
    df = make_ohlcv(100, seed=5)
    lab = forward_net_return_labels(df, horizon=4, cost_threshold=0.0)
    i = 10
    expected = df["open"].iloc[i + 1 + 4] / df["open"].iloc[i + 1] - 1
    assert np.isclose(lab["fwd_ret"].iloc[i], expected)
    assert lab["t1"].iloc[i] == df["open_time"].iloc[i + 5]
    assert lab["fwd_ret"].iloc[-5:].isna().all()  # sin futuro suficiente => NaN


def test_labels_nan_across_gaps():
    df = make_ohlcv(100, seed=6).drop(index=[50]).reset_index(drop=True)
    lab = forward_net_return_labels(df, horizon=4, cost_threshold=0.0)
    # filas cuya ventana [t, t+5] cruza el hueco no tienen etiqueta
    t_gap = pd.Timestamp("2023-01-01", tz="UTC") + pd.Timedelta(hours=50)
    affected = lab[(lab.index < t_gap) & (lab.index >= t_gap - pd.Timedelta(hours=5))]
    assert affected["y"].isna().all()

"""Pruebas del experimento v2: volatility targeting, meta-etiquetado y puerta de promoción."""
import numpy as np
import pandas as pd

from conftest import make_ohlcv
from quant_ai.config import load_config
from quant_ai.decision_engine.policy import vol_target_weights
from quant_ai.features.engineering import build_features
from quant_ai.training.dataset import SymbolDataset
from quant_ai.training.experiment_v2 import evaluate_gate, primary_signal, variant_datasets


def test_vol_target_bounds_and_quantization():
    idx = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    hourly = pd.Series([0.001, 0.005, 0.01, 0.05, np.nan], index=idx)  # anual ≈ 9 %, 47 %, 94 %, 468 %
    w = vol_target_weights(hourly, target_annual=0.40, max_weight=0.5, step=0.1)
    assert w.iloc[0] == 0.5                       # baja volatilidad: tope
    assert (w <= 0.5).all() and (w >= 0).all()    # nunca apalancado
    assert np.allclose((w / 0.1) % 1, 0, atol=1e-9) or np.allclose(w % 0.1, 0, atol=1e-9)
    assert w.iloc[3] == 0.0 and w.iloc[4] == 0.0  # volatilidad extrema o desconocida: fuera
    assert w.is_monotonic_decreasing


def test_meta_labeling_trains_only_on_primary_bars():
    cfg = load_config()
    df = make_ohlcv(1500, seed=3)
    feats = build_features(df)
    ds = SymbolDataset("BTCUSDT", df, feats, pd.DataFrame(index=feats.index), list(feats.columns))
    groups = {"BTCUSDT": {"base": feats}}
    spec = {"features": ["base"], "horizon": 24, "primary": "ema_cross"}
    vds = variant_datasets(cfg, {"BTCUSDT": ds}, groups, spec)["BTCUSDT"]
    prim = primary_signal(feats, "ema_cross").reindex(vds.labels.index).fillna(False)
    assert vds.labels.loc[~prim, "y"].isna().all()           # sin señal primaria: sin etiqueta
    assert vds.labels.loc[prim, "y"].notna().any()
    # etiqueta a 24 h: entrada open[t+1], salida open[t+25]
    i = int(np.flatnonzero(prim.to_numpy() & vds.labels["fwd_ret"].notna().to_numpy())[0])
    expected = df["open"].iloc[i + 25] / df["open"].iloc[i + 1] - 1
    assert np.isclose(vds.labels["fwd_ret"].iloc[i], expected)


def test_meta_policy_resets_state_when_primary_turns_off():
    """Sin señal primaria la histéresis no puede quedar 'dentro': al reactivarse exige p >= enter_prob."""
    from quant_ai.training.experiment_v2 import variant_weights

    cfg = load_config()
    idx = pd.date_range("2024-01-01", periods=16, freq="1h", tz="UTC")
    on = np.r_[[1] * 5, [0] * 5, [1] * 6].astype(bool)
    feats = pd.DataFrame({"ema_dist_168": np.where(on, 0.02, -0.02), "ema_dist_24": 0.0, "vol_168": 0.01},
                         index=idx)
    p = np.r_[[0.60] * 5, [0.50] * 5, [0.50] * 5, [0.56]]
    preds = pd.DataFrame({"symbol": "BTCUSDT", "p_ensemble": p, "uncertainty": 0.0}, index=idx)
    ds = SymbolDataset("BTCUSDT", pd.DataFrame(), feats, pd.DataFrame(index=idx), list(feats.columns))
    w = variant_weights(cfg, {"sizing": {}}, {"primary": "ema_cross", "sizing": "fixed"}, {"BTCUSDT": ds},
                        preds)["BTCUSDT"]
    assert (w.iloc[:5] > 0).all()                 # señal activa y p >= 0.55: dentro
    assert (w.iloc[5:10] == 0).all()              # señal apagada: fuera aunque p >= exit_prob
    assert (w.iloc[10:15] == 0).all()             # señal reactivada con p=0.50 < 0.55: NO re-entra
    assert w.iloc[15] > 0                         # p=0.56 >= 0.55: entra


def test_gate_requires_dsr_bootstrap_and_complete_evidence():
    v2 = {"gate": {"min_dsr": 0.95, "must_beat": ["buy_and_hold"], "max_drawdown_not_worse_than": "buy_and_hold",
                   "positive_under": "stress_x2"}}
    base = {"ml_a": {"sharpe": 1.5, "max_drawdown": -0.2, "total_return": 0.5},
            "ml_b": {"sharpe": 1.5, "max_drawdown": -0.2, "total_return": 0.5},
            "ml_nan": {"sharpe": float("nan"), "max_drawdown": -0.2, "total_return": 0.5},
            "buy_and_hold": {"sharpe": 1.0, "max_drawdown": -0.5, "total_return": 1.0}}
    stress = {"ml_a": {"total_return": 0.1}, "ml_b": {"total_return": -0.1}, "ml_nan": {"total_return": 0.1}}
    good_boot = {"buy_and_hold": {"diff": 0.5, "lo": 0.1, "hi": 0.9}}
    boot = {"ml_a": good_boot, "ml_b": good_boot, "ml_nan": good_boot}
    dsr = {k: {"dsr": 0.97} for k in ("ml_a", "ml_b", "ml_nan")}
    sc = {"taker_base": base, "stress_x2": stress}
    g = evaluate_gate(v2, sc, dsr, boot, is_complete=True)
    assert g["ml_a"]["passed"]
    assert not g["ml_b"]["passed"]                      # pierde con estrés de costes
    assert not g["ml_nan"]["passed"]                    # métricas no finitas nunca pasan
    assert "buy_and_hold" not in g                      # las referencias no son candidatas
    # el IC del bootstrap incluye 0 -> no pasa aunque el Sharpe puntual sea mayor
    weak = {"ml_a": {"buy_and_hold": {"diff": 0.5, "lo": -0.2, "hi": 1.2}}}
    assert not evaluate_gate(v2, sc, dsr, weak, True)["ml_a"]["passed"]
    # DSR bajo, prueba mínima o referencia ausente -> no pasa
    assert not evaluate_gate(v2, sc, {"ml_a": {"dsr": 0.5}}, boot, True)["ml_a"]["passed"]
    assert not evaluate_gate(v2, sc, dsr, boot, is_complete=False)["ml_a"]["passed"]
    v2_missing = {"gate": {**v2["gate"], "must_beat": ["ema_cross_vol_target"]}}
    assert not evaluate_gate(v2_missing, sc, dsr, boot, True)["ml_a"]["passed"]

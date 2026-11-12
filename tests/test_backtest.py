import numpy as np
import pandas as pd
import pytest

from conftest import make_ohlcv
from quant_ai.backtesting.engine import CostModel, Rules, run_backtest
from quant_ai.backtesting.metrics import performance_metrics

ZERO = CostModel(0, 0, 0, max_participation=1e9)


def _prices(df):
    return df[["open_time", "open", "high", "low", "close", "volume"]]


def test_signal_executes_at_next_open_not_same_bar():
    df = make_ohlcv(50, seed=1)
    w = pd.Series(0.0, index=df["open_time"])
    w.iloc[10] = 1.0  # señal al cierre de la vela 10
    w.iloc[11:] = 1.0
    res = run_backtest({"X": _prices(df)}, {"X": w}, ZERO, {"X": Rules(1e-8, 0)}, 1000.0)
    first = res.fills.iloc[0]
    assert first["time"] == df["open_time"].iloc[11]
    assert np.isclose(first["fill_price"], df["open"].iloc[11])


def test_costs_reduce_equity_and_are_recorded():
    df = make_ohlcv(100, seed=2)
    w = pd.Series(np.tile([1.0] * 5 + [0.0] * 5, 10), index=df["open_time"])
    c = CostModel(10, 1, 2, max_participation=1e9)
    r0 = run_backtest({"X": _prices(df)}, {"X": w}, ZERO, {"X": Rules(1e-8, 0)}, 1000.0)
    r1 = run_backtest({"X": _prices(df)}, {"X": w}, c, {"X": Rules(1e-8, 0)}, 1000.0)
    assert r1.equity["equity"].iloc[-1] < r0.equity["equity"].iloc[-1]
    assert r1.fills["fee"].sum() > 0
    buy = r1.fills[r1.fills.side == "BUY"].iloc[0]
    assert buy["fill_price"] > buy["ref_price"]  # compras pagan spread+slippage


def test_buy_and_hold_matches_price_return_without_costs():
    df = make_ohlcv(200, seed=3)
    w = pd.Series(1.0, index=df["open_time"])
    r = run_backtest({"X": _prices(df)}, {"X": w}, ZERO, {"X": Rules(1e-10, 0)}, 1000.0)
    expected = df["close"].iloc[-1] / df["open"].iloc[1]
    assert np.isclose(r.equity["equity"].iloc[-1] / 1000.0, expected, rtol=1e-6)


def test_no_leverage_and_cash_never_negative():
    a, b = make_ohlcv(150, seed=4), make_ohlcv(150, seed=5)
    w = pd.Series(1.0, index=a["open_time"])  # pide 100 % en cada activo => 200 %
    r = run_backtest({"A": _prices(a), "B": _prices(b)}, {"A": w, "B": w}, CostModel(10, 1, 2, 1e9),
                     None, 1000.0)
    assert (r.equity["cash"] >= -1e-6).all()
    assert (r.equity["gross_exposure"] <= 1.0 + 1e-6).all()


def test_partial_fill_by_participation():
    df = make_ohlcv(30, seed=6)
    df["volume"] = 1.0
    w = pd.Series(1.0, index=df["open_time"])
    r = run_backtest({"X": _prices(df)}, {"X": w}, CostModel(0, 0, 0, max_participation=0.01),
                     {"X": Rules(1e-8, 0)}, 1_000_000.0)
    assert r.fills["qty"].iloc[0] == pytest.approx(0.01)
    assert (r.fills["qty"] < r.fills["requested_qty"]).all()


def test_min_notional_blocks_tiny_orders():
    df = make_ohlcv(30, seed=7)
    w = pd.Series(0.001, index=df["open_time"])
    r = run_backtest({"X": _prices(df)}, {"X": w}, ZERO, {"X": Rules(1e-8, 5.0)}, 1000.0, rebalance_band=0)
    assert r.fills.empty


def test_stop_fills_at_open_when_gapping_below_stop():
    df = make_ohlcv(40, seed=8)
    # precios planos (rango ±0,1 %) para que el stop solo pueda dispararse en el gap
    df[["open", "close"]] = 100.0
    df["high"], df["low"] = 100.1, 99.9
    w = pd.Series(1.0, index=df["open_time"])
    stops = pd.Series(0.02, index=df["open_time"])
    # gap bajista fuerte en la vela 20: abre en 80, muy por debajo del stop (98)
    df.loc[20, ["open", "high", "low", "close"]] = [80.0, 81.0, 79.0, 80.0]
    r = run_backtest({"X": _prices(df)}, {"X": w}, ZERO, {"X": Rules(1e-8, 0)}, 1000.0, stops={"X": stops})
    stop_fill = r.fills[r.fills.reason == "stop_loss"].iloc[0]
    assert stop_fill["time"] == df["open_time"].iloc[20]
    assert np.isclose(stop_fill["fill_price"], df["open"].iloc[20])  # no al precio del stop


def test_after_stop_requires_fresh_signal():
    """Tras un stop no se recompra mientras el peso objetivo siga > 0: la señal debe apagarse y volver."""
    df = make_ohlcv(80, seed=21)
    df[["open", "close"]] = 100.0
    df["high"], df["low"] = 100.1, 99.9
    df.loc[10, ["open", "high", "low", "close"]] = [80.0, 81.0, 79.0, 80.0]   # gap: dispara el stop
    w = pd.Series(1.0, index=df["open_time"])
    w.iloc[50] = 0.0          # la señal se apaga una vela y vuelve a encenderse
    stops = pd.Series(0.02, index=df["open_time"])
    r = run_backtest({"X": _prices(df)}, {"X": w}, ZERO, {"X": Rules(1e-8, 0)}, 1000.0, stops={"X": stops},
                     stop_cooldown_bars=5)
    buys = r.fills[r.fills.side == "BUY"]["time"].tolist()
    assert len(buys) == 2                                   # entrada inicial + re-entrada con señal nueva
    assert buys[1] > df["open_time"].iloc[50]               # no recompró al terminar el enfriamiento (vela 15)


def test_paired_bootstrap_detects_and_rejects():
    from quant_ai.backtesting.metrics import paired_sharpe_diff_ci

    rng = np.random.default_rng(3)
    common = rng.normal(0, 0.01, 5000)
    better = common + 0.0008
    ci = paired_sharpe_diff_ci(better, common, 8760, block=168, n_boot=300)
    assert ci["lo"] > 0                                      # ventaja clara y persistente
    same = paired_sharpe_diff_ci(common + rng.normal(0, 0.002, 5000), common, 8760, block=168, n_boot=300)
    assert same["lo"] < 0 < same["hi"]                       # sin ventaja: el IC incluye 0


def test_psr_and_dsr_behave():
    from quant_ai.backtesting.metrics import deflated_sharpe, probabilistic_sharpe

    rng = np.random.default_rng(0)
    good = rng.normal(0.001, 0.01, 5000)   # SR por periodo 0.1
    noise = rng.normal(0.0, 0.01, 5000)
    assert probabilistic_sharpe(good) > 0.99
    assert 0.05 < probabilistic_sharpe(noise) < 0.95
    # elegir el mejor de 100 intentos de puro ruido no debe parecer significativo tras deflactar
    trials = [rng.normal(0, 0.01, 2000) for _ in range(100)]
    srs = [t.mean() / t.std(ddof=1) for t in trials]
    best = trials[int(np.argmax(srs))]
    assert probabilistic_sharpe(best) > 0.9          # sin deflactar parece bueno...
    assert deflated_sharpe(best, srs)["dsr"] < 0.9   # ...deflactado no lo es


def test_metrics_basic():
    eq = pd.Series(np.linspace(100, 110, 1000), index=pd.date_range("2024-01-01", periods=1000, freq="1h", tz="UTC"))
    m = performance_metrics(eq, ci=False)
    assert m["total_return"] == pytest.approx(0.10)
    assert m["max_drawdown"] == 0

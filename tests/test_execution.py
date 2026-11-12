"""Pruebas de la capa de ejecución (sin red): broker paper, bloqueo LIVE, kill switch y ciclo completo."""
import json
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from conftest import make_ohlcv
from quant_ai.backtesting.engine import CostModel, Rules, run_backtest
from quant_ai.brokers.base import Broker, BrokerError, LiveTradingBlocked, OrderResult, Quote
from quant_ai.brokers.factory import get_broker
from quant_ai.brokers.paper import LocalPaperBroker
from quant_ai.config import load_config
from quant_ai.decision_engine.policy import (PolicyParams, stop_guard_start, stop_guard_step,
                                             weights_from_probs)
from quant_ai.execution import runner
from quant_ai.execution.runner import run_cycle
from quant_ai.features.engineering import build_feature_set
from quant_ai.models.ml import EnsembleModel


def _quote(price):
    return lambda s: Quote(s, price * 0.9999, price * 1.0001)


def _broker(tmp_path, price=100.0, quote_fn=None):
    return LocalPaperBroker(tmp_path, 10_000, 10, 2, {"BTCUSDT": 1e-5, "ETHUSDT": 1e-4},
                            {"BTCUSDT": 5, "ETHUSDT": 5}, quote_fn=quote_fn or _quote(price))


def test_paper_broker_fill_costs_and_idempotency(tmp_path):
    b = _broker(tmp_path)
    r = b.place_market_order("BTCUSDT", "BUY", 10, "oid-1")
    assert r.status == "filled" and r.avg_price > 100.0  # paga ask + slippage
    pf = b.load()
    assert pf["cash"] == pytest.approx(10_000 - r.qty_filled * r.avg_price - r.fee)
    dup = b.place_market_order("BTCUSDT", "BUY", 10, "oid-1")
    assert dup.status == "duplicate"
    assert b.load()["positions"]["BTCUSDT"]["qty"] == pytest.approx(r.qty_filled)  # no se duplicó
    s = b.place_market_order("BTCUSDT", "SELL", 999, "oid-2")
    assert s.qty_filled == pytest.approx(r.qty_filled) and "BTCUSDT" not in b.load()["positions"]


def test_paper_broker_never_spends_more_than_cash(tmp_path):
    b = _broker(tmp_path)
    r = b.place_market_order("BTCUSDT", "BUY", 1e9, "big")
    assert b.load()["cash"] >= -1e-9 and r.qty_filled * r.avg_price <= 10_000


def test_live_mode_is_blocked(monkeypatch):
    monkeypatch.setenv("QUANT_AI_MODE", "live")
    with pytest.raises(LiveTradingBlocked):
        get_broker(load_config())


def _setup_home(tmp_path, monkeypatch, strong_signal: bool):
    """Datos sintéticos + modelo diminuto registrado en un QUANT_AI_HOME temporal."""
    monkeypatch.setenv("QUANT_AI_HOME", str(tmp_path))
    monkeypatch.delenv("QUANT_AI_MODE", raising=False)
    cfg = load_config(overrides={"compute": {"n_jobs": 1}})
    n = 2500
    now = pd.Timestamp.now(tz="UTC").floor("1h")
    start = now - pd.Timedelta(hours=n)
    raw = {}
    for i, s in enumerate(cfg.symbols):
        df = make_ohlcv(n, seed=10 + i, start=str(start.tz_localize(None)))
        raw[s] = df
        p = cfg.data_dir / "clean" / f"{s}_1h.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False)
    feats = build_feature_set(raw, cfg["features"])
    reg = cfg.models_dir / "registry" / "vtest"
    reg.mkdir(parents=True)
    for s, f in feats.items():
        f = f.dropna()
        y = np.ones(len(f)) if strong_signal else np.zeros(len(f))
        y[:20] = 1 - y[:20]  # ambas clases presentes
        m = EnsembleModel(list(f.columns), n_jobs=1)
        half = len(f) // 2
        m.fit(f.iloc[:half], y[:half], f.iloc[half:], y[half:])
        joblib.dump(m, reg / f"{s}.joblib")
    return cfg


def test_run_cycle_buys_with_strong_signal_and_logs(tmp_path, monkeypatch):
    cfg = _setup_home(tmp_path, monkeypatch, strong_signal=True)
    broker = _broker(cfg.state_dir, price=100.0)
    out = run_cycle(cfg, update_data=False, broker=broker, log=lambda *_: None)
    assert out["status"] == "ok"
    decisions = [json.loads(line) for line in (cfg.state_dir / "paper" / "decisions.jsonl").read_text("utf-8").splitlines()]
    assert len(decisions) == 2 and all(d["action"] == "BUY" for d in decisions)
    assert all(d["risk"]["approved"] and d["executed"] for d in decisions)
    assert all(d["probabilities"]["p_ensemble"] >= 0.55 for d in decisions)
    pf = broker.load()
    assert set(pf["positions"]) == set(cfg.symbols) and pf["model_status"] == "candidate"
    # repetir el ciclo en la misma vela no duplica órdenes (idempotencia)
    run_cycle(cfg, update_data=False, broker=broker, log=lambda *_: None)
    orders = (cfg.state_dir / "paper" / "orders.jsonl").read_text("utf-8").splitlines()
    assert sum(json.loads(o)["status"] == "filled" for o in orders) == 2
    hb = json.loads((cfg.state_dir / "heartbeat.json").read_text("utf-8"))
    assert hb["status"] == "ok"


def test_kill_switch_blocks_new_positions(tmp_path, monkeypatch):
    cfg = _setup_home(tmp_path, monkeypatch, strong_signal=True)
    (tmp_path / "KILL_SWITCH").write_text("alto")
    broker = _broker(cfg.state_dir, price=100.0)
    run_cycle(cfg, update_data=False, broker=broker, log=lambda *_: None)
    assert broker.load()["positions"] == {}
    d = json.loads((cfg.state_dir / "paper" / "decisions.jsonl").read_text("utf-8").splitlines()[0])
    assert d["risk"]["approved"] is False and any("KILL" in r for r in d["risk"]["reasons"])
    assert json.loads((cfg.state_dir / "heartbeat.json").read_text("utf-8"))["status"] == "kill_switch"


def test_abstains_with_weak_signal(tmp_path, monkeypatch):
    cfg = _setup_home(tmp_path, monkeypatch, strong_signal=False)
    broker = _broker(cfg.state_dir)
    run_cycle(cfg, update_data=False, broker=broker, log=lambda *_: None)
    assert broker.load()["positions"] == {}


def test_ood_uses_each_symbol_stats(tmp_path, monkeypatch):
    """El detector OOD debe comparar cada activo con SU distribución, no con la de otro."""
    cfg = _setup_home(tmp_path, monkeypatch, strong_signal=True)
    reg = cfg.models_dir / "registry" / "vtest"
    btc = joblib.load(reg / "BTCUSDT.joblib")
    # Estadísticos de BTC absurdos: si se aplicaran a ETH, ETH quedaría bloqueado por OOD
    btc.feature_stats = {"mean": {k: 1e6 for k in btc.feature_stats["mean"]},
                         "std": {k: 1e-3 for k in btc.feature_stats["std"]}, "n_train": 1}
    joblib.dump(btc, reg / "BTCUSDT.joblib")
    broker = _broker(cfg.state_dir, price=100.0)
    run_cycle(cfg, update_data=False, broker=broker, log=lambda *_: None)
    lines = (cfg.state_dir / "paper" / "decisions.jsonl").read_text("utf-8").splitlines()
    d = {json.loads(x)["symbol"]: json.loads(x) for x in lines}
    assert d["BTCUSDT"]["risk"]["approved"] is False and d["BTCUSDT"]["risk"]["checks"]["ood_features"]
    assert d["ETHUSDT"]["risk"]["approved"] is True and not d["ETHUSDT"]["risk"]["checks"]["ood_features"]


def test_paper_model_is_pinned_across_retrains(tmp_path, monkeypatch):
    """Un reentrenamiento nuevo no reemplaza en silencio el modelo con el que opera paper."""
    import shutil

    cfg = _setup_home(tmp_path, monkeypatch, strong_signal=True)
    broker = _broker(cfg.state_dir, price=100.0)
    run_cycle(cfg, update_data=False, broker=broker, log=lambda *_: None)
    assert (cfg.state_dir / "paper" / "MODEL").read_text("utf-8") == "vtest"
    shutil.copytree(cfg.models_dir / "registry" / "vtest", cfg.models_dir / "registry" / "vzzz_nuevo")
    run_cycle(cfg, update_data=False, broker=broker, log=lambda *_: None)
    assert broker.load()["model_version"] == "vtest"


def test_testnet_rounds_to_lot_step(monkeypatch):
    from quant_ai.brokers.binance_testnet import BinanceSpotTestnetBroker

    monkeypatch.setenv("BINANCE_TESTNET_API_KEY", "k")
    monkeypatch.setenv("BINANCE_TESTNET_API_SECRET", "s")
    b = BinanceSpotTestnetBroker(lot_steps={"BTCUSDT": 1e-5})
    assert b.round_qty("BTCUSDT", 0.0123456789) == pytest.approx(0.01234)
    assert "s" not in repr(b).replace("testnet", "").replace("BinanceSpot", "").split("key=")[1]


# ----------------------------------------------------------------------------- re-entrada tras stop-loss
class _ScriptedModel:
    """Modelo falso: la probabilidad de cada vela sale de un guion (sin entrenar nada)."""

    def __init__(self, probs: pd.Series):
        self.probs, self.feature_stats = probs, {}

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"p_ensemble": self.probs.reindex(X.index).to_numpy(), "uncertainty": 0.0},
                            index=X.index)


class _MemBroker(Broker):
    """Broker no local mínimo (saldos en memoria): el runner lleva su libro en state/paper/portfolio.json."""
    name = "mem"

    def __init__(self, px: dict):
        self.px, self.cash, self.qty, self.orders = px, 10_000.0, {}, {}

    def get_quote(self, symbol):
        return Quote(symbol, self.px[symbol] * 0.9999, self.px[symbol] * 1.0001)

    def get_balances(self):
        return {"cash": self.cash, "positions": dict(self.qty)}

    def get_order(self, client_order_id):
        return self.orders.get(client_order_id)

    def place_market_order(self, symbol, side, qty, client_order_id):
        q = self.get_quote(symbol)
        sign, px = (1, q.ask) if side == "BUY" else (-1, q.bid)
        qty = min(qty, self.cash / px) if side == "BUY" else min(qty, self.qty.get(symbol, 0.0))
        self.cash -= sign * qty * px
        self.qty[symbol] = self.qty.get(symbol, 0.0) + sign * qty
        if self.qty[symbol] <= 1e-12:
            del self.qty[symbol]
        self.orders[client_order_id] = OrderResult(client_order_id, symbol, side, qty, qty, px, 0.0, "filled",
                                                   self.name)
        return self.orders[client_order_id]


def _scripted_home(tmp_path, monkeypatch, closes: dict, probs: dict, cooldown: int = 3):
    """Ciclos de paper guionizados vela a vela, sin red ni entrenamiento.

    Precios continuos (apertura = cierre anterior; las caídas ocurren dentro de la vela), ATR del 1 % (stop a
    3 % de la entrada) y probabilidades fijadas por vela. Límites de pérdida y de exposición bruta relajados
    para que el motor de riesgo no se confunda con la regla de re-entrada. ``run(i)`` ejecuta el ciclo que
    decide la vela i un minuto después de su cierre, con cotización = cierre de i y, por defecto, un broker
    local NUEVO, de modo que todo estado entre ciclos sale del archivo de cartera.
    """
    monkeypatch.setenv("QUANT_AI_HOME", str(tmp_path))
    monkeypatch.delenv("QUANT_AI_MODE", raising=False)
    cfg = load_config(overrides={"compute": {"n_jobs": 1}, "strategy": {"stop_cooldown_bars": cooldown},
                                 "risk": {"max_daily_loss_pct": 0.9, "max_drawdown_pct": 0.9,
                                          "max_gross_exposure": 2.0}})
    n = len(next(iter(closes.values())))
    t = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    raw, feats, models = {}, {}, {}
    for s in cfg.symbols:
        c = np.asarray(closes.get(s, [100.0] * n), dtype=float)
        o = np.r_[c[0], c[:-1]]
        raw[s] = pd.DataFrame({"open_time": t, "open": o, "high": np.maximum(o, c) + 0.1,
                               "low": np.minimum(o, c) - 0.1, "close": c, "volume": 1e6,
                               "close_time": t + pd.Timedelta("1h") - pd.Timedelta("1ms")})
        feats[s] = pd.DataFrame({"atr_pct_14": 0.01, "vol_24": 0.01}, index=t)
        models[s] = _ScriptedModel(pd.Series(probs.get(s, [0.3] * n), index=t, dtype=float))
    clock, px = {}, {}
    monkeypatch.setattr(runner, "load_all",
                        lambda _c: {s: d[d["open_time"] <= clock["t"]] for s, d in raw.items()})
    monkeypatch.setattr(runner, "build_feature_set",
                        lambda r, _f: {s: feats[s].loc[:r[s]["open_time"].iloc[-1]] for s in r})
    monkeypatch.setattr(runner, "load_model", lambda _c: (models, "vtest", "candidate"))

    def run(i, broker=None):
        clock["t"] = t[i]
        px.update({s: raw[s]["close"].iloc[i] for s in raw})
        broker = broker or _broker(cfg.state_dir, quote_fn=lambda s: Quote(s, px[s] * 0.9999, px[s] * 1.0001))
        return run_cycle(cfg, update_data=False, broker=broker, now=t[i] + pd.Timedelta("61min"),
                         log=lambda *_: None)

    return SimpleNamespace(cfg=cfg, t=t, raw=raw, feats=feats, px=px, run=run)


def _bars(h, symbol):
    """Decisiones registradas en decisions.jsonl para ``symbol`` como lista de (índice de vela, decisión)."""
    lines = (h.cfg.state_dir / "paper" / "decisions.jsonl").read_text("utf-8").splitlines()
    out = [json.loads(x) for x in lines]
    return [(h.t.get_loc(pd.Timestamp(d["bar_time"])), d) for d in out if d["symbol"] == symbol]


def _portfolio(h):
    return json.loads((h.cfg.state_dir / "paper" / "portfolio.json").read_text("utf-8"))


def test_stop_guard_step_rules():
    """Función pura: límite del enfriamiento y qué levanta (y qué no) la espera de señal nueva."""
    p, b, h = PolicyParams(), pd.Timestamp("2024-01-01 10:00", tz="UTC"), pd.Timedelta("1h")
    g = stop_guard_start(b, 0.60, p, 3 * h)
    assert g == {"stop_bar": b.isoformat(), "cooldown_until": (b + 3 * h).isoformat(), "need_fresh": True}
    assert stop_guard_start(b, 0.44, p, 3 * h)["need_fresh"] is False   # señal ya apagada en la vela b
    g2, why = stop_guard_step(g, b + 2 * h, 0.50, p)     # banda de histéresis: la señal sigue encendida
    assert g2["need_fresh"] and len(why) == 2 and why[0].startswith("enfriamiento")
    g3, why = stop_guard_step(g, b + 3 * h, 0.60, p)     # b + N: ya no hay enfriamiento
    assert why == ["esperando señal nueva tras stop-loss"] and g3["need_fresh"]
    assert stop_guard_step(g, b + 3 * h, np.nan, p) == (None, [])
    assert stop_guard_step(g, b + 3 * h, 0.449, p) == (None, [])
    assert stop_guard_step(None, b, 0.9, p) == (None, [])


def test_no_rebuy_after_stop_while_signal_stays_on(tmp_path, monkeypatch):
    """Tras un stop ejecutado y con señal fuerte sostenida no se recompra: ni durante el enfriamiento (ni
    reejecutando el ciclo en la misma vela) ni después de él, porque falta una señal nueva."""
    n = 12
    h = _scripted_home(tmp_path, monkeypatch, closes={"BTCUSDT": [100.0] * 2 + [95.0] * (n - 2)},
                       probs={"BTCUSDT": [0.3] + [0.6] * (n - 1)}, cooldown=3)
    for i in range(n):
        h.run(i)
        if i == 2:
            h.run(i)  # reejecución en la vela del stop: antes del cambio aquí se recompraba
    btc = _bars(h, "BTCUSDT")
    assert [(i, d["action"], d["executed"]) for i, d in btc[:3]] == [(0, "HOLD", False), (1, "BUY", True),
                                                                       (2, "CLOSE", True)]
    assert btc[2][1]["reasons"][0].startswith("stop-loss")
    blocked = btc[3:]
    assert [i for i, _ in blocked] == [2] + list(range(3, n))
    for i, d in blocked:
        assert d["action"] == "HOLD" and not d["executed"] and d["size_qty"] == 0 and d["stop_loss"] is None
        assert d["probabilities"]["p_ensemble"] >= 0.55             # la política habría comprado
        assert "esperando señal nueva tras stop-loss" in d["reasons"]
        cooling = [r for r in d["reasons"] if r.startswith("enfriamiento")]
        assert cooling == (["enfriamiento tras stop-loss hasta la vela 2024-01-01 05:00 UTC"] if i < 5 else
                           [])                                       # stop en la vela 2, N = 3
    orders = (h.cfg.state_dir / "paper" / "orders.jsonl").read_text("utf-8").splitlines()
    orders = [json.loads(x) for x in orders]
    assert [(o["symbol"], o["side"]) for o in orders] == [("BTCUSDT", "BUY"), ("BTCUSDT", "SELL")]
    assert _portfolio(h)["stop_guard"]["BTCUSDT"]["need_fresh"] is True


@pytest.mark.parametrize("off", [0.40, np.nan], ids=["p_bajo_salida", "prediccion_invalida"])
def test_rebuys_after_signal_resets_and_cooldown_ends(tmp_path, monkeypatch, off):
    """Si la señal se apaga (p < exit_prob o predicción inválida) y vuelve a encenderse, se recompra en
    cuanto termina el enfriamiento, y no antes."""
    # vela: 0 nada | 1 compra | 2 stop | 3 señal apagada | 4 fuerte en enfriamiento | 5 recompra | 6 mantener
    h = _scripted_home(tmp_path, monkeypatch, closes={"BTCUSDT": [100.0, 100.0] + [95.0] * 5},
                       probs={"BTCUSDT": [0.3, 0.6, 0.6, off, 0.6, 0.6, 0.5]}, cooldown=3)
    for i in range(4):
        h.run(i)
    assert _portfolio(h)["stop_guard"]["BTCUSDT"]["need_fresh"] is False
    for i in range(4, 7):
        h.run(i)
    btc = dict(_bars(h, "BTCUSDT"))
    assert btc[2]["action"] == "CLOSE" and btc[2]["executed"]
    assert btc[4]["action"] == "HOLD" and btc[4]["reasons"][-1].startswith("enfriamiento tras stop-loss")
    assert not any("esperando" in r for r in btc[4]["reasons"])
    assert btc[5]["action"] == "BUY" and btc[5]["executed"] and btc[5]["risk"]["approved"]
    assert btc[6]["action"] == "HOLD"
    pf = _portfolio(h)
    assert pf["positions"]["BTCUSDT"]["stop_price"] and pf["stop_guard"] == {}


def test_stop_guard_lives_in_portfolio_file_and_is_per_symbol(tmp_path, monkeypatch):
    """El estado se lee del archivo de cartera (broker nuevo en cada ciclo) y no afecta a un símbolo sin
    stop."""
    h = _scripted_home(tmp_path, monkeypatch,
                       closes={"BTCUSDT": [100.0, 100.0, 95.0, 95.0, 95.0], "ETHUSDT": [100.0] * 5},
                       probs={"BTCUSDT": [0.3, 0.6, 0.6, 0.6, 0.6], "ETHUSDT": [0.3, 0.3, 0.3, 0.6, 0.6]},
                       cooldown=24)
    for i in range(4):
        h.run(i)
    pf = _portfolio(h)
    assert pf["stop_guard"] == {"BTCUSDT": {"stop_bar": h.t[2].isoformat(), "need_fresh": True,
                                            "cooldown_until": (h.t[2] + pd.Timedelta(hours=24)).isoformat()}}
    btc, eth = dict(_bars(h, "BTCUSDT")), dict(_bars(h, "ETHUSDT"))
    assert btc[3]["action"] == "HOLD" and len(btc[3]["reasons"]) == 3
    assert eth[3]["action"] == "BUY" and eth[3]["executed"] and set(pf["positions"]) == {"ETHUSDT"}
    # Sin el estado en el archivo el ciclo siguiente recompra: el bloqueo viene del archivo, no de la memoria
    del pf["stop_guard"]["BTCUSDT"]
    (h.cfg.state_dir / "paper" / "portfolio.json").write_text(json.dumps(pf), "utf-8")
    h.run(4)
    btc = dict(_bars(h, "BTCUSDT"))
    assert btc[4]["action"] == "BUY" and btc[4]["executed"]


def test_stop_guard_with_non_local_broker(tmp_path, monkeypatch):
    """Con un broker no local el estado se guarda en el libro de ``Book`` y sobrevive a la reconciliación."""
    h = _scripted_home(tmp_path, monkeypatch, closes={"BTCUSDT": [100.0, 100.0] + [95.0] * 4},
                       probs={"BTCUSDT": [0.3] + [0.6] * 5}, cooldown=2)
    b = _MemBroker(h.px)
    for i in range(6):
        h.run(i, broker=b)
    btc = dict(_bars(h, "BTCUSDT"))
    assert [btc[i]["action"] for i in range(6)] == ["HOLD", "BUY", "CLOSE", "HOLD", "HOLD", "HOLD"]
    assert btc[2]["executed"] and btc[2]["reasons"][0].startswith("stop-loss")
    assert btc[5]["reasons"][-1] == "esperando señal nueva tras stop-loss" and "BTCUSDT" not in b.qty
    assert _portfolio(h)["stop_guard"]["BTCUSDT"]["need_fresh"] is True


@pytest.mark.parametrize("p_stop, back_at, action, last_reason", [
    (0.40, 5, "BUY", None),
    (0.40, 4, "HOLD", "enfriamiento tras stop-loss hasta la vela 2024-01-01 05:00 UTC"),
    (0.60, 6, "HOLD", "esperando señal nueva tras stop-loss"),
], ids=["fin_enfriamiento", "aun_en_enfriamiento", "velas_no_vistas_no_levantan_espera"])
def test_cooldown_counts_bars_not_cycles(tmp_path, monkeypatch, p_stop, back_at, action, last_reason):
    """Con el proceso apagado tras el stop (vela 2, N = 3), el enfriamiento se mide en tiempo de vela y no en
    ciclos ejecutados. Una señal apagada en velas que el proceso no decidió no levanta la espera
    (conservador)."""
    probs = [0.3, 0.6, p_stop, 0.4, 0.6, 0.6, 0.6]   # vela 3: señal apagada que el proceso apagado no ve
    h = _scripted_home(tmp_path, monkeypatch, closes={"BTCUSDT": [100.0, 100.0] + [95.0] * 5},
                       probs={"BTCUSDT": probs}, cooldown=3)
    for i in (0, 1, 2, back_at):
        h.run(i)
    btc = dict(_bars(h, "BTCUSDT"))
    assert btc[2]["action"] == "CLOSE" and btc[2]["executed"]
    assert btc[back_at]["action"] == action and btc[back_at]["executed"] == (action == "BUY")
    assert last_reason is None or btc[back_at]["reasons"][-1] == last_reason


def test_stop_guard_survives_api_error_right_after_stop(tmp_path, monkeypatch):
    """Broker no local: si la API falla al reconciliar justo después de la venta por stop, el ciclo aborta,
    pero el bloqueo ya quedó guardado y el ciclo siguiente no recompra."""
    h = _scripted_home(tmp_path, monkeypatch, closes={"BTCUSDT": [100.0, 100.0, 95.0, 95.0]},
                       probs={"BTCUSDT": [0.3, 0.6, 0.6, 0.6]}, cooldown=3)
    b = _MemBroker(h.px)
    h.run(0, broker=b)
    h.run(1, broker=b)

    def fail():
        raise BrokerError("timeout al reconciliar")

    def sell_then_fail(*args):
        res = _MemBroker.place_market_order(b, *args)
        b.get_balances = fail
        return res

    b.place_market_order = sell_then_fail
    assert h.run(2, broker=b)["status"] == "error" and "BTCUSDT" not in b.qty
    del b.get_balances, b.place_market_order                   # la API se recupera
    assert _portfolio(h)["stop_guard"]["BTCUSDT"]["stop_bar"] == h.t[2].isoformat()
    h.run(3, broker=b)
    d = dict(_bars(h, "BTCUSDT"))[3]
    assert d["action"] == "HOLD" and "esperando señal nueva tras stop-loss" in d["reasons"]


def test_stop_reentry_parity_with_backtest(tmp_path, monkeypatch):
    """Paridad a pequeña escala: la misma secuencia de probabilidades y precios da los mismos stops y las
    mismas entradas en ``run_backtest`` (política vectorial con histéresis) y en ciclos sucesivos de
    ``run_cycle``.

    Cubre: señal sostenida tras el stop (bloquea más allá del enfriamiento), espera levantada con
    p < exit_prob y con predicción inválida, stop en una vela cuya señal ya está apagada y el límite exacto
    b + N. No cubre: caídas que solo perforan el stop por un instante (el backtest mira el mínimo de la vela,
    el ciclo solo el bid del momento), saltos entre el cierre de una vela y la apertura de la siguiente, ni la
    recompra del backtest con p en la banda de histéresis cuando su política vuelve a "entrar" durante el
    enfriamiento.
    """
    nan, n_cool = np.nan, 3
    probs = [0.3, 0.6, 0.5, 0.5, 0.5,           # 1: entrada
             0.6, 0.6, 0.6, 0.6, 0.6, 0.6,      # 5: stop con señal encendida; 6-10 bloqueadas
             0.4, 0.6, 0.5, 0.5, 0.5,           # 11: señal apagada; 12: re-entrada
             0.4, 0.6, 0.6, 0.6, 0.5, 0.5,      # 16: stop con señal apagada; 17-18 enfriamiento; 19 = 16 + N
             0.6, nan, 0.6, 0.6, 0.5]           # 22: stop; 23: predicción inválida; 25 = 22 + N
    closes = [100.0] * 5 + [95.0] * 11 + [90.0] * 6 + [85.0] * 5
    h = _scripted_home(tmp_path, monkeypatch, closes={"BTCUSDT": closes}, probs={"BTCUSDT": probs},
                       cooldown=n_cool)
    for i in range(len(closes)):
        h.run(i)
    live = _bars(h, "BTCUSDT")
    live_entries = [h.t[i] for i, d in live if d["action"] == "BUY" and d["executed"]]
    live_stops = [h.t[i] for i, d in live if d["action"] == "CLOSE" and d["executed"]
                  and d["reasons"][0].startswith("stop-loss")]

    w = weights_from_probs(pd.Series(probs, index=h.t), None, PolicyParams.from_config(h.cfg["strategy"]))
    prices = h.raw["BTCUSDT"][["open_time", "open", "high", "low", "close", "volume"]]
    stops = h.cfg["strategy"]["atr_stop_mult"] * h.feats["BTCUSDT"]["atr_pct_14"]
    res = run_backtest({"BTCUSDT": prices}, {"BTCUSDT": w}, CostModel(10, 1, 2, 1e9),
                       {"BTCUSDT": Rules(1e-8, 0)}, 10_000.0, stops={"BTCUSDT": stops},
                       stop_cooldown_bars=n_cool)
    # el backtest ejecuta en la apertura siguiente lo decidido al cierre: la vela decidida es la anterior
    bt_entries = [t - pd.Timedelta("1h") for t in res.fills.loc[res.fills.side == "BUY", "time"]]
    bt_stops = list(res.fills.loc[res.fills.reason == "stop_loss", "time"])

    assert bt_stops == live_stops == [h.t[5], h.t[16], h.t[22]]
    assert bt_entries == live_entries == [h.t[1], h.t[12], h.t[19], h.t[25]]
    blocked = [i for i, d in live if any("tras stop-loss" in r for r in d["reasons"])]
    assert blocked == [6, 7, 8, 9, 10, 17, 18, 24]

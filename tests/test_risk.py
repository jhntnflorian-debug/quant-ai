import pandas as pd

from quant_ai.decision_engine.policy import PolicyParams, step
from quant_ai.risk_management.engine import MarketState, OrderProposal, PortfolioState, RiskEngine, RiskLimits

NOW = pd.Timestamp("2026-01-01 12:05", tz="UTC")


def _mk(tmp_path, **lim):
    return RiskEngine(RiskLimits(**lim), feature_stats={"mean": {"a": 0.0}, "std": {"a": 1.0}}, root=tmp_path)


def _order(side="BUY", notional=1000, w=0.5, oid="o1"):
    return OrderProposal("BTCUSDT", side, notional, w, oid, 50000)


def _mkt(age_min=10, vol=0.01, feats=None):
    return MarketState(NOW - pd.Timedelta(minutes=age_min), vol, feats)


def _pf(eq=10000, peak=10000, day=10000, w=None, gross=0.0):
    return PortfolioState(eq, peak, day, w or {}, gross)


def test_approves_normal_order(tmp_path):
    d = _mk(tmp_path).evaluate(_order(), _mkt(), _pf(), NOW)
    assert d.approved, d.reasons


def test_kill_switch_blocks_buys_but_allows_sells(tmp_path):
    (tmp_path / "KILL_SWITCH").write_text("stop")
    eng = _mk(tmp_path)
    assert not eng.evaluate(_order(), _mkt(), _pf(), NOW).approved
    assert eng.evaluate(_order("SELL", oid="s1"), _mkt(), _pf(w={"BTCUSDT": 0.5}, gross=0.5), NOW).approved


def test_stale_data_blocks(tmp_path):
    d = _mk(tmp_path).evaluate(_order(), _mkt(age_min=500), _pf(), NOW)
    assert not d.approved and any("obsoletos" in r for r in d.reasons)


def test_limits_block(tmp_path):
    eng = _mk(tmp_path, max_order_notional=500)
    assert not eng.evaluate(_order(notional=1000, oid="a"), _mkt(), _pf(), NOW).approved
    assert not eng.evaluate(_order(w=0.8, oid="b"), _mkt(), _pf(), NOW).approved           # peso por activo
    assert not eng.evaluate(_order(notional=100, w=0.5, oid="c"), _mkt(), _pf(gross=0.7), NOW).approved  # apalancamiento
    assert not eng.evaluate(_order(notional=100, oid="d"), _mkt(vol=0.09), _pf(), NOW).approved
    assert not eng.evaluate(_order(notional=100, oid="e"), _mkt(), _pf(eq=9600, day=10000), NOW).approved
    pct = _mk(tmp_path, max_order_pct_equity=0.3)
    assert not pct.evaluate(_order(notional=4000, oid="f"), _mkt(), _pf(), NOW).approved   # 40 % > 30 %


def test_drawdown_triggers_emergency(tmp_path):
    eng = _mk(tmp_path)
    d = eng.evaluate(_order(notional=100), _mkt(), _pf(eq=8000, peak=10000, day=8000), NOW)
    assert not d.approved and eng.emergency_reason
    # sigue bloqueado aunque el drawdown se recupere, hasta revisión manual
    d2 = eng.evaluate(_order(notional=100, oid="o2"), _mkt(), _pf(), NOW)
    assert not d2.approved
    eng.clear_emergency()
    assert eng.evaluate(_order(notional=100, oid="o3"), _mkt(), _pf(), NOW).approved


def test_duplicate_order_rejected(tmp_path):
    eng = _mk(tmp_path)
    assert eng.evaluate(_order(notional=100), _mkt(), _pf(), NOW).approved
    assert not eng.evaluate(_order(notional=100), _mkt(), _pf(), NOW).approved


def test_ood_blocks(tmp_path):
    eng = RiskEngine(RiskLimits(ood_max_features=1), feature_stats={"mean": {"a": 0}, "std": {"a": 1}}, root=tmp_path)
    assert not eng.evaluate(_order(notional=100), _mkt(feats={"a": 50.0}), _pf(), NOW).approved


def test_api_errors_trigger_emergency(tmp_path):
    eng = _mk(tmp_path, max_api_errors=2)
    eng.record_api_error()
    eng.record_api_error()
    assert not eng.evaluate(_order(notional=100), _mkt(), _pf(), NOW).approved


def test_policy_long_flat_and_abstention():
    p = PolicyParams(0.55, 0.45, 0.15, 0.5)
    assert step(0.6, 0.05, False, p)[0] == "BUY"
    assert step(0.6, 0.30, False, p)[:2] == ("HOLD", 0.0)       # abstención por incertidumbre
    assert step(0.5, 0.05, True, p)[:2] == ("HOLD", 0.5)        # histéresis
    assert step(0.4, 0.05, True, p)[:2] == ("SELL", 0.0)        # salir a efectivo, nunca corto
    assert step(0.3, 0.05, False, p)[1] == 0.0

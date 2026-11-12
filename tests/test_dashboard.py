"""Pruebas de humo del dashboard con streamlit.testing.v1.AppTest.

Se comprueba que cada página carga sin excepciones (a) con los artefactos reales del
proyecto y (b) con un QUANT_AI_HOME vacío, donde deben salir avisos de "sin datos".
Los ficheros de `test_paper_artifacts_render` son fixtures de prueba en tmp_path: solo
ejercitan el código de las páginas de paper y nunca se muestran fuera de los tests.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT / "dashboard"
PAGES = ["principal", "entrenamiento", "decisiones", "rendimiento", "datos", "arquitectura"]
FOOTER = ("Resultados simulados; no son beneficios reales. Modo PAPER. "
          "Operación real bloqueada sin autorización expresa.")


def run_page(name: str) -> AppTest:
    return AppTest.from_file(str(DASH / "app_pages" / f"{name}.py"), default_timeout=90).run()


def texts(at: AppTest) -> str:
    parts = [w.value for w in at.warning] + [w.value for w in at.info] + [w.value for w in at.error]
    parts += [w.value for w in at.caption] + [w.value for w in at.markdown]
    return "\n".join(map(str, parts))


@pytest.fixture
def real_home(monkeypatch):
    monkeypatch.delenv("QUANT_AI_HOME", raising=False)


@pytest.fixture
def empty_home(monkeypatch, tmp_path):
    monkeypatch.setenv("QUANT_AI_HOME", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize("page", PAGES)
def test_page_real_artifacts(real_home, page):
    at = run_page(page)
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("page", PAGES)
def test_page_empty_home_shows_warning(empty_home, page):
    at = run_page(page)
    assert not at.exception, [e.value for e in at.exception]
    assert any("Sin datos" in w.value or "Sin modelo" in w.value for w in list(at.warning) + list(at.info)), \
        f"{page}: se esperaba un aviso de 'sin datos'"
    assert not (empty_home / "state").exists(), "el dashboard no debe crear carpetas"


def test_empty_home_gives_commands(empty_home):
    at = run_page("rendimiento")
    assert any("run_experiments.py" in c.value for c in at.code)
    at = run_page("principal")
    assert any("run_paper.py" in c.value or "download_data.py" in c.value for c in at.code)


def test_entrypoint_has_footer(real_home):
    at = AppTest.from_file(str(DASH / "app.py"), default_timeout=90).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(FOOTER in c.value for c in at.caption)


def test_entrypoint_footer_with_stop(empty_home):
    """Rendimiento termina con st.stop() en un home vacío; el pie debe seguir apareciendo."""
    at = AppTest.from_file(str(DASH / "app.py"), default_timeout=90).run()
    at.switch_page("app_pages/rendimiento.py").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(FOOTER in c.value for c in at.caption)


def test_smoke_run_is_flagged(real_home):
    runs = ROOT / "reports" / "runs"
    smoke = [d.name for d in runs.glob("*_smoke")] if runs.exists() else []
    if not smoke:
        pytest.skip("no hay corrida smoke real en disco")
    at = run_page("rendimiento")
    at.selectbox(key="perf_run").set_value(smoke[0]).run()
    assert not at.exception
    assert any("PRUEBA MÍNIMA" in e.value for e in at.error)


def test_paper_artifacts_render(empty_home):
    """Fixtures mínimos en tmp: ejercita principal, decisiones y rendimiento con paper presente."""
    (empty_home / "data" / "clean").mkdir(parents=True)
    (empty_home / "state" / "paper").mkdir(parents=True)
    n = 400
    t = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    close = 100 + np.cumsum(np.ones(n) * 0.1)
    pd.DataFrame({"open_time": t, "open": close, "high": close + 1, "low": close - 1, "close": close,
                  "volume": 1.0}).to_parquet(empty_home / "data" / "clean" / "TESTUSDT_1h.parquet")
    bar = t[200]
    (empty_home / "state" / "paper" / "portfolio.json").write_text(json.dumps({
        "mode": "paper", "cash": 900.0, "positions": {"TESTUSDT": {"qty": 1.0, "avg_price": 100.0,
        "stop_price": 95.0, "opened_at": str(bar)}}, "equity": 1010.0, "peak_equity": 1020.0,
        "day_start_equity": 1000.0, "initial_cash": 1000.0, "model_version": "v0", "model_status": "candidate"}))
    dec = {"decision_id": "d1", "created_at": str(bar), "bar_time": str(bar), "symbol": "TESTUSDT",
           "action": "BUY", "direction": "long", "horizon_bars": 24, "target_weight": 0.1,
           "current_weight": 0.0, "ref_price": float(close[200]), "stop_loss": 95.0,
           "probabilities": {"p_logistic": 0.6, "p_lightgbm": 0.62, "p_ensemble": 0.61},
           "uncertainty": 0.02, "regime": {"trend": "alcista", "vol": "baja"}, "reasons": ["prueba"],
           "top_features": [{"feature": "rsi", "value": 55.0, "z": 0.4}],
           "risk": {"approved": True, "reasons": ["ok"], "checks": {"max_weight": True}},
           "executed": True, "fill": {"price": 100.1, "qty": 1.0, "fee": 0.1}, "model_version": "v0"}
    (empty_home / "state" / "paper" / "decisions.jsonl").write_text(json.dumps(dec) + "\n")
    (empty_home / "state" / "heartbeat.json").write_text(json.dumps({
        "last_cycle_at": str(bar), "status": "ok", "message": "", "last_bar": {"TESTUSDT": str(bar)},
        "cycle_seconds": 1.2}))
    for page in ("principal", "decisiones", "rendimiento"):
        at = run_page(page)
        assert not at.exception, (page, [e.value for e in at.exception])

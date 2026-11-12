"""Actualización de datos sin red: un fallo de la API REST no debe detener la preparación del entrenamiento."""
import pandas as pd
import pytest
import requests

from conftest import make_ohlcv
from quant_ai.config import load_config
from quant_ai.data import binance, store


def _home_with_data(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANT_AI_HOME", str(tmp_path))
    cfg = load_config()
    for i, s in enumerate(cfg.symbols):
        p = store.clean_path(cfg, s)
        p.parent.mkdir(parents=True, exist_ok=True)
        make_ohlcv(500, seed=i).to_parquet(p, index=False)
    monkeypatch.setattr(binance, "month_range", lambda *a, **k: [])  # sin descargas mensuales

    def blocked(*a, **k):
        raise requests.HTTPError("451 Client Error: restricted location")
    monkeypatch.setattr(binance, "fetch_rest_klines", blocked)
    return cfg


def test_rest_failure_is_non_fatal_for_training(tmp_path, monkeypatch):
    cfg = _home_with_data(tmp_path, monkeypatch)
    logs = []
    res = store.update_all(cfg, use_rest=True, log=logs.append, rest_best_effort=True)
    for s in cfg.symbols:
        assert res[s]["rows"] == 500 and res[s]["rest_rows_added"] == 0
        assert "451" in res[s]["rest_error"]
        assert len(pd.read_parquet(store.clean_path(cfg, s))) == 500   # los datos guardados no se pierden
    assert any("AVISO" in m for m in logs)


def test_rest_failure_stays_fatal_by_default(tmp_path, monkeypatch):
    """El ciclo de paper usa el valor por defecto: sin datos al día no se debe seguir en silencio."""
    cfg = _home_with_data(tmp_path, monkeypatch)
    with pytest.raises(requests.HTTPError):
        store.update_all(cfg, use_rest=True, log=lambda *_: None)


def test_rest_failure_without_any_data_is_fatal(tmp_path, monkeypatch):
    cfg = _home_with_data(tmp_path, monkeypatch)
    for s in cfg.symbols:
        store.clean_path(cfg, s).unlink()
    with pytest.raises(requests.HTTPError):
        store.update_all(cfg, use_rest=True, log=lambda *_: None, rest_best_effort=True)

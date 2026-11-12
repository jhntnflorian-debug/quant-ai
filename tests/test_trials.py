"""Registro de intentos del DSR: las pruebas mínimas cuentan como intento pero no aportan Sharpe."""
import json

from quant_ai.config import load_config
from quant_ai.validation.trials import collect_trials


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _run(home, run_id, strategies, smoke=False, holdout=False):
    _write(home / "reports" / "runs" / run_id / "summary.json",
           {"run_id": run_id, "is_smoke": smoke, "final_holdout_used": holdout,
            "strategies": {k: {"sharpe": v} for k, v in strategies.items()}})


def test_sharpe_de_la_corrida_completa_mas_reciente(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANT_AI_HOME", str(tmp_path))
    _run(tmp_path, "20260101T000000Z_smoke", {"ml_x": 1.6, "solo_smoke": -2.8, "cero": 0.9}, smoke=True)
    _run(tmp_path, "20260102T000000Z", {"ml_x": 0.5, "ml_x_cost2x": 0.1, "cero": 0.0})
    _run(tmp_path, "20260103T000000Z", {"ml_x": 0.7})
    _run(tmp_path, "20260104T000000Z_smoke", {"ml_x": 3.0}, smoke=True)
    _run(tmp_path, "20260105T000000Z_final", {"ml_x": 9.9, "en_holdout": 1.0}, holdout=True)
    trials = collect_trials(load_config())
    x = trials["estrategia:ml_x"]
    assert x["sharpe_annual"] == 0.7 and x["sharpe_source"] == "20260103T000000Z"
    assert x["sources"] == ["20260101T000000Z_smoke", "20260102T000000Z", "20260103T000000Z",
                            "20260104T000000Z_smoke"]
    solo = trials["estrategia:solo_smoke"]          # se cuenta como intento, sin Sharpe
    assert solo["sharpe_annual"] is None and solo["sharpe_source"] is None
    assert trials["estrategia:cero"]["sharpe_annual"] == 0.0     # un 0.0 legítimo no se pierde
    assert "estrategia:ml_x_cost2x" not in trials and "estrategia:en_holdout" not in trials
    assert len(trials) == 3


def test_carpetas_dl_y_rl_smoke_no_duplican_identidad(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANT_AI_HOME", str(tmp_path))
    dl = tmp_path / "models" / "dl"
    _run(tmp_path, "20260101T000000Z", {"buy_and_hold": 1.1})
    for rel, sharpe in [("chart_cnn_w48_smoke", 2.5), ("secuencias_v2_smoke/gru", -3.0),
                        ("secuencias_v2/gru", 0.4), ("secuencias_v1_smoke/tcn", 5.0)]:
        name = "dl_chart_cnn" if "chart" in rel else "dl_" + rel.rsplit("/", 1)[-1]
        _write(dl / rel / "backtest_summary.json",
               {"strategies": {name: {"sharpe": sharpe}, "buy_and_hold": {"sharpe": 7.0}}})
    # Carpeta smoke sin sufijo en el nombre: se reconoce por run_info.json.
    _write(dl / "chart_cnn_w96" / "backtest_summary.json", {"strategies": {"dl_chart_cnn": {"sharpe": 1.9}}})
    _write(dl / "chart_cnn_w96" / "run_info.json", {"smoke": True})
    _write(tmp_path / "models" / "rl" / "rl_v1_ppo_BTCUSDT_smoke" / "eval_summary.json",
           {"seeds": {"0": {"metrics": {"sharpe": 4.0}}}})
    _write(tmp_path / "models" / "rl" / "rl_v1_ppo_BTCUSDT" / "eval_summary.json",
           {"seeds": {"0": {"metrics": {"sharpe": -1.0}}}})
    trials = collect_trials(load_config())
    assert set(trials) == {"estrategia:buy_and_hold", "estrategia:dl_chart_cnn_w48", "estrategia:dl_v2_gru",
                           "estrategia:secuencias_v1/tcn:dl_tcn", "estrategia:dl_chart_cnn_w96",
                           "rl:rl_v1_ppo_BTCUSDT:seed0"}
    assert trials["estrategia:dl_chart_cnn_w48"]["sharpe_annual"] is None
    assert trials["estrategia:dl_chart_cnn_w96"]["sharpe_annual"] is None
    assert trials["estrategia:secuencias_v1/tcn:dl_tcn"]["sharpe_annual"] is None
    gru = trials["estrategia:dl_v2_gru"]
    assert gru["sharpe_annual"] == 0.4 and gru["sharpe_source"] == "secuencias_v2/gru"
    assert set(gru["sources"]) == {"secuencias_v2/gru", "secuencias_v2_smoke/gru"}
    # Las referencias de un backtest DL no pisan el valor de la corrida completa.
    assert trials["estrategia:buy_and_hold"]["sharpe_annual"] == 1.1
    rl = trials["rl:rl_v1_ppo_BTCUSDT:seed0"]
    assert rl["sharpe_annual"] == -1.0
    assert set(rl["sources"]) == {"rl_v1_ppo_BTCUSDT", "rl_v1_ppo_BTCUSDT_smoke"}

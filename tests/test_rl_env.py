"""Contratos de RL; se omiten sin instalar dependencias opcionales localmente."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gymnasium")
from gymnasium.utils.env_checker import check_env

from quant_ai.backtesting.engine import CostModel, Rules, run_backtest
from quant_ai.reinforcement_learning.env import TradingEnv


def market(n=80):
    index = pd.date_range("2023-12-30", periods=n, freq="1h", tz="UTC")
    close = 100 * np.exp(np.arange(n) * 0.001)
    prices = pd.DataFrame({"open_time": index, "open": np.r_[100, close[:-1]],
                           "high": close * 1.01, "low": close * 0.99,
                           "close": close, "volume": 100_000.0})
    features = pd.DataFrame({"trend": np.arange(n, dtype=float), "constant": 1.0}, index=index)
    return features, prices


def environment(features=None, prices=None, **kwargs):
    f, p = market()
    f = f if features is None else features
    p = p if prices is None else prices
    if kwargs.get("training"):
        normalization = TradingEnv.fit_normalization(f, f.index[30], f.index[-1] + pd.Timedelta(hours=1))
    else:
        normalization = TradingEnv.fit_normalization(f, f.index[0], f.index[30])
    return TradingEnv(f, p, f.index[30], f.index[-1] + pd.Timedelta(hours=1),
                      normalization=normalization, **kwargs)


def test_training_episodes_never_cross_data_gaps():
    """Binance tiene huecos reales: entrenamiento los admite, pero ningún episodio los cruza."""
    f, p = market(200)
    keep = np.ones(len(f), dtype=bool)
    keep[100:103] = False  # hueco de 3 velas
    f, p = f[keep], p[keep]
    norm = TradingEnv.fit_normalization(f, f.index[0], f.index[-1] + pd.Timedelta(hours=1))
    env = TradingEnv(f, p, f.index[0], f.index[-1] + pd.Timedelta(hours=1), normalization=norm,
                     training=True, episode_length=20)
    for seed in range(30):
        env.reset(seed=seed)
        start = env.i
        span = env.index[start + 20] - env.index[start]
        assert span == pd.Timedelta(hours=20)
    with pytest.raises(ValueError):  # evaluación sigue exigiendo continuidad
        TradingEnv(f, p, f.index[0], f.index[-1] + pd.Timedelta(hours=1),
                   normalization={**norm, "train_end": str(f.index[0])}, training=False)


def test_gymnasium_contract():
    check_env(environment(), skip_render_check=True)
    check_env(environment(training=True, episode_length=12), skip_render_check=True)


def test_cash_action_has_no_cost():
    env = environment()
    env.reset(seed=0)
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(0)
        assert reward == 0
        assert info["costs_paid"] == 0
        assert info["operations"] == 0
        assert info["equity"] == env.initial_cash
        assert env.observation_space.contains(obs)


@pytest.mark.parametrize("max_weight", [0.5, 1.0])
def test_buy_hold_matches_price_return_less_entry_costs(max_weight):
    # Banda amplia: acción 2 mantiene las unidades compradas sin rebalancear.
    env = environment(max_weight=max_weight, rebalance_band=1.0)
    _, initial = env.reset(seed=0)
    _, _, _, _, entry = env.step(2)
    qty, cash = env.qty, env.cash
    while not env.done:
        _, _, _, _, last = env.step(2)
    expected = cash + qty * env.close_prices[-1]
    assert last["equity"] == pytest.approx(expected)
    assert last["operations"] == 1
    price_pnl = qty * (env.close_prices[-1] - env.open_prices[1])
    assert last["equity"] - initial["equity"] == pytest.approx(price_pnl - entry["costs_paid"])


def test_future_prices_do_not_change_current_observation_or_order_size():
    features, prices = market()
    altered = prices.copy()
    altered.loc[35:, ["open", "high", "low", "close"]] *= 3
    altered.loc[35:, "volume"] = 0
    a, b = environment(features, prices), environment(features, altered)
    oa, _ = a.reset(seed=4)
    ob, _ = b.reset(seed=4)
    np.testing.assert_array_equal(oa, ob)
    for action in [2, 1, 0, 2]:
        oa, *_ = a.step(action)
        ob, *_ = b.step(action)
        np.testing.assert_array_equal(oa, ob)
    a.step(1)
    b.step(1)
    assert a.fills[-1]["requested_qty"] == b.fills[-1]["requested_qty"]


def test_no_short_positions_or_negative_cash_with_opening_gaps():
    features, prices = market(180)
    prices.loc[::2, "open"] *= 4
    prices.loc[1::2, "open"] *= 0.1
    env = environment(features, prices, max_weight=1.0, rebalance_band=0)
    env.reset(seed=8)
    rng = np.random.default_rng(8)
    while not env.done:
        _, _, _, _, info = env.step(int(rng.integers(3)))
        assert env.cash >= 0
        assert env.qty >= 0
        assert 0 <= info["weight"] <= 1


def test_execution_and_equity_match_project_backtester():
    features, prices = market()
    prices.loc[31:35, "volume"] = 0.1  # Liquidez previa, fills parciales y mínimo.
    prices.loc[40:42, "open"] *= 1.15
    costs, rules = CostModel(), Rules()
    env = environment(features, prices, costs=costs, rules=rules)
    _, initial = env.reset(seed=0)
    actions = np.resize([2, 2, 1, 0, 0, 2, 1], len(env.index) - 1)
    equity = [initial["equity"]]
    for action in actions:
        _, _, _, _, info = env.step(int(action))
        equity.append(info["equity"])
    weights = pd.Series(np.r_[actions * 0.25, 0], index=env.index)
    subset = prices[prices.open_time.isin(env.index)]
    reference = run_backtest({"BTCUSDT": subset}, {"BTCUSDT": weights}, costs,
                             {"BTCUSDT": rules}, rebalance_band=env.rebalance_band)
    np.testing.assert_allclose(equity, reference.equity.equity, rtol=1e-12)
    for key in ["qty", "fill_price", "fee", "impact_cost", "requested_qty"]:
        np.testing.assert_allclose([fill[key] for fill in env.fills], reference.fills[key])


def test_training_normalization_is_frozen_and_finite():
    f, p = market()
    f.loc[f.index[:2], "constant"] = np.nan
    stats = TradingEnv.fit_normalization(f, f.index[0], f.index[30])
    f.loc[f.index[30]:, "trend"] = 1e12
    f.loc[f.index[30], "constant"] = np.nan
    env = TradingEnv(f, p, f.index[30], f.index[-1], normalization=stats)
    obs, _ = env.reset()
    assert obs[0] == 10
    assert obs[1] == 0
    assert stats["mean"][0] == pytest.approx(14.5)
    assert obs.dtype == np.float32
    with pytest.raises(ValueError, match="normaliz"):
        TradingEnv(f, p, f.index[20], f.index[-1], normalization=stats)


def test_random_episodes_are_seeded_fixed_length_and_stay_in_range():
    env = environment(training=True, episode_length=7)
    starts = []
    for seed in range(10):
        _, first = env.reset(seed=seed)
        starts.append(first["time"])
        for step in range(7):
            _, _, terminated, truncated, last = env.step(0)
            assert not terminated
            assert truncated == (step == 6)
        assert last["time"] < env.end
        assert (last["time"] - first["time"]) == pd.Timedelta(hours=7)
        with pytest.raises(RuntimeError):
            env.step(0)
    assert len(set(starts)) > 1
    _, first = env.reset(seed=0)
    assert first["time"] == starts[0]


def test_reward_penalizes_only_drawdown_increases_and_executed_turnover():
    f, p = market()
    p.loc[31, "close"] = 70
    p.loc[32, "close"] = 90
    env = environment(f, p, rebalance_band=1, dd_penalty=0.2, turnover_penalty=0.03)
    _, before = env.reset()
    _, reward, _, _, first = env.step(2)
    expected = (np.log(first["equity"] / before["equity"])
                - 0.2 * first["drawdown"] - 0.03 * first["turnover"])
    assert reward == pytest.approx(expected)
    _, reward, _, _, recovered = env.step(2)
    assert reward == pytest.approx(np.log(recovered["equity"] / first["equity"]))
    assert recovered["turnover"] == 0


def test_missing_hour_is_rejected_instead_of_trading_across_gap():
    f, p = market()
    with pytest.raises(ValueError, match="1h"):
        environment(f.drop(f.index[35]), p)


def train_cli():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "train_rl.py"
    spec = importlib.util.spec_from_file_location("train_rl", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("arguments", [
    ["--eval-end", "2025-11-01T01:00:00Z"],
    ["--train-end", "2024-02-01"],
    ["--eval-start", "2025-11-01", "--eval-end", "2025-12-01"],
])
def test_cli_rejects_holdout_and_overlapping_ranges(arguments):
    from quant_ai.config import load_config

    cli = train_cli()
    with pytest.raises(ValueError):
        cli.validate_ranges(cli.parser().parse_args(arguments), load_config())


def test_cli_allows_exclusive_holdout_boundary_and_numeric_checkpoint_order(tmp_path):
    from quant_ai.config import load_config

    cli = train_cli()
    _, train_end, eval_start, eval_end = cli.validate_ranges(cli.parser().parse_args([]), load_config())
    assert train_end == eval_start
    assert eval_end == load_config().final_holdout_start
    for name in ["rl_9_steps.zip", "rl_100_steps.zip", "rl_invalid_steps.zip"]:
        (tmp_path / name).touch()
    steps, checkpoint = cli.latest_checkpoint(tmp_path)
    assert steps == 100
    assert checkpoint.name == "rl_100_steps.zip"

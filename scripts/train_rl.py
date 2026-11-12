"""Entrena PPO/DQN spot de un activo por semilla, exclusivamente en desarrollo."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from quant_ai.backtesting.engine import CostModel, Rules, run_backtest
from quant_ai.backtesting.metrics import performance_metrics
from quant_ai.config import load_config
from quant_ai.training.dataset import build_datasets

ROOT = Path(__file__).resolve().parents[1]


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/base.yaml")
    ap.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
    ap.add_argument("--symbol", choices=["BTCUSDT", "ETHUSDT"], default="BTCUSDT")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--train-start", help="Por defecto market.history_start; inclusivo UTC.")
    ap.add_argument("--train-end", default="2024-01-01", help="Exclusivo UTC.")
    ap.add_argument("--eval-start", default="2024-01-01", help="Inclusivo UTC.")
    ap.add_argument("--eval-end", default="2025-11-01", help="Exclusivo UTC; incluye octubre 31.")
    ap.add_argument("--run", help="Nombre de carpeta dentro de <models_dir>/rl/.")
    ap.add_argument("--models-dir", type=Path, help="Sobrescribe models_dir, por ejemplo en Drive.")
    ap.add_argument("--timesteps", type=int, default=200_000, help="Objetivo TOTAL, también al reanudar.")
    ap.add_argument("--episode-length", type=int, default=720, help="Pasos horarios por episodio train.")
    ap.add_argument("--eval-freq", type=int, default=25_000)
    ap.add_argument("--checkpoint-freq", type=int, default=25_000)
    ap.add_argument("--max-weight", type=float, help="Por defecto strategy.target_weight.")
    ap.add_argument("--rebalance-band", type=float, default=0.05)
    ap.add_argument("--dd-penalty", type=float, default=0.1)
    ap.add_argument("--turnover-penalty", type=float, default=0.001)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--resume", action="store_true", help="Retoma el último checkpoint numérico de cada semilla.")
    ap.add_argument("--smoke", action="store_true", help="256 pasos; evaluación completa, sin tocar holdout.")
    return ap


def validate_ranges(args, cfg):
    """Se llama ANTES de construir datasets o importar SB3; no permite --final."""
    start = pd.to_datetime(args.train_start or cfg["market"]["history_start"], utc=True)
    train_end, eval_start, eval_end = [pd.to_datetime(value, utc=True) for value in
                                      (args.train_end, args.eval_start, args.eval_end)]
    if not start < train_end <= eval_start < eval_end:
        raise ValueError("Se requiere train_start < train_end <= eval_start < eval_end.")
    if eval_end > cfg.final_holdout_start:
        raise ValueError("El rango solicitado invade el holdout final: eval_end es exclusivo.")
    if cfg.interval != "1h" or args.symbol not in cfg.symbols:
        raise ValueError("RL requiere BTCUSDT/ETHUSDT configurado con velas spot de 1h.")
    return start, train_end, eval_start, eval_end


def write_json(path, value):
    # JSON estricto: métricas indefinidas (p. ej. Calmar sin drawdown) se guardan como null.
    def clean(item):
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (tuple, list)):
            return [clean(v) for v in item]
        if isinstance(item, (float, np.floating)):
            return float(item) if np.isfinite(item) else None
        if isinstance(item, np.integer):
            return int(item)
        return item

    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def latest_checkpoint(folder):
    checkpoints = []
    for path in folder.glob("rl_*_steps.zip"):
        match = re.fullmatch(r"rl_(\d+)_steps.zip", path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    return max(checkpoints, default=(0, None), key=lambda pair: pair[0])


def evaluate_policy(model, env):
    """Un episodio determinista completo, incluida la fila inicial en efectivo."""
    observation, info = env.reset(seed=0)
    rows, reward_sum = [info], 0.0
    while not env.done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, _, _, info = env.step(int(np.asarray(action).item()))
        rows.append(info)
        reward_sum += reward
    return pd.DataFrame(rows).set_index("time"), float(reward_sum)


def make_evaluation_callback(env, output, frequency, resume_steps):
    from stable_baselines3.common.callbacks import BaseCallback

    class EvaluationCSV(BaseCallback):
        def __init__(self):
            super().__init__()
            self.rows = []
            if output.exists():
                previous = pd.read_csv(output)
                self.rows = previous.loc[previous.timesteps < resume_steps].to_dict("records")
            self.last_step = None

        def record(self):
            if self.last_step == self.num_timesteps:
                return
            _, reward = evaluate_policy(self.model, env)
            # Evaluación determinista de un único episodio completo: media = suma del episodio.
            self.rows.append({"timesteps": self.num_timesteps, "ep_reward_mean": reward})
            temporary = output.with_suffix(".tmp.csv")
            pd.DataFrame(self.rows).to_csv(temporary, index=False)
            temporary.replace(output)
            self.last_step = self.num_timesteps
            print(f"Evaluación: timesteps={self.num_timesteps}, ep_reward_mean={reward:.6f}", flush=True)

        def _on_training_start(self):
            self.record()

        def _on_step(self):
            if self.num_timesteps % frequency == 0:
                self.record()
            return True

        def _on_training_end(self):
            # PPO puede actualizar pesos tras la última llamada a _on_step.
            self.rows = [row for row in self.rows if row["timesteps"] != self.num_timesteps]
            self.last_step = None
            self.record()

    return EvaluationCSV()


def main():
    ap = parser()
    args = ap.parse_args()
    os.environ.setdefault("QUANT_AI_HOME", str(ROOT))
    cfg = load_config(args.config)
    try:
        train_start, train_end, eval_start, eval_end = validate_ranges(args, cfg)
    except ValueError as exc:
        ap.error(str(exc))
    if args.resume and not args.run:
        ap.error("--resume requiere --run para identificar el experimento.")
    if args.run and (args.run in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run)):
        ap.error("--run debe ser un nombre de carpeta, sin rutas.")
    if (len(set(args.seeds)) != len(args.seeds) or any(seed < 0 or seed >= 2**32 for seed in args.seeds)
            or min(args.timesteps, args.episode_length, args.eval_freq, args.checkpoint_freq) <= 0):
        ap.error("Semillas únicas en [0, 2**32) y longitudes/frecuencias positivas requeridas.")
    if args.smoke:
        args.timesteps = min(args.timesteps, 256)
        args.episode_length = min(args.episode_length, 64)
        args.eval_freq, args.checkpoint_freq = min(args.eval_freq, 128), min(args.checkpoint_freq, 128)
    # Dependencias opcionales: jamás se instalan desde este script.
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from quant_ai.reinforcement_learning.env import TradingEnv

    datasets = build_datasets(cfg, final=False)
    ds = datasets[args.symbol]
    features = ds.features[ds.feature_names]
    normalization = TradingEnv.fit_normalization(features, train_start, train_end)
    costs = CostModel.from_config(cfg["costs"])
    instrument = cfg["instrument_rules"][args.symbol]
    rules = Rules(lot_step=instrument["lot_step"], min_notional=instrument["min_notional"])
    max_weight = args.max_weight if args.max_weight is not None else cfg["strategy"]["target_weight"]
    env_kwargs = dict(normalization=normalization, costs=costs, rules=rules,
                      max_weight=max_weight, initial_cash=cfg["backtest"]["initial_cash"],
                      rebalance_band=args.rebalance_band, dd_penalty=args.dd_penalty,
                      turnover_penalty=args.turnover_penalty)
    train_env = Monitor(TradingEnv(features, ds.prices, train_start, train_end, training=True,
                                  episode_length=args.episode_length, **env_kwargs))
    eval_env = TradingEnv(features, ds.prices, eval_start, eval_end, **env_kwargs)
    # No informar un periodo completo si faltan sus extremos o sus horas internas.
    for env, start, end in [(train_env.unwrapped, train_start, train_end), (eval_env, eval_start, eval_end)]:
        if env.index[0] != start or env.index[-1] + pd.Timedelta(hours=1) != end:
            raise ValueError(f"Faltan datos para cubrir exactamente [{start}, {end}).")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = (args.models_dir or cfg.models_dir) / "rl" / (args.run or f"{args.algo}_{args.symbol}_{stamp}")
    # Huella del contenido realmente usado: rechaza reanudación con otros datos.
    fingerprint = hashlib.sha256()
    for frame in [features, ds.prices.set_index("open_time")[["open", "high", "low", "close", "volume"]]]:
        used = frame.loc[(frame.index >= train_start) & (frame.index < eval_end)]
        fingerprint.update(pd.util.hash_pandas_object(used, index=True).values.tobytes())
    manifest = {"algo": args.algo, "symbol": args.symbol, "seeds": args.seeds,
                "train_start": train_start.isoformat(), "train_end": train_end.isoformat(),
                "eval_start": eval_start.isoformat(), "eval_end": eval_end.isoformat(),
                "episode_length": args.episode_length, "max_weight": max_weight,
                "rebalance_band": args.rebalance_band, "dd_penalty": args.dd_penalty,
                "turnover_penalty": args.turnover_penalty, "smoke": args.smoke,
                "config": cfg.raw, "normalization": normalization, "data_sha256": fingerprint.hexdigest()}
    manifest_path = out / "experiment.json"
    if out.exists() and any(out.iterdir()):
        if not args.resume or not manifest_path.exists():
            ap.error("La carpeta contiene artefactos; use otra --run o --resume con su manifiesto.")
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            ap.error("Datos, normalización o configuración distintos; inicie otro --run.")
    out.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    write_json(out / "normalization.json", normalization)

    # Una entrada con el mismo peso, precio, liquidez y costes; luego conserva unidades.
    benchmark_weights = pd.Series(np.nan, index=eval_env.index)
    benchmark_weights.iloc[0] = max_weight
    benchmark_prices = ds.prices[ds.prices.open_time.isin(eval_env.index)]
    benchmark = run_backtest({args.symbol: benchmark_prices}, {args.symbol: benchmark_weights}, costs,
                             {args.symbol: rules}, initial_cash=env_kwargs["initial_cash"],
                             rebalance_band=args.rebalance_band)
    ann = cfg["backtest"]["annualization_bars"]
    baseline = performance_metrics(benchmark.equity.equity, benchmark.trades, benchmark.fills,
                                   benchmark.equity.gross_exposure, ann=ann)
    benchmark.equity.to_csv(out / "buy_hold_equity.csv")
    summary_path = out / "eval_summary.json"
    summary = {"period": {"start": eval_start.isoformat(), "end_exclusive": eval_end.isoformat()},
               "symbol": args.symbol, "algo": args.algo, "max_weight": max_weight,
               "buy_and_hold": baseline, "seeds": {}, "stability": {}}
    if args.resume and summary_path.exists():
        summary["seeds"] = json.loads(summary_path.read_text(encoding="utf-8"))["seeds"]
    algorithm = PPO if args.algo == "ppo" else DQN
    try:
        for seed in args.seeds:
            seed_dir = out / f"seed_{seed}"
            checkpoint_dir = seed_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            steps, checkpoint = latest_checkpoint(checkpoint_dir) if args.resume else (0, None)
            if checkpoint:
                model = algorithm.load(checkpoint, env=train_env, device=args.device, seed=seed)
                if args.algo == "dqn":
                    replay = checkpoint_dir / f"rl_replay_buffer_{steps}_steps.pkl"
                    if not replay.exists():
                        raise FileNotFoundError(f"Falta replay buffer del checkpoint DQN: {replay}")
                    model.load_replay_buffer(replay)
            else:
                if args.algo == "ppo":
                    hyperparams = {"n_steps": 64 if args.smoke else 2048,
                                   "batch_size": 32 if args.smoke else 64,
                                   "n_epochs": 2 if args.smoke else 10}
                else:
                    hyperparams = {"buffer_size": 100_000, "learning_starts": 32 if args.smoke else 1000,
                                   "batch_size": 32 if args.smoke else 64}
                model = algorithm("MlpPolicy", train_env, seed=seed, device=args.device, verbose=1, **hyperparams)
            evaluation = make_evaluation_callback(eval_env, seed_dir / "metrics.csv", args.eval_freq, steps)
            checkpoints = CheckpointCallback(save_freq=args.checkpoint_freq, save_path=str(checkpoint_dir),
                                             name_prefix="rl", save_replay_buffer=args.algo == "dqn")
            remaining = args.timesteps - model.num_timesteps
            if remaining > 0:
                model.learn(total_timesteps=remaining, callback=[checkpoints, evaluation], reset_num_timesteps=False)
            model.save(seed_dir / "model.zip")
            # Checkpoint final tras actualizar pesos; PPO periódico puede caer dentro de un rollout.
            model.save(checkpoint_dir / f"rl_{model.num_timesteps}_steps.zip")
            if args.algo == "dqn":
                model.save_replay_buffer(checkpoint_dir / f"rl_replay_buffer_{model.num_timesteps}_steps.pkl")
            curve, episode_reward = evaluate_policy(model, eval_env)
            curve.to_csv(seed_dir / "eval_equity.csv")
            # También deja CSV correcto si la semilla ya alcanzó el objetivo antes de --resume.
            metric_path = seed_dir / "metrics.csv"
            rows = pd.read_csv(metric_path) if metric_path.exists() else pd.DataFrame(columns=["timesteps", "ep_reward_mean"])
            rows = rows.loc[rows.timesteps < model.num_timesteps].reset_index(drop=True)
            rows.loc[len(rows)] = [model.num_timesteps, episode_reward]
            rows["timesteps"] = rows["timesteps"].astype(int)
            rows.to_csv(metric_path, index=False)
            metrics = performance_metrics(curve.equity, pd.DataFrame(eval_env.trades), pd.DataFrame(eval_env.fills),
                                          curve.weight, ann=ann)
            summary["seeds"][str(seed)] = {"timesteps": model.num_timesteps, "ep_reward_mean": episode_reward,
                                           "metrics": metrics,
                                           "return_minus_buy_hold": metrics["total_return"] - baseline["total_return"]}
            summary["stability"] = {}
            for name in ["total_return", "sharpe", "max_drawdown"]:
                values = [entry["metrics"][name] for entry in summary["seeds"].values()]
                summary["stability"][name] = {"mean": float(np.mean(values)), "std": float(np.std(values)),
                                              "n_seeds": len(values)}
            write_json(summary_path, summary)
            del model
    finally:
        train_env.close()
        eval_env.close()
    print(f"Artefactos RL guardados en {out}")


if __name__ == "__main__":
    main()

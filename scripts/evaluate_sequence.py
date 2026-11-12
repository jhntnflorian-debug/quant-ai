"""Evalúa secuencias y baselines sobre el mismo intervalo de precios."""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from quant_ai.config import load_config
from quant_ai.decision_engine.policy import PolicyParams, weights_from_probs
from quant_ai.models.baselines import RULE_STRATEGIES
from quant_ai.training.dataset import build_datasets
from quant_ai.training.experiment import run_strategy_backtests

ROOT = Path(__file__).resolve().parents[1]


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def evaluate(pred_path, name="dl_gru", config=None, cfg=None, datasets=None):
    pred_path = Path(pred_path)
    predictions = pd.read_parquet(pred_path).sort_index()
    required = {"symbol", "fold", "p_seq", "p_seq_raw", "y", "fwd_ret"}
    if predictions.empty or not required.issubset(predictions.columns):
        raise ValueError("Las predicciones están vacías o tienen columnas incompletas.")
    if not isinstance(predictions.index, pd.DatetimeIndex) or predictions.index.tz is None:
        raise ValueError("Las predicciones necesitan un índice open_time con zona horaria.")
    if name in RULE_STRATEGIES:
        raise ValueError("El nombre secuencial no puede sobrescribir un baseline.")
    os.environ.setdefault("QUANT_AI_HOME", str(ROOT))
    cfg = cfg or load_config(config or ROOT / "configs/base.yaml")
    symbols = predictions.symbol.unique().tolist()
    if not set(symbols).issubset(cfg.symbols):
        raise ValueError("Hay símbolos no definidos en la configuración.")
    if predictions.index.max() >= cfg.final_holdout_start:
        raise ValueError("Las predicciones invaden el holdout final.")
    if not np.isfinite(predictions.p_seq).all() or not predictions.p_seq.between(0, 1).all():
        raise ValueError("p_seq debe contener probabilidades finitas entre cero y uno.")
    if predictions.reset_index().duplicated([predictions.index.name or "index", "symbol"]).any():
        raise ValueError("Hay predicciones duplicadas por símbolo y open_time.")
    datasets = datasets or build_datasets(cfg, final=False)
    start, end = predictions.index.min(), predictions.index.max()
    # La API existente solo recibe start. Recortar precios limita también el final
    # y evita comparar el modelo con un buy_and_hold de un periodo más largo.
    selected = {}
    weights = {name: {}}
    policy = PolicyParams.from_config(cfg["strategy"])
    for symbol in symbols:
        ds = datasets[symbol]
        prices = ds.prices.loc[ds.prices.open_time.between(start, end)].copy()
        if prices.empty:
            raise ValueError(f"{symbol}: no hay precios en el periodo de predicción.")
        selected[symbol] = replace(ds, prices=prices)
        index = ds.features.index[(ds.features.index >= start) & (ds.features.index <= end)]
        probs = predictions.loc[predictions.symbol == symbol, "p_seq"].reindex(index)
        # Una ventana inválida da efectivo explícito; nunca se propaga su señal anterior.
        weights[name][symbol] = weights_from_probs(probs, None, policy)
    for baseline, function in RULE_STRATEGIES.items():
        weights[baseline] = {symbol: function(ds.features, cfg["strategy"]["target_weight"])
                             for symbol, ds in selected.items()}
    # Comparación directa DL vs ML (Experimento 1 vs 2): ensemble LR+LightGBM de la última corrida
    # walk-forward completa, recortado al MISMO tramo. Si no existe, se omite y se informa.
    ml_source = None
    runs_dir = cfg.reports_dir / "runs"
    for run in sorted(runs_dir.glob("*"), reverse=True) if runs_dir.exists() else []:
        summary_path, pred_ml = run / "summary.json", run / "predictions.parquet"
        if summary_path.exists() and pred_ml.exists():
            if json.loads(summary_path.read_text(encoding="utf-8")).get("kind") == "full":
                ml_source = run.name
                ml_pred = pd.read_parquet(pred_ml)
                weights["ml_ensemble"] = {}
                for symbol, ds in selected.items():
                    index = ds.features.index[(ds.features.index >= start) & (ds.features.index <= end)]
                    ps = ml_pred[ml_pred.symbol == symbol]
                    weights["ml_ensemble"][symbol] = weights_from_probs(
                        ps["p_ensemble"].reindex(index), ps["uncertainty"].reindex(index), policy)
                break
    # mismo tratamiento de stops para DL y reglas de tendencia (no para comprar y mantener)
    timing = {k for k in weights if k != "buy_and_hold"}
    results, equity, _, _ = run_strategy_backtests(cfg, selected, weights, start, stop_names=timing)
    summary = {"name": name, "period": {"start": start.isoformat(), "end": end.isoformat()},
               "symbols": symbols, "strategies": results, "final_holdout_used": False,
               "missing_prediction_policy": "flat", "prediction_rows": len(predictions),
               "ml_ensemble_source_run": ml_source}
    (pred_path.parent / "backtest_summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    equity.to_parquet(pred_path.parent / "equity.parquet")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--name", default="dl_gru")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/base.yaml")
    args = parser.parse_args()
    evaluate(args.pred, args.name, args.config)


if __name__ == "__main__":
    main()

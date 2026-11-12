"""Experimento v2 (ablación + meta-etiquetado + volatility targeting). Pensado para Colab (CPU, n_jobs=-1).

Ejemplos:
  python scripts/run_experiments_v2.py --n-jobs -1                     # todas las variantes (Colab)
  python scripts/run_experiments_v2.py --only v0_base_h12 v1_base_h24 --max-folds 1 --symbols BTCUSDT --tag v2smoke
"""
import argparse

from quant_ai.config import load_config
from quant_ai.training.dataset import build_datasets
from quant_ai.training.experiment_v2 import load_v2_config, run_experiment_v2

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--v2-config", default=None)
    ap.add_argument("--only", nargs="*", default=None, help="subconjunto de variantes")
    ap.add_argument("--max-folds", type=int, default=None, help="solo para pruebas mínimas")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--tag", default="v2")
    a = ap.parse_args()
    ov = {}
    if a.n_jobs is not None:
        ov["compute"] = {"n_jobs": a.n_jobs}
    if a.symbols:
        ov["market"] = {"symbols": a.symbols}
    cfg = load_config(a.config, ov or None)
    datasets = build_datasets(cfg, final=False)
    run_experiment_v2(cfg, datasets, load_v2_config(a.v2_config), only=a.only, max_folds=a.max_folds, tag=a.tag)

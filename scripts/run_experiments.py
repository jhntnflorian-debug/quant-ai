"""Ejecuta el experimento walk-forward (baselines + ML) y opcionalmente registra el modelo de producción.

Ejemplos:
  python scripts/run_experiments.py --skip-ml                 # solo baselines (ligero, apto para PC local)
  python scripts/run_experiments.py --max-folds 2 --symbols BTCUSDT --tag smoke   # prueba mínima
  python scripts/run_experiments.py --n-jobs -1 --register    # corrida completa (Colab, CPU alta RAM)
  python scripts/run_experiments.py --final                   # SOLO evaluación final pre-registrada
"""
import argparse
import json

from quant_ai.config import load_config
from quant_ai.training.dataset import build_datasets
from quant_ai.training.experiment import run_experiment, train_production_model

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-ml", action="store_true")
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--n-jobs", type=int, default=None, help="hilos para LightGBM (-1 = todos)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--register", action="store_true", help="entrenar y registrar modelo de producción")
    ap.add_argument("--final", action="store_true",
                    help="evalúa en el holdout final pre-registrado (usar UNA sola vez con el diseño congelado)")
    a = ap.parse_args()

    overrides = {}
    if a.n_jobs is not None:
        overrides["compute"] = {"n_jobs": a.n_jobs}
    if a.symbols:
        overrides["market"] = {"symbols": a.symbols}
    cfg = load_config(a.config, overrides or None)
    if a.final:
        ans = input("Vas a usar el HOLDOUT FINAL. ¿El diseño está congelado? Escribe 'si, congelado': ")
        if ans.strip().lower() != "si, congelado":
            raise SystemExit("Cancelado: el holdout sigue sin tocar.")
    print(f"Símbolos: {cfg.symbols} | holdout final desde {cfg.final_holdout_start} "
          f"({'EN USO' if a.final else 'excluido'})")
    datasets = build_datasets(cfg, final=a.final)
    out = run_experiment(cfg, datasets, skip_ml=a.skip_ml, max_folds=a.max_folds, tag=a.tag, final=a.final)
    if a.register and not a.final:
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        train_production_model(cfg, datasets, summary)

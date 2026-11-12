"""Descarga/actualiza incrementalmente las velas de Binance definidas en configs/base.yaml.

Uso:  python scripts/download_data.py [--no-rest] [--strict-rest]

Si la API REST falla (p. ej. HTTP 451 desde una VM de Colab en una región restringida), se avisa y se
continúa con los archivos mensuales y los datos ya guardados: el periodo de desarrollo no depende de las
últimas horas. Con --strict-rest el fallo detiene el script. Al final se comprueba que los datos cubren
el periodo de desarrollo completo.
"""
import argparse

import pandas as pd

from quant_ai.config import load_config
from quant_ai.data.store import update_all

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-rest", action="store_true", help="solo archivos mensuales, sin API REST")
    ap.add_argument("--strict-rest", action="store_true", help="detenerse si la API REST falla")
    a = ap.parse_args()
    cfg = load_config(a.config)
    print(f"Actualizando {cfg.symbols} {cfg.interval} en {cfg.data_dir}")
    results = update_all(cfg, use_rest=not a.no_rest, rest_best_effort=not a.strict_rest)
    # El entrenamiento necesita el periodo de desarrollo completo (hasta el inicio del holdout final).
    last_needed = cfg.final_holdout_start - pd.Timedelta(cfg.interval)
    short = {s: str(r["quality"]["end"]) for s, r in results.items()
             if pd.Timestamp(r["quality"]["end"]) < last_needed}
    if short:
        raise SystemExit(f"Datos incompletos para entrenar: {short} no llega a {last_needed}.")
    print("Datos listos: cubren todo el periodo de desarrollo.")

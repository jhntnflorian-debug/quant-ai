"""Evaluación final de la familia v2 con trato idéntico (ejecutar al terminar TODAS las etapas v2).

Re-evalúa juntos: variantes ML de la última corrida v2 completa, GRU/TCN v2 y la CNN de gráficos, sobre la
ventana común, con los mismos costes, stops y referencias. Calcula el DSR con todos los intentos acumulados,
el bootstrap pareado frente a las referencias y la puerta v2. No registra ni activa modelos.

Uso: python scripts/consolidate_v2.py [--config configs/base.yaml]
"""
import argparse

from quant_ai.config import load_config
from quant_ai.training.dataset import build_datasets
from quant_ai.training.experiment_v2 import consolidate_v2, load_v2_config

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--v2-config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = consolidate_v2(cfg, build_datasets(cfg, final=False), load_v2_config(a.v2_config))
    print((out / "report.md").read_text(encoding="utf-8"))

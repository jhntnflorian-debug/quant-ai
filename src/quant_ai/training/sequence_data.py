"""Ventanas causales perezosas y particiones temporales purgadas."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from quant_ai.validation.splits import Fold, purged_walk_forward, split_fit_validation


def fit_standardizer(values: np.ndarray, positions: np.ndarray) -> dict:
    """Ajusta únicamente sobre filas de ajuste; columnas ausentes quedan en cero."""
    if not len(positions):
        raise ValueError("No hay filas de ajuste para normalizar.")
    fit = np.asarray(values[positions], dtype=np.float64)
    valid = np.isfinite(fit)
    count = valid.sum(axis=0)
    mean = np.where(valid, fit, 0).sum(axis=0) / np.maximum(count, 1)
    variance = np.where(valid, (fit - mean) ** 2, 0).sum(axis=0) / np.maximum(count, 1)
    std = np.sqrt(variance)
    std[(std < 1e-8) | ~np.isfinite(std)] = 1
    return {"mean": mean.tolist(), "std": std.tolist()}


def standardize(values: np.ndarray, stats: dict) -> np.ndarray:
    x = (values.astype(np.float64) - np.asarray(stats["mean"])) / np.asarray(stats["std"])
    return np.nan_to_num(x, nan=0, posinf=0, neginf=0).astype(np.float32)


class WindowDataset(Dataset):
    """Comparte un array 2D; cada __getitem__ obtiene una vista hasta t incluida.

    Las posiciones representan finales de ventana en el índice completo, nunca
    posiciones del DataFrame filtrado por etiquetas. No se materializa (N,L,F).
    """
    def __init__(self, values, labels, index, positions, window=64):
        self.values = np.asarray(values, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.index = pd.DatetimeIndex(index)
        self.window = int(window)
        if self.window < 1 or self.values.ndim != 2:
            raise ValueError("Se requiere window positivo y valores de dimensión (N,F).")
        if len(self.index) != len(self.values) or len(self.labels) != len(self.index):
            raise ValueError("Valores, etiquetas e índice deben estar alineados.")
        if not self.index.is_monotonic_increasing or not self.index.is_unique or self.index.hasnans:
            raise ValueError("open_time debe ser creciente, único y sin NaT.")
        positions = np.asarray(list(positions), dtype=np.int64)
        if np.any((positions < 0) | (positions >= len(self.index))):
            raise ValueError("Posiciones fuera del índice.")
        positions = positions[positions >= self.window - 1]
        # También se comprueba cada salto: evita que intervalos irregulares se compensen.
        gaps = np.asarray((self.index[1:] - self.index[:-1]) != pd.Timedelta(hours=1), dtype=int)
        cumulative = np.r_[0, np.cumsum(gaps)]
        starts = positions - self.window + 1
        continuous = cumulative[positions] == cumulative[starts]
        spans = self.index[positions] - self.index[starts]
        continuous &= spans == pd.Timedelta(hours=self.window - 1)
        self.positions = positions[continuous]

    @property
    def times(self):
        return self.index[self.positions]

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, item):
        end = self.positions[item]
        return (torch.from_numpy(self.values[end - self.window + 1:end + 1]),
                torch.tensor(self.labels[end], dtype=torch.float32))


@dataclass
class SequenceFold:
    symbol: str
    fold: Fold
    fit: WindowDataset
    validation: WindowDataset   # early stopping; es el mismo objeto que calibration salvo separate_validation
    calibration: WindowDataset
    test: WindowDataset
    normalization: dict
    feature_names: list[str]


def build_sequence_folds(cfg, ds, window=64, last_folds=None, smoke=False,
                         separate_validation=False) -> Iterator[SequenceFold]:
    """Produce un fold a la vez y comparte su matriz normalizada entre sus datasets.

    La prueba conserva todas las velas con ventana válida, incluso sin y/t1.
    Smoke submuestrea los finales de ajuste/validación/calibración, nunca las filas de
    contexto ni el tramo de prueba. El holdout final permanece excluido.

    ``separate_validation=False`` reproduce v1: la calibración sirve también para el early stopping.
    Con ``True`` se aparta el último 20 % del ajuste como validación (``split_fit_validation``, con
    purga) y el estandarizador se ajusta solo con el ajuste resultante.
    """
    if last_folds is not None and last_folds < 1:
        raise ValueError("last_folds debe ser positivo o None.")
    features = ds.features.loc[ds.features.index < cfg.final_holdout_start, ds.feature_names]
    frame = ds.frame()
    frame = frame.loc[frame.y.notna() & (frame.index < cfg.final_holdout_start)
                      & (frame.t1 < cfg.final_holdout_start)]
    if frame.empty or features.empty:
        raise ValueError(f"{ds.symbol}: no hay datos de desarrollo etiquetados.")
    wf = cfg["splits"]["walk_forward"]
    folds = purged_walk_forward(
        frame.index, frame.t1, wf["first_test_start"], wf["test_months"], wf["min_train_months"],
        pd.Timedelta(cfg.interval) * cfg["splits"]["embargo_bars"], wf["calibration_fraction"],
        end=features.index.max(), expanding=wf.get("expanding", True))
    if last_folds:
        folds = folds[-last_folds:]
    if not folds:
        raise ValueError(f"{ds.symbol}: no se generaron folds con la configuración actual.")
    raw = features.to_numpy(dtype=np.float32)
    labels = ds.labels.y.reindex(features.index).to_numpy(dtype=np.float32)
    embargo = pd.Timedelta(cfg.interval) * cfg["splits"]["embargo_bars"]

    def subsample(pos, n):
        return pos[np.linspace(0, len(pos) - 1, min(n, len(pos)), dtype=int)] if smoke else pos

    for fold in folds:
        if separate_validation:
            fit_idx, val_idx = split_fit_validation(fold.train_idx, frame.index, frame.t1, embargo)
            val_pos = subsample(features.index.get_indexer(frame.index[val_idx]), 512)
        else:
            fit_idx = fold.train_idx
        fit_pos = subsample(features.index.get_indexer(frame.index[fit_idx]), 2048)
        cal_pos = subsample(features.index.get_indexer(frame.index[fold.calib_idx]), 512)
        test_pos = np.flatnonzero((features.index >= fold.test_start) & (features.index < fold.test_end))
        stats = fit_standardizer(raw, fit_pos)
        values = standardize(raw, stats)
        fit, cal, test = [WindowDataset(values, labels, features.index, p, window) for p in (fit_pos, cal_pos, test_pos)]
        val = WindowDataset(values, labels, features.index, val_pos, window) if separate_validation else cal
        if any(len(part) == 0 for part in (fit, val, cal, test)):
            raise ValueError(f"{ds.symbol} fold {fold.k}: tramo vacío tras excluir ventanas con huecos.")
        yield SequenceFold(ds.symbol, fold, fit, val, cal, test, stats, list(ds.feature_names))

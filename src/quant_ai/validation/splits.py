"""Validación temporal: walk-forward con purga y embargo, y guardia del holdout final."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class HoldoutViolation(RuntimeError):
    """Se intentó usar datos del holdout final sin la bandera explícita --final."""


@dataclass
class Fold:
    k: int
    train_idx: np.ndarray      # posiciones enteras del conjunto de ajuste
    calib_idx: np.ndarray      # posiciones para calibrar probabilidades (cronológicamente después del ajuste)
    test_idx: np.ndarray
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def guard_holdout(index: pd.DatetimeIndex, holdout_start: pd.Timestamp, final: bool = False) -> None:
    if not final and len(index) and index.max() >= holdout_start:
        raise HoldoutViolation(
            f"Los datos llegan a {index.max()} y el holdout final empieza en {holdout_start}. "
            "Recorta los datos o usa --final (solo para la evaluación final pre-registrada).")


def restrict_to_development(df: pd.DataFrame, holdout_start: pd.Timestamp, t1_col: str | None = "t1") -> pd.DataFrame:
    """Recorta a periodo de desarrollo: filas antes del holdout cuya etiqueta TAMBIÉN se resuelve antes."""
    out = df[df.index < holdout_start]
    if t1_col and t1_col in out:
        out = out[out[t1_col].isna() | (out[t1_col] < holdout_start)]
    return out


def purged_walk_forward(index: pd.DatetimeIndex, t1: pd.Series, first_test_start: str | pd.Timestamp,
                        test_months: int, min_train_months: int, embargo: pd.Timedelta,
                        calibration_fraction: float = 0.2, end: pd.Timestamp | None = None,
                        expanding: bool = True) -> list[Fold]:
    """Genera folds cronológicos.

    - test: [ts, ts + test_months)
    - train: muestras con t < ts - embargo y t1 < ts - embargo (purga de etiquetas solapadas)
    - calib: último ``calibration_fraction`` del train; el tramo de ajuste se purga respecto a calib.
    """
    index = pd.DatetimeIndex(index)
    t1 = pd.Series(t1, index=index)
    ts = pd.Timestamp(first_test_start)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    end = end or index.max()
    folds = []
    k = 0
    t1_vals = t1.to_numpy()
    times = index.to_numpy()
    while ts < end:
        te = min(ts + pd.DateOffset(months=test_months), end + pd.Timedelta(1, "ns"))
        cutoff = ts - embargo
        valid_t1 = ~pd.isna(t1_vals)
        train_mask = (index < cutoff) & valid_t1 & (t1 < cutoff).to_numpy()
        if not expanding:
            train_mask &= index >= ts - pd.DateOffset(months=min_train_months)
        test_mask = (index >= ts) & (index < te) & valid_t1
        train_pos = np.flatnonzero(train_mask)
        if len(train_pos) and (index[train_pos[-1]] - index[train_pos[0]]) >= pd.Timedelta(days=30 * min_train_months):
            n_cal = int(len(train_pos) * calibration_fraction)
            if n_cal > 0:
                cal_pos = train_pos[-n_cal:]
                cal_start = times[cal_pos[0]]
                fit_pos = train_pos[:-n_cal]
                # purga: el ajuste no puede tener etiquetas que se resuelvan dentro de calibración
                fit_pos = fit_pos[(t1_vals[fit_pos] < cal_start)]
            else:
                cal_pos, fit_pos = np.array([], dtype=int), train_pos
            test_pos = np.flatnonzero(test_mask)
            if len(test_pos):
                folds.append(Fold(k, fit_pos, cal_pos, test_pos, ts, te))
                k += 1
        ts = te
    return folds


def split_fit_validation(train_idx: np.ndarray, index: pd.DatetimeIndex, t1: pd.Series,
                         embargo: pd.Timedelta, fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Aparta del tramo de ajuste una validación para early stopping, distinta de la calibración.

    - validación: último ``fraction`` cronológico de ``train_idx`` (al menos una fila)
    - ajuste: el resto, sin las filas cuya etiqueta se resuelve (``t1``) a menos de ``embargo`` del
      inicio de la validación (purga)

    ``train_idx`` son posiciones enteras en ``index``/``t1`` en orden cronológico, p. ej. ``Fold.train_idx``.
    """
    train_idx = np.asarray(train_idx, dtype=int)
    if len(train_idx) < 2:
        raise ValueError("Se necesitan al menos dos filas para separar ajuste y validación.")
    index = pd.DatetimeIndex(index)
    t1 = pd.Series(t1, index=index)
    n_val = max(1, int(len(train_idx) * fraction))
    val, fit = train_idx[-n_val:], train_idx[:-n_val]
    fit = fit[(t1.iloc[fit] < index[val[0]] - embargo).to_numpy()]
    return fit, val

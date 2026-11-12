"""Imágenes OHLC causales y particiones purgadas para la CNN."""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import repeat
import numpy as np
import pandas as pd
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object

from quant_ai.labeling.labels import make_labels
from quant_ai.validation.splits import purged_walk_forward, split_fit_validation


def render_window(o, h, l, c, v, height=64, vol_height=16):
    """Media móvil expansiva de hasta N cierres, sin contexto anterior ni futuro."""
    o, h, l, c, v = [np.asarray(a, dtype=float) for a in (o, h, l, c, v)]
    if height < 2 or vol_height < 1 or not len(o):
        raise ValueError("Dimensiones de imagen inválidas.")
    if any(a.shape != o.shape or a.ndim != 1 or not np.isfinite(a).all() for a in (o,h,l,c,v)):
        raise ValueError("OHLCV debe ser finito, unidimensional y alineado.")
    if np.any(v < 0) or np.any(h < np.maximum(o,c)) or np.any(l > np.minimum(o,c)):
        raise ValueError("Velas OHLCV inválidas.")
    n = len(o)
    image = np.zeros((1, height + vol_height, 3*n), dtype=np.uint8)
    lo, hi = l.min(), h.max()
    def row(a):
        return np.rint((hi-a) / (hi-lo if hi > lo else 1) * (height-1)).astype(int).clip(0,height-1)
    ro,rh,rl,rc = map(row,(o,h,l,c))
    ma = row(np.cumsum(c) / np.arange(1,n+1))
    # Une los puntos de la media con intensidad menor que las velas.
    for i in range(n):
        if i:
            xs = np.arange(3*(i-1)+1,3*i+2)
            ys = np.rint(np.linspace(ma[i-1],ma[i],len(xs))).astype(int)
            image[0,ys,xs] = 128
        image[0,ma[i],3*i+1] = 128
    for i in range(n):
        x = 3*i
        image[0,rh[i]:rl[i]+1,x+1] = 255
        image[0,ro[i],x:x+2] = 255
        image[0,rc[i],x+1:x+3] = 255
        size = int(round(v[i]/v.max()*vol_height)) if v.max() else 0
        if size:
            image[0,height+vol_height-size:,x+1] = 255
    return image


def _render_chunk(values, ends, window):
    """Trabajo de cada proceso: numpy puro, sin torch."""
    return np.stack([render_window(*values[end-window+1:end+1].T) for end in ends])


def prerender_windows(values, positions, window=48, workers=1, chunk=2048):
    """Renderiza una sola vez la imagen de cada posición final, sin torch.

    Devuelve ``(rows, images)``: ``images[rows[p]]`` es idéntica byte a byte a ``render_window`` sobre la
    ventana que termina en ``p``, y ``rows[p] == -1`` si ``p`` no se pre-renderizó. ``rows`` es un array
    (no un dict) para que los procesos del DataLoader lo compartan sin copiar páginas de memoria.
    ``workers > 1`` reparte trozos entre procesos (acotado por ``os.cpu_count()``); el resultado es el mismo.
    """
    values = np.asarray(values)
    ends = np.unique(np.asarray(positions, dtype=np.int64))
    if len(ends) and (ends[0] < window-1 or ends[-1] >= len(values)):
        raise ValueError("Hay posiciones sin ventana completa o fuera de los valores.")
    rows = np.full(len(values), -1, dtype=np.int32)
    rows[ends] = np.arange(len(ends), dtype=np.int32)
    images = np.empty((len(ends), 1, 64+16, 3*window), dtype=np.uint8)  # alto por defecto de render_window
    # Cada trozo viaja con su propio contexto de ventana: no se copia la serie completa a cada proceso.
    starts = range(0, len(ends), chunk)
    slices = [values[ends[i]-window+1:ends[min(i+chunk, len(ends))-1]+1] for i in starts]
    local = [ends[i:i+chunk] - (ends[i]-window+1) for i in starts]
    workers = max(1, min(int(workers or 1), os.cpu_count() or 1, len(slices)))
    if workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for i, block in zip(starts, pool.map(_render_chunk, slices, local, repeat(window))):
                    images[i:i+len(block)] = block
            return rows, images
        except (OSError, RuntimeError) as error:  # BrokenProcessPool es RuntimeError: el entorno no deja crear procesos
            print(f"Aviso: el pre-renderizado en paralelo falló ({type(error).__name__}); se repite en un proceso.",
                  flush=True)
    for i, block in zip(starts, map(_render_chunk, slices, local, repeat(window))):
        images[i:i+len(block)] = block
    return rows, images


def attach_image_cache(parts, workers=1):
    """Pre-renderiza la unión de posiciones de varios tramos (fit/validación/calibración/test de los folds
    de un activo) y les asigna la caché. Todos deben compartir el mismo array OHLCV y la misma ventana.
    Devuelve ``(n_imágenes, bytes)``."""
    parts = list(parts)
    values, window = parts[0].values, parts[0].window
    if any(p.values is not values or p.window != window for p in parts):
        raise ValueError("La caché exige el mismo array OHLCV y la misma ventana en todos los tramos.")
    rows, images = prerender_windows(values, np.concatenate([p.positions for p in parts]), window, workers)
    for p in parts:
        p.rows, p.images = rows, images
    return len(images), images.nbytes + rows.nbytes


class ChartDataset(Dataset):
    """Renderiza cada ventana al solicitarla, o la lee de la caché de ``attach_image_cache`` (misma imagen);
    comparte únicamente OHLCV."""
    def __init__(self, values, labels, index, positions, window=48):
        if torch is None:
            raise ImportError("ChartDataset requiere PyTorch en Colab.")
        from quant_ai.training.sequence_data import WindowDataset
        checked = WindowDataset(values, labels, index, positions, window)
        self.__dict__.update(checked.__dict__)
        if self.values.shape[1] != 5:
            raise ValueError("Se esperan cinco columnas OHLCV.")
        self.rows = self.images = None
    @property
    def times(self):
        return self.index[self.positions]
    def __len__(self):
        return len(self.positions)
    def __getitem__(self, item):
        end = self.positions[item]
        row = self.rows[end] if self.rows is not None else -1
        image = self.images[row] if row >= 0 else render_window(*self.values[end-self.window+1:end+1].T)
        return torch.from_numpy(image.astype(np.float32)/255), torch.tensor(self.labels[end])


@dataclass
class ChartFold:
    symbol: str
    fold: object
    fit: ChartDataset
    validation: ChartDataset
    calibration: ChartDataset
    test: ChartDataset
    normalization: dict
    feature_names: list


def build_chart_folds(cfg, ds, window=48, last_folds=16, smoke=False):
    """Reserva validación y calibración distintas y purga sus fronteras."""
    if last_folds is not None and last_folds < 1:
        raise ValueError("last_folds debe ser positivo.")
    prices = ds.prices.loc[ds.prices.open_time < cfg.final_holdout_start].sort_values("open_time")
    labels = make_labels(prices, {"method":"forward_net_return", "horizon_bars":24}, .0026, freq=cfg.interval)
    frame = labels.loc[labels.y.notna() & (labels.t1 < cfg.final_holdout_start)]
    index = pd.DatetimeIndex(prices.open_time)
    values = prices[["open","high","low","close","volume"]].to_numpy(dtype=np.float32)
    ys = labels.y.reindex(index).to_numpy(dtype=np.float32)
    wf = cfg["splits"]["walk_forward"]
    embargo = pd.Timedelta(cfg.interval)*cfg["splits"]["embargo_bars"]
    folds = purged_walk_forward(frame.index, frame.t1, wf["first_test_start"], 3,
        wf["min_train_months"], embargo, wf["calibration_fraction"], end=index.max(),
        expanding=wf.get("expanding",True))
    if not folds:
        raise ValueError("No hay folds de desarrollo disponibles.")
    for fold in folds[-last_folds:] if last_folds else folds:
        fit, val = split_fit_validation(fold.train_idx, frame.index, frame.t1, embargo, .2)
        positions = [index.get_indexer(frame.index[p]) for p in (fit,val,fold.calib_idx)]
        positions.append(np.flatnonzero((index >= fold.test_start) & (index < fold.test_end)))
        parts = [ChartDataset(values,ys,index,p,window) for p in positions]
        if smoke:
            for part in parts:
                part.positions = part.positions[:512]
        if any(not len(part) for part in parts):
            raise ValueError("Tramo vacío tras purgar y excluir huecos.")
        yield ChartFold(ds.symbol,fold,*parts,{},["open","high","low","close","volume"])

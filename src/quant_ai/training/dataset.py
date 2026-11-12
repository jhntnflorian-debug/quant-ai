"""Construcción reproducible del dataset (características + etiquetas) por símbolo.

Por defecto recorta al periodo de desarrollo (antes del holdout final). Para usar el holdout hay que
pasar ``final=True`` explícitamente (lo hace solo ``scripts/run_experiments.py --final``).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_ai.config import Config
from quant_ai.data.store import load_all
from quant_ai.features.engineering import build_feature_set, feature_columns
from quant_ai.labeling.labels import make_labels
from quant_ai.validation.splits import guard_holdout, restrict_to_development


@dataclass
class SymbolDataset:
    symbol: str
    prices: pd.DataFrame      # OHLCV limpio (columnas Binance, open_time como columna)
    features: pd.DataFrame    # index=open_time
    labels: pd.DataFrame      # index=open_time; y, fwd_ret, t1
    feature_names: list[str]

    def frame(self) -> pd.DataFrame:
        """Características + etiquetas alineadas (filas con etiqueta válida)."""
        return self.features.join(self.labels, how="inner")


def build_datasets(cfg: Config, final: bool = False, raw: dict[str, pd.DataFrame] | None = None
                   ) -> dict[str, SymbolDataset]:
    raw = raw if raw is not None else load_all(cfg)
    hs = cfg.final_holdout_start
    feats = build_feature_set(raw, cfg["features"])
    out = {}
    for sym, df in raw.items():
        f = feats[sym]
        lab = make_labels(df, cfg["labeling"], cfg["labeling"]["cost_threshold_bps"] / 1e4,
                          atr_pct=f["atr_pct_14"], freq=cfg.interval)
        prices = df
        if not final:
            lab = restrict_to_development(lab, hs)
            f = f[f.index < hs]
            prices = df[df["open_time"] < hs]
            guard_holdout(f.index, hs)
            guard_holdout(lab.index, hs)
        out[sym] = SymbolDataset(sym, prices.reset_index(drop=True), f, lab, feature_columns(f))
    return out

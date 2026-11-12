"""Carga de configuración YAML y resolución de rutas del proyecto."""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"


def _deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Config:
    """Envoltura ligera sobre el dict de configuración con accesos tipados a lo crítico."""

    raw: dict[str, Any]
    home: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    # --- rutas -----------------------------------------------------------------
    def path(self, name: str) -> Path:
        p = Path(self.raw["paths"][name])
        p = p if p.is_absolute() else self.home / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def data_dir(self) -> Path:
        return self.path("data_dir")

    @property
    def models_dir(self) -> Path:
        return self.path("models_dir")

    @property
    def reports_dir(self) -> Path:
        return self.path("reports_dir")

    @property
    def state_dir(self) -> Path:
        return self.path("state_dir")

    # --- valores críticos ------------------------------------------------------
    @property
    def symbols(self) -> list[str]:
        return list(self.raw["market"]["symbols"])

    @property
    def interval(self) -> str:
        return self.raw["market"]["interval"]

    @property
    def final_holdout_start(self) -> pd.Timestamp:
        return pd.Timestamp(self.raw["splits"]["final_holdout_start"], tz="UTC")

    @property
    def round_trip_cost(self) -> float:
        c = self.raw["costs"]
        return 2 * (c["fee_bps"] + c["half_spread_bps"] + c["slippage_bps"]) / 1e4


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    """Carga la configuración base y aplica overrides opcionales.

    ``QUANT_AI_HOME`` permite redirigir datos/modelos (p. ej. a Google Drive en Colab).
    """
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if overrides:
        raw = _deep_update(raw, overrides)
    home = Path(os.environ.get("QUANT_AI_HOME") or PROJECT_ROOT)
    mode = os.environ.get("QUANT_AI_MODE")
    if mode:
        raw["project"]["mode"] = mode
    return Config(raw=raw, home=home)

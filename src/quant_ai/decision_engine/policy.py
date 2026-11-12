"""Política de decisión compartida por backtest y paper trading (misma lógica en ambos modos).

Entrada: probabilidad calibrada de que comprar supere costes, incertidumbre y estado previo.
Salida: peso objetivo long/flat con histéresis y abstención explícita.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PolicyParams:
    enter_prob: float = 0.55
    exit_prob: float = 0.45
    max_uncertainty: float = 0.15
    target_weight: float = 0.5

    @classmethod
    def from_config(cls, s: dict) -> "PolicyParams":
        return cls(s["enter_prob"], s["exit_prob"], s["max_uncertainty"], s["target_weight"])


def step(prob: float, uncertainty: float, in_position: bool, p: PolicyParams) -> tuple[str, float, str]:
    """Una decisión. Devuelve (acción, peso objetivo, motivo). Acciones: BUY, SELL, HOLD, CLOSE.

    Spot sin apalancamiento: SELL/CLOSE significan salir a efectivo; nunca abrir cortos.
    """
    if not np.isfinite(prob):
        return ("CLOSE" if in_position else "HOLD", 0.0, "sin predicción válida -> efectivo")
    unsure = np.isfinite(uncertainty) and uncertainty > p.max_uncertainty
    if in_position:
        if prob < p.exit_prob:
            return "SELL", 0.0, f"p={prob:.3f} < salida {p.exit_prob}"
        return "HOLD", p.target_weight, f"mantener: p={prob:.3f} >= salida {p.exit_prob}"
    if unsure:
        return "HOLD", 0.0, f"abstención: incertidumbre {uncertainty:.3f} > {p.max_uncertainty}"
    if prob >= p.enter_prob:
        return "BUY", p.target_weight, f"p={prob:.3f} >= entrada {p.enter_prob}"
    return "HOLD", 0.0, f"abstención: p={prob:.3f} < entrada {p.enter_prob}"


def stop_guard_step(guard: dict | None, bar_time: pd.Timestamp, prob: float, p: PolicyParams
                    ) -> tuple[dict | None, list[str]]:
    """Reglas de re-entrada tras stop-loss de ``run_backtest`` aplicadas a una vela decidida (paper trading).

    - Enfriamiento: no se entra mientras ``bar_time < cooldown_until`` (backtest: ``i < cooldown_until``).
    - Señal nueva: la política vectorial del backtest no sabe que hubo stop y sigue "en posición", así que su
      señal solo se apaga con p no finita o p < exit_prob (la incertidumbre no cuenta). La espera se levanta
      en la primera vela así, también durante el enfriamiento.
    Devuelve el estado actualizado (None si ya no bloquea nada) y los motivos que bloquean una entrada.
    """
    if not guard:
        return None, []
    need_fresh = bool(guard["need_fresh"]) and step(prob, np.nan, True, p)[1] > 0
    until = pd.Timestamp(guard["cooldown_until"])
    reasons = []
    if bar_time < until:
        reasons.append(f"enfriamiento tras stop-loss hasta la vela {until:%Y-%m-%d %H:%M} UTC")
    if need_fresh:
        reasons.append("esperando señal nueva tras stop-loss")
    return ({**guard, "need_fresh": need_fresh} if reasons else None), reasons


def stop_guard_start(bar_time: pd.Timestamp, prob: float, p: PolicyParams,
                     cooldown: pd.Timedelta) -> dict | None:
    """Estado tras un stop-loss ejecutado al decidir ``bar_time`` (la vela ``i`` del stop en el backtest):
    ``cooldown_until = i + N`` y espera de señal nueva, evaluada ya con la probabilidad de esa misma vela."""
    guard = {"stop_bar": bar_time.isoformat(), "cooldown_until": (bar_time + cooldown).isoformat(),
             "need_fresh": True}
    return stop_guard_step(guard, bar_time, prob, p)[0]


def vol_target_weights(vol_hourly: pd.Series, target_annual: float, max_weight: float, ann: int = 8760,
                       step: float = 0.1) -> pd.Series:
    """Peso por volatility targeting: min(max_weight, objetivo / volatilidad anualizada del activo).

    ``target_annual`` es la volatilidad anual objetivo que aporta CADA posición (peso × volatilidad),
    con tope ``max_weight`` (sin apalancamiento). Usa una volatilidad lenta (p. ej. 168 h) y cuantiza a
    ``step`` para no rebalancear por ruido: el peso solo cambia al cruzar un escalón."""
    ann_vol = vol_hourly * np.sqrt(ann)
    w = np.minimum(max_weight, target_annual / ann_vol.replace(0, np.nan))
    w = np.floor(w / step + 1e-9) * step
    return w.clip(lower=0, upper=max_weight).fillna(0.0)


def weights_from_probs(prob: pd.Series, uncertainty: pd.Series | None, p: PolicyParams) -> pd.Series:
    """Versión vectorial (con estado) de ``step`` para backtesting."""
    probs = prob.to_numpy(dtype=float)
    unc = uncertainty.reindex(prob.index).to_numpy(dtype=float) if uncertainty is not None else np.full(len(probs), np.nan)
    out = np.zeros(len(probs))
    inpos = False
    for i in range(len(probs)):
        _, w, _ = step(probs[i], unc[i], inpos, p)
        out[i] = w
        inpos = w > 0
    return pd.Series(out, index=prob.index)

"""Motor de decisiones: combina predicciones calibradas, régimen e incertidumbre en una decisión estructurada.

La decisión resultante pasa SIEMPRE por el motor de riesgo independiente antes de ejecutarse.
No se inventa confianza: solo se reportan probabilidades calibradas (Platt) del ensemble.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from quant_ai.decision_engine.policy import PolicyParams, step


@dataclass
class Decision:
    decision_id: str
    created_at: str
    bar_time: str
    symbol: str
    action: str                     # BUY | SELL | HOLD | CLOSE
    direction: str                  # long | flat
    horizon_bars: int
    target_weight: float
    current_weight: float
    size_qty: float
    notional: float
    ref_price: float
    order_type: str
    stop_loss: float | None
    probabilities: dict
    uncertainty: float | None
    exp_ret: float | None
    exp_vol: float | None
    regime: dict
    reasons: list[str] = field(default_factory=list)
    top_features: list[dict] = field(default_factory=list)
    risk: dict = field(default_factory=dict)
    executed: bool = False
    fill: dict | None = None
    model_version: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def regime_of(f_row: pd.Series) -> dict:
    """Régimen simple y causal a partir de la fila actual de características."""
    trend = "alcista" if f_row.get("ema_dist_168", 0) > 0.02 else ("bajista" if f_row.get("ema_dist_168", 0) < -0.02
                                                                    else "lateral")
    vr = f_row.get("vol_ratio_24_168", np.nan)
    vol = "alta_vol" if np.isfinite(vr) and vr > 1.5 else "normal_vol"
    return {"trend": trend, "vol": vol}


def top_contributions(model, X_row: pd.DataFrame, k: int = 5) -> list[dict]:
    """Contribuciones por característica del miembro LightGBM (TreeSHAP nativo, en log-odds).
    Son asociativas: describen el modelo, no prueban causalidad."""
    try:
        lgbm = model.members["lightgbm"]
        contrib = lgbm.booster_.predict(X_row[model.features], pred_contrib=True)[0][:-1]
    except Exception:
        return []
    mean, std = model.feature_stats.get("mean", {}), model.feature_stats.get("std", {})
    order = np.argsort(-np.abs(contrib))[:k]
    out = []
    for i in order:
        name = model.features[i]
        v = float(X_row[name].iloc[0]) if pd.notna(X_row[name].iloc[0]) else None
        z = None
        if v is not None and std.get(name) and np.isfinite(std[name]):
            z = (v - mean.get(name, 0.0)) / std[name]
        out.append({"feature": name, "value": v, "z": z, "contribution": float(contrib[i])})
    return out


def decide(symbol: str, bar_time: pd.Timestamp, X_row: pd.DataFrame, model, policy: PolicyParams,
           current_weight: float, position_qty: float, equity: float, price: float, horizon: int,
           atr_stop_mult: float, model_version: str | None = None) -> Decision:
    pred = model.predict(X_row).iloc[0]
    prob = float(pred["p_ensemble"])
    unc = float(pred["uncertainty"])
    in_pos = position_qty > 0
    action, target_w, why = step(prob, unc, in_pos, policy)
    f = X_row.iloc[0]
    reasons = [why]
    if action == "HOLD" and in_pos:
        target_w = current_weight  # mantener: no se rebalancea en cada vela
    qty_delta = (target_w - current_weight) * equity / price if price > 0 else 0.0
    if action in ("SELL", "CLOSE"):
        qty_delta = -position_qty
    atr_pct = float(f.get("atr_pct_14", np.nan))
    stop = price * (1 - atr_stop_mult * atr_pct) if action == "BUY" and np.isfinite(atr_pct) else None
    vol24 = float(f.get("vol_24", np.nan))
    return Decision(
        decision_id=str(uuid.uuid4()), created_at=datetime.now(timezone.utc).isoformat(),
        bar_time=str(bar_time), symbol=symbol, action=action, direction="long" if target_w > 0 else "flat",
        horizon_bars=horizon, target_weight=float(target_w), current_weight=float(current_weight),
        size_qty=float(abs(qty_delta)), notional=float(abs(qty_delta) * price), ref_price=float(price),
        order_type="market", stop_loss=stop,
        probabilities={k: float(pred[k]) for k in ("p_logistic", "p_lightgbm", "p_ensemble") if k in pred},
        uncertainty=unc, exp_ret=float(pred["exp_ret"]) if "exp_ret" in pred else None,
        exp_vol=float(vol24 * np.sqrt(horizon)) if np.isfinite(vol24) else None,
        regime=regime_of(f), reasons=reasons, top_features=top_contributions(model, X_row),
        model_version=model_version)

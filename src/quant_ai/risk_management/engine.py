"""Motor de riesgo INDEPENDIENTE con autoridad de veto.

No depende del modelo ni del motor de decisiones: recibe una propuesta de orden y el estado de
cartera/mercado, y devuelve aprobada/bloqueada con motivos. Reducir exposición (vender) siempre se
permite salvo que no haya posición; aumentar exposición pasa por todos los controles.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class RiskLimits:
    max_order_pct_equity: float = 0.55
    max_order_notional: float = 20000.0
    max_position_weight: float = 0.5
    max_gross_exposure: float = 1.0
    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.15
    max_data_age_minutes: float = 90
    max_volatility_pct: float = 0.04
    ood_zscore: float = 6.0
    ood_max_features: int = 3
    max_api_errors: int = 5
    kill_switch_file: str = "KILL_SWITCH"

    @classmethod
    def from_config(cls, r: dict) -> "RiskLimits":
        return cls(**{k: v for k, v in r.items() if k in cls.__dataclass_fields__})


@dataclass
class OrderProposal:
    symbol: str
    side: str                      # BUY | SELL
    notional: float                # valor estimado en USDT
    target_weight: float
    client_order_id: str
    price: float


@dataclass
class MarketState:
    last_bar_close_time: pd.Timestamp
    realized_vol_24: float          # desviación de retornos horarios (24h)
    features: dict | None = None    # vector de características para OOD
    spread_bps: float | None = None


@dataclass
class PortfolioState:
    equity: float
    peak_equity: float
    day_start_equity: float
    position_weights: dict          # símbolo -> peso actual
    gross_exposure: float


@dataclass
class RiskDecision:
    approved: bool
    reasons: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)
    emergency: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class RiskEngine:
    def __init__(self, limits: RiskLimits, feature_stats: dict | None = None, state_dir: Path | None = None,
                 root: Path | None = None):
        self.limits = limits
        self.feature_stats = feature_stats or {}
        self.root = Path(root) if root else Path.cwd()
        self.state_dir = Path(state_dir) if state_dir else None
        self.api_errors = 0
        self.seen_order_ids: set[str] = set()
        self.emergency_reason: str | None = None
        self._load_state()

    # ------------------------------------------------------------------ persistencia
    def _state_file(self) -> Path | None:
        return self.state_dir / "risk_state.json" if self.state_dir else None

    def _load_state(self) -> None:
        p = self._state_file()
        if p and p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            self.seen_order_ids = set(d.get("seen_order_ids", [])[-5000:])
            self.api_errors = int(d.get("api_errors", 0))
            self.emergency_reason = d.get("emergency_reason")

    def save_state(self) -> None:
        p = self._state_file()
        if p:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"seen_order_ids": sorted(self.seen_order_ids)[-5000:],
                                     "api_errors": self.api_errors,
                                     "emergency_reason": self.emergency_reason,
                                     "updated_at": datetime.now(timezone.utc).isoformat()}, indent=2),
                         encoding="utf-8")

    # ------------------------------------------------------------------ controles globales
    def kill_switch_active(self) -> bool:
        return (self.root / self.limits.kill_switch_file).exists()

    def record_api_error(self) -> None:
        self.api_errors += 1
        if self.api_errors >= self.limits.max_api_errors:
            self.enter_emergency(f"{self.api_errors} errores consecutivos de API/broker")

    def record_api_success(self) -> None:
        self.api_errors = 0

    def enter_emergency(self, reason: str) -> None:
        self.emergency_reason = reason

    def clear_emergency(self) -> None:
        """Solo manual: un humano revisa y limpia el modo de emergencia."""
        self.emergency_reason = None
        self.api_errors = 0

    def ood_count(self, features: dict | None) -> tuple[int, list[str]]:
        if not features or not self.feature_stats:
            return 0, []
        mean, std = self.feature_stats.get("mean", {}), self.feature_stats.get("std", {})
        bad = []
        for k, v in features.items():
            m, s = mean.get(k), std.get(k)
            if m is None or s is None or not np.isfinite(s) or s == 0 or v is None or not np.isfinite(v):
                continue
            if abs((v - m) / s) > self.limits.ood_zscore:
                bad.append(k)
        return len(bad), bad

    # ------------------------------------------------------------------ evaluación
    def evaluate(self, order: OrderProposal, market: MarketState, pf: PortfolioState,
                 now: pd.Timestamp | None = None) -> RiskDecision:
        L = self.limits
        now = now or pd.Timestamp.now(tz="UTC")
        reasons, checks = [], {}
        reducing = order.side == "SELL"

        # Duplicados (idempotencia): aplica a todo
        if order.client_order_id in self.seen_order_ids:
            return RiskDecision(False, [f"orden duplicada {order.client_order_id}"], {"duplicate": True})

        if reducing:
            ok = pf.position_weights.get(order.symbol, 0.0) > 0
            dec = RiskDecision(ok, [] if ok else ["no hay posición que vender"], {"reducing": True})
            if ok:
                self.seen_order_ids.add(order.client_order_id)
            return dec

        # --- controles que bloquean NUEVA exposición ---
        checks["kill_switch"] = self.kill_switch_active()
        if checks["kill_switch"]:
            reasons.append("KILL SWITCH activo: no se abren posiciones")
        checks["emergency"] = self.emergency_reason
        if self.emergency_reason:
            reasons.append(f"modo emergencia: {self.emergency_reason}")

        age_min = (now - market.last_bar_close_time).total_seconds() / 60
        checks["data_age_min"] = round(age_min, 1)
        if age_min > L.max_data_age_minutes:
            reasons.append(f"datos obsoletos: última vela cerró hace {age_min:.0f} min > {L.max_data_age_minutes}")

        checks["realized_vol_24"] = market.realized_vol_24
        if not np.isfinite(market.realized_vol_24) or market.realized_vol_24 > L.max_volatility_pct:
            reasons.append(f"volatilidad {market.realized_vol_24:.4f} > {L.max_volatility_pct}")

        if market.spread_bps is not None:
            checks["spread_bps"] = market.spread_bps
            if market.spread_bps > 20:
                reasons.append(f"spread {market.spread_bps:.1f} bps demasiado amplio")

        n_ood, bad = self.ood_count(market.features)
        checks["ood_features"] = bad
        if n_ood >= L.ood_max_features:
            reasons.append(f"entrada fuera de distribución ({n_ood} características con |z|>{L.ood_zscore})")

        checks["order_notional"] = order.notional
        if order.notional > L.max_order_notional:
            reasons.append(f"nocional {order.notional:.0f} > tope absoluto por orden {L.max_order_notional:.0f}")
        if pf.equity > 0 and order.notional > L.max_order_pct_equity * pf.equity:
            reasons.append(f"nocional {order.notional:.0f} > {L.max_order_pct_equity:.0%} del capital")
        if order.target_weight > L.max_position_weight + 1e-9:
            reasons.append(f"peso objetivo {order.target_weight:.2f} > máximo por activo {L.max_position_weight}")
        new_gross = pf.gross_exposure + max(0.0, order.target_weight - pf.position_weights.get(order.symbol, 0.0))
        checks["gross_after"] = round(new_gross, 4)
        if new_gross > L.max_gross_exposure + 1e-9:
            reasons.append(f"exposición bruta {new_gross:.2f} > {L.max_gross_exposure} (sin apalancamiento)")

        dd = pf.equity / pf.peak_equity - 1 if pf.peak_equity > 0 else 0.0
        checks["drawdown"] = round(dd, 4)
        if dd <= -L.max_drawdown_pct:
            reasons.append(f"drawdown {dd:.1%} supera el límite {-L.max_drawdown_pct:.0%}")
            self.enter_emergency(f"drawdown {dd:.1%}")
        day = pf.equity / pf.day_start_equity - 1 if pf.day_start_equity > 0 else 0.0
        checks["daily_pnl"] = round(day, 4)
        if day <= -L.max_daily_loss_pct:
            reasons.append(f"pérdida diaria {day:.1%} supera el límite {-L.max_daily_loss_pct:.0%}")

        approved = not reasons
        if approved:
            self.seen_order_ids.add(order.client_order_id)
        return RiskDecision(approved, reasons, checks, emergency=bool(self.emergency_reason))


def backtest_risk_hook(limits: RiskLimits, vol24: dict[str, pd.Series], review_bars: int = 720):
    """Adaptador del motor de riesgo para el backtester (subconjunto evaluable con datos históricos:
    volatilidad, peso máximo y drawdown). Mantiene paridad de reglas con paper trading.

    En vivo, superar el drawdown máximo activa el modo emergencia y exige revisión humana. En el
    backtest se emula así: se liquida, se detiene la apertura de posiciones durante ``review_bars``
    velas (30 días por defecto) y luego se reinicia el pico de referencia (supuesto documentado).
    """
    st = {"peak": None, "halt_until": None}

    def hook(ts, sym, w, state):
        eq = state["equity"]
        st["peak"] = eq if st["peak"] is None else max(st["peak"], eq)
        i = state["i"]
        if st["halt_until"] is not None:
            if i < st["halt_until"]:
                return 0.0, "drawdown_halt"
            st["halt_until"], st["peak"] = None, eq
        if eq / st["peak"] - 1 <= -limits.max_drawdown_pct:
            st["halt_until"] = i + review_bars
            return 0.0, "drawdown_halt"
        if w <= 0:
            return w, None
        v = vol24[sym].get(ts, np.nan)
        if state["qty"] == 0 and (not np.isfinite(v) or v > limits.max_volatility_pct):
            return 0.0, "volatility"
        return min(w, limits.max_position_weight), None

    return hook

"""Motor de backtesting multi-activo, long/flat, vela a vela y auditable.

Secuencia por cada marca temporal T (vela [T, T+1h)):
1. APERTURA de T: se ejecutan las órdenes decididas al cierre de T-1 al precio ``open[T]`` más
   medio spread + slippage (compras pagan más, ventas reciben menos) y comisión sobre el nocional.
   El tamaño ejecutable se limita a ``max_participation * volume[T-1]`` (volumen ya conocido);
   el resto no se ejecuta (fill parcial) y se descarta.
2. DURANTE T: si hay stop-loss y ``low[T] <= stop`` se vende a ``min(open[T], stop)`` menos costes
   (si el precio abre por debajo del stop, se ejecuta a la apertura: no hay ejecución "mágica" en el stop).
3. CIERRE de T: se valora la cartera a ``close[T]`` y se lee el peso objetivo decidido con información
   hasta el cierre de T. La orden resultante se ejecutará en la apertura de T+1.

Nunca se usa un precio que no estaba disponible al momento de la señal.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CostModel:
    fee_bps: float = 10.0
    half_spread_bps: float = 1.0
    slippage_bps: float = 2.0
    max_participation: float = 0.01

    @classmethod
    def from_config(cls, c: dict, multiplier: float = 1.0) -> "CostModel":
        return cls(c["fee_bps"] * multiplier, c["half_spread_bps"] * multiplier,
                   c["slippage_bps"] * multiplier, c["max_participation"])

    @property
    def impact(self) -> float:
        return (self.half_spread_bps + self.slippage_bps) / 1e4

    @property
    def fee(self) -> float:
        return self.fee_bps / 1e4


@dataclass
class Rules:
    lot_step: float = 1e-5
    min_notional: float = 5.0

    def round_qty(self, qty: float) -> float:
        return np.floor(qty / self.lot_step + 1e-9) * self.lot_step


@dataclass
class BacktestResult:
    equity: pd.DataFrame          # index=open_time; equity, cash, gross_exposure, pos_<sym>
    fills: pd.DataFrame           # cada ejecución individual
    trades: pd.DataFrame          # operaciones de ida y vuelta (entrada -> salida)
    meta: dict = field(default_factory=dict)


def run_backtest(prices: dict[str, pd.DataFrame], target_weights: dict[str, pd.Series],
                 costs: CostModel, rules: dict[str, Rules] | None = None, initial_cash: float = 10_000.0,
                 stops: dict[str, pd.Series] | None = None, rebalance_band: float = 0.05,
                 risk_hook=None, stop_cooldown_bars: int = 24) -> BacktestResult:
    """Ejecuta el backtest.

    prices: símbolo -> DataFrame con columnas open_time, open, high, low, close, volume.
    target_weights: símbolo -> Serie (index=open_time) con el peso objetivo en [0, 1] decidido al CIERRE.
    stops: símbolo -> Serie con distancia de stop en fracción del precio (p. ej. 3*ATR/close) conocida al cierre.
    risk_hook: callable(ts, sym, target_w, state) -> (target_w, motivo|None). Permite al motor de riesgo vetar.
    stop_cooldown_bars: tras un stop-loss no se reabre el activo durante N velas (evita re-entradas en bucle);
    además, tras un stop se exige una señal NUEVA (el peso objetivo debe pasar por 0 antes de volver a entrar).
    """
    rules = rules or {}
    syms = list(prices)
    idx = sorted(set().union(*[set(p["open_time"]) for p in prices.values()]))
    idx = pd.DatetimeIndex(idx)
    n = len(idx)
    arr = {}
    for s in syms:
        p = prices[s].set_index("open_time").reindex(idx)
        w = target_weights[s].reindex(idx)
        st = stops[s].reindex(idx) if stops and s in stops else pd.Series(np.nan, index=idx)
        arr[s] = {k: p[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close", "volume")}
        arr[s]["w"] = w.to_numpy(dtype=float)
        arr[s]["stop_dist"] = st.to_numpy(dtype=float)

    cash = float(initial_cash)
    qty = {s: 0.0 for s in syms}
    last_px = {s: np.nan for s in syms}
    pending: dict[str, float] = {}           # orden en cantidad (con signo) a ejecutar en la próxima apertura
    pending_reason: dict[str, str] = {}
    stop_px = {s: np.nan for s in syms}
    cooldown_until = {s: -1 for s in syms}
    need_fresh = {s: False for s in syms}  # tras un stop: exigir señal NUEVA (peso 0 y luego > 0)
    open_trade: dict[str, dict] = {}
    fills, trades = [], []
    eq_rows = np.zeros((n, 3 + len(syms)))
    peak = initial_cash

    def _fill(i, s, side_qty, ref_price, reason, prev_vol):
        nonlocal cash
        r = rules.get(s, Rules())
        q = abs(side_qty)
        if np.isfinite(prev_vol):
            q = min(q, costs.max_participation * prev_vol)
        if side_qty > 0:
            px = ref_price * (1 + costs.impact)
            q = min(q, cash / (px * (1 + costs.fee)))  # restricción de capital
        else:
            px = ref_price * (1 - costs.impact)
            q = min(q, qty[s])
        q = r.round_qty(q) if side_qty > 0 or q < qty[s] - 1e-12 else q  # vender todo no se redondea
        notional = q * px
        if q <= 0 or (notional < r.min_notional and not (side_qty < 0 and q >= qty[s] - 1e-12)):
            return 0.0
        fee = notional * costs.fee
        if side_qty > 0:
            cash -= notional + fee
            qty[s] += q
        else:
            cash += notional - fee
            qty[s] -= q
            if qty[s] < 1e-12:
                qty[s] = 0.0
        fills.append({"time": idx[i], "symbol": s, "side": "BUY" if side_qty > 0 else "SELL", "qty": q,
                      "ref_price": ref_price, "fill_price": px, "notional": notional, "fee": fee,
                      "impact_cost": abs(px - ref_price) * q, "requested_qty": abs(side_qty), "reason": reason})
        # contabilidad de operaciones de ida y vuelta
        if side_qty > 0:
            t = open_trade.setdefault(s, {"symbol": s, "entry_time": idx[i], "qty": 0.0, "cost": 0.0, "fees": 0.0})
            t["qty"] += q
            t["cost"] += notional
            t["fees"] += fee
        elif s in open_trade:
            t = open_trade[s]
            frac = q / t["qty"] if t["qty"] > 0 else 1.0
            t.setdefault("proceeds", 0.0)
            t["proceeds"] += notional
            t["fees"] += fee
            t.setdefault("sold_cost", 0.0)
            t["sold_cost"] += t["cost"] * frac
            t["cost"] -= t["cost"] * frac
            t["qty"] -= q
            if qty[s] == 0.0:
                pnl = t["proceeds"] - t["sold_cost"] - t["fees"]
                trades.append({"symbol": s, "entry_time": t["entry_time"], "exit_time": idx[i],
                               "entry_value": t["sold_cost"], "exit_value": t["proceeds"], "fees": t["fees"],
                               "pnl": pnl, "return": pnl / t["sold_cost"] if t["sold_cost"] else 0.0,
                               "bars_held": int((idx[i] - t["entry_time"]) / pd.Timedelta("1h")),
                               "exit_reason": reason})
                del open_trade[s]
        return q

    for i in range(n):
        # 1) ejecuciones a la apertura
        for s, oq in list(pending.items()):
            o = arr[s]["open"][i]
            if not np.isfinite(o):
                continue  # sin vela: la orden espera (no se inventa precio)
            prev_vol = arr[s]["volume"][i - 1] if i > 0 else np.nan
            _fill(i, s, oq, o, pending_reason.get(s, "signal"), prev_vol)
            del pending[s]
            pending_reason.pop(s, None)
            if qty[s] > 0 and np.isfinite(arr[s]["stop_dist"][i - 1] if i > 0 else np.nan) and np.isnan(stop_px[s]):
                stop_px[s] = o * (1 - arr[s]["stop_dist"][i - 1])
            if qty[s] == 0:
                stop_px[s] = np.nan
        # 2) stops intrabar
        for s in syms:
            if qty[s] > 0 and np.isfinite(stop_px[s]):
                lo, o = arr[s]["low"][i], arr[s]["open"][i]
                if np.isfinite(lo) and lo <= stop_px[s]:
                    _fill(i, s, -qty[s], min(o, stop_px[s]), "stop_loss", np.nan)
                    stop_px[s] = np.nan
                    pending.pop(s, None)
                    cooldown_until[s] = i + stop_cooldown_bars
                    need_fresh[s] = True
        # 3) cierre: valoración y nuevas órdenes
        for s in syms:
            c = arr[s]["close"][i]
            if np.isfinite(c):
                last_px[s] = c
        pos_val = {s: qty[s] * last_px[s] if qty[s] > 0 else 0.0 for s in syms}
        equity = cash + sum(pos_val.values())
        peak = max(peak, equity)
        gross = sum(pos_val.values()) / equity if equity > 0 else 0.0
        eq_rows[i, :3] = (equity, cash, gross)
        for j, s in enumerate(syms):
            eq_rows[i, 3 + j] = pos_val[s] / equity if equity > 0 else 0.0
        for s in syms:
            w = arr[s]["w"][i]
            c = last_px[s]
            if not np.isfinite(w) or not np.isfinite(c) or not np.isfinite(arr[s]["close"][i]):
                continue
            reason = "signal"
            if risk_hook is not None:
                state = {"equity": equity, "peak": peak, "cash": cash, "gross": gross, "qty": qty[s],
                         "weight": pos_val[s] / equity if equity > 0 else 0.0, "i": i}
                w, why = risk_hook(idx[i], s, w, state)
                if why:
                    reason = f"risk:{why}"
            # Regla de re-entrada tras stop: la SEÑAL de la estrategia (no el 0 forzado por el enfriamiento)
            # debe apagarse y volver a encenderse. Así una política con histéresis no recompra con p entre
            # exit_prob y enter_prob.
            signal_w = w
            if i < cooldown_until[s] and qty[s] == 0:
                w, reason = 0.0, "stop_cooldown"
            if need_fresh[s] and qty[s] == 0:
                if signal_w <= 0:
                    need_fresh[s] = False
                else:
                    w, reason = 0.0, "esperando_senal_nueva"
            cur_w = pos_val[s] / equity if equity > 0 else 0.0
            if w <= 0 and qty[s] > 0:
                pending[s], pending_reason[s] = -qty[s], ("exit" if reason == "signal" else reason)
            elif w > 0 and (qty[s] == 0 or abs(w - cur_w) > rebalance_band):
                target_qty = w * equity / c
                pending[s], pending_reason[s] = target_qty - qty[s], reason
            else:
                pending.pop(s, None)

    equity_df = pd.DataFrame(eq_rows, index=idx, columns=["equity", "cash", "gross_exposure"] +
                             [f"w_{s}" for s in syms])
    fills_df = pd.DataFrame(fills)
    trades_df = pd.DataFrame(trades)
    # posiciones abiertas al final: se reportan, no se cierran artificialmente
    open_positions = {s: {"qty": qty[s], "mark": last_px[s]} for s in syms if qty[s] > 0}
    return BacktestResult(equity_df, fills_df, trades_df, {"open_positions": open_positions,
                                                           "initial_cash": initial_cash})

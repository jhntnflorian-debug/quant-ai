"""Broker de paper trading local: ejecuta órdenes simuladas contra precios REALES de Binance.

- Precio de compra = ask actual (bookTicker público) + slippage; venta = bid - slippage; comisión parametrizada.
- Estado persistente en ``state/paper/portfolio.json`` y libro de órdenes en ``state/paper/orders.jsonl``.
- Idempotente: un ``client_order_id`` repetido devuelve la orden existente sin volver a ejecutar.
- No requiere claves de API.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

from quant_ai.brokers.base import Broker, BrokerError, OrderResult, Quote
from quant_ai.data.binance import REST_BASE, USER_AGENT


class LocalPaperBroker(Broker):
    name = "local_paper"

    def __init__(self, state_dir: Path, initial_cash: float, fee_bps: float, slippage_bps: float,
                 lot_steps: dict[str, float], min_notional: dict[str, float], session: requests.Session | None = None,
                 quote_fn=None):
        self.dir = Path(state_dir) / "paper"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fee = fee_bps / 1e4
        self.slip = slippage_bps / 1e4
        self.lot_steps, self.min_notional = lot_steps, min_notional
        self.session = session or requests.Session()
        self._quote_fn = quote_fn  # inyectable para pruebas
        self.portfolio_path = self.dir / "portfolio.json"
        self.orders_path = self.dir / "orders.jsonl"
        if not self.portfolio_path.exists():
            now = datetime.now(timezone.utc)
            self._save({"mode": "paper", "cash": float(initial_cash), "positions": {}, "equity": float(initial_cash),
                        "peak_equity": float(initial_cash), "day_start_equity": float(initial_cash),
                        "day": now.date().isoformat(), "initial_cash": float(initial_cash),
                        "model_version": None, "model_status": "none", "updated_at": now.isoformat()})

    # ------------------------------------------------------------------ estado
    def load(self) -> dict:
        return json.loads(self.portfolio_path.read_text(encoding="utf-8"))

    def _save(self, pf: dict) -> None:
        tmp = self.portfolio_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(pf, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.portfolio_path)  # escritura atómica

    def save(self, pf: dict) -> None:
        pf["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save(pf)

    # ------------------------------------------------------------------ API Broker
    def get_quote(self, symbol: str) -> Quote:
        if self._quote_fn:
            return self._quote_fn(symbol)
        try:
            r = self.session.get(f"{REST_BASE}/api/v3/ticker/bookTicker", params={"symbol": symbol}, timeout=10,
                                 headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
            d = r.json()
            return Quote(symbol, float(d["bidPrice"]), float(d["askPrice"]))
        except (requests.RequestException, KeyError, ValueError) as e:
            raise BrokerError(f"No se pudo obtener cotización de {symbol}: {e}") from e

    def get_balances(self) -> dict:
        pf = self.load()
        return {"cash": pf["cash"], "positions": {s: p["qty"] for s, p in pf["positions"].items()}}

    def _orders(self) -> dict[str, dict]:
        if not self.orders_path.exists():
            return {}
        out = {}
        for line in self.orders_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                o = json.loads(line)
                out[o["client_order_id"]] = o
        return out

    def get_order(self, client_order_id: str) -> OrderResult | None:
        o = self._orders().get(client_order_id)
        if not o:
            return None
        return OrderResult(o["client_order_id"], o["symbol"], o["side"], o["qty_requested"], o["qty_filled"],
                           o["avg_price"], o["fee"], o["status"], o["broker"], o["created_at"], o.get("message", ""))

    def place_market_order(self, symbol: str, side: str, qty: float, client_order_id: str,
                           stop_price: float | None = None) -> OrderResult:
        existing = self.get_order(client_order_id)
        if existing:
            existing.status, existing.message = "duplicate", "client_order_id ya ejecutado; no se repite"
            return existing
        q = self.get_quote(symbol)
        pf = self.load()
        step = self.lot_steps.get(symbol, 1e-5)
        pos = pf["positions"].get(symbol, {"qty": 0.0, "avg_price": 0.0, "stop_price": None, "opened_at": None})
        if side == "BUY":
            px = q.ask * (1 + self.slip)
            qty = min(qty, pf["cash"] / (px * (1 + self.fee)))
            qty = float(np.floor(qty / step + 1e-9) * step)
        else:
            px = q.bid * (1 - self.slip)
            qty = min(qty, pos["qty"])
            if qty < pos["qty"]:
                qty = float(np.floor(qty / step + 1e-9) * step)
        notional = qty * px
        if qty <= 0 or (notional < self.min_notional.get(symbol, 5.0) and not (side == "SELL" and qty >= pos["qty"])):
            res = OrderResult(client_order_id, symbol, side, qty, 0.0, px, 0.0, "rejected", self.name,
                              message=f"nocional {notional:.2f} por debajo del mínimo o sin saldo")
        else:
            fee = notional * self.fee
            if side == "BUY":
                new_qty = pos["qty"] + qty
                pos["avg_price"] = (pos["avg_price"] * pos["qty"] + px * qty) / new_qty
                pos["qty"] = new_qty
                pos["opened_at"] = pos.get("opened_at") or datetime.now(timezone.utc).isoformat()
                if stop_price:
                    pos["stop_price"] = stop_price
                pf["cash"] -= notional + fee
            else:
                pos["qty"] -= qty
                pf["cash"] += notional - fee
            if pos["qty"] <= 1e-12:
                pf["positions"].pop(symbol, None)
            else:
                pf["positions"][symbol] = pos
            self.save(pf)
            res = OrderResult(client_order_id, symbol, side, qty, qty, px, fee, "filled", self.name,
                              message=f"bid={q.bid} ask={q.ask} spread={q.spread_bps:.2f}bps")
        with open(self.orders_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(res.to_dict(), ensure_ascii=False) + "\n")
        return res

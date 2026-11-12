"""Adaptador para Binance Spot TESTNET (fondos virtuales). Opcional.

- Claves desde variables de entorno BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET
  (créalas en https://testnet.binance.vision/). Nunca se imprimen ni se registran.
- Solo acepta la URL del testnet: cualquier otra base URL se rechaza por código.
- Idempotencia: antes de enviar, consulta si ya existe una orden con ese newClientOrderId.
- El testnet se reinicia periódicamente (borra saldos y órdenes): la reconciliación lo tolera.
"""
from __future__ import annotations

import hashlib
import hmac
import math
import os
import time
from urllib.parse import urlencode

import requests

from quant_ai.brokers.base import Broker, BrokerError, OrderResult, Quote

TESTNET_BASE = "https://testnet.binance.vision"


class BinanceSpotTestnetBroker(Broker):
    name = "binance_testnet"

    def __init__(self, base_url: str = TESTNET_BASE, session: requests.Session | None = None,
                 quote_asset: str = "USDT", lot_steps: dict[str, float] | None = None,
                 min_notional: dict[str, float] | None = None):
        if base_url.rstrip("/") != TESTNET_BASE:
            raise BrokerError("Este adaptador solo opera contra el TESTNET de Binance.")
        self.base = base_url.rstrip("/")
        self.key = os.environ.get("BINANCE_TESTNET_API_KEY", "")
        self._secret = os.environ.get("BINANCE_TESTNET_API_SECRET", "").encode()
        if not self.key or not self._secret:
            raise BrokerError("Faltan BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET en el entorno (.env).")
        self.session = session or requests.Session()
        self.quote_asset = quote_asset
        self.lot_steps = lot_steps or {}
        self.min_notional = min_notional or {}

    def __repr__(self) -> str:  # evita filtrar secretos en logs
        return f"BinanceSpotTestnetBroker(base={self.base!r}, key=***)"

    def _signed(self, method: str, path: str, params: dict) -> dict:
        params = {**params, "timestamp": int(time.time() * 1000), "recvWindow": 5000}
        q = urlencode(params)
        sig = hmac.new(self._secret, q.encode(), hashlib.sha256).hexdigest()
        try:
            r = self.session.request(method, f"{self.base}{path}?{q}&signature={sig}",
                                     headers={"X-MBX-APIKEY": self.key}, timeout=15)
        except requests.RequestException as e:
            raise BrokerError(f"Error de red con el testnet: {e}") from e
        if r.status_code >= 400:
            raise BrokerError(f"Testnet {r.status_code}: {r.text[:200]}")
        return r.json()

    def get_quote(self, symbol: str) -> Quote:
        try:
            r = self.session.get(f"{self.base}/api/v3/ticker/bookTicker", params={"symbol": symbol}, timeout=10)
            r.raise_for_status()
            d = r.json()
            return Quote(symbol, float(d["bidPrice"]), float(d["askPrice"]))
        except (requests.RequestException, KeyError, ValueError) as e:
            raise BrokerError(f"Cotización testnet falló: {e}") from e

    def get_balances(self) -> dict:
        acc = self._signed("GET", "/api/v3/account", {})
        bal = {b["asset"]: float(b["free"]) + float(b["locked"]) for b in acc["balances"]}
        return {"cash": bal.get(self.quote_asset, 0.0),
                "positions": {f"{a}{self.quote_asset}": q for a, q in bal.items() if a != self.quote_asset and q > 0}}

    def get_order(self, client_order_id: str, symbol: str | None = None) -> OrderResult | None:
        if not symbol:
            return None
        try:
            o = self._signed("GET", "/api/v3/order", {"symbol": symbol, "origClientOrderId": client_order_id})
        except BrokerError as e:
            if "-2013" in str(e):  # orden no existe
                return None
            raise
        return self._to_result(o)

    def _to_result(self, o: dict) -> OrderResult:
        filled = float(o.get("executedQty", 0))
        quote = float(o.get("cummulativeQuoteQty", 0))
        fee = sum(float(f.get("commission", 0)) for f in o.get("fills", []))
        status = {"FILLED": "filled", "PARTIALLY_FILLED": "partial"}.get(o.get("status"), "rejected")
        return OrderResult(o["clientOrderId"], o["symbol"], o["side"], float(o.get("origQty", 0)), filled,
                           quote / filled if filled else 0.0, fee, status, self.name)

    def round_qty(self, symbol: str, qty: float) -> float:
        """Redondea hacia abajo al LOT_SIZE del instrumento (Binance rechaza cantidades fuera de paso)."""
        step = self.lot_steps.get(symbol, 1e-5)
        return math.floor(qty / step + 1e-9) * step

    def place_market_order(self, symbol: str, side: str, qty: float, client_order_id: str, **_) -> OrderResult:
        prev = self.get_order(client_order_id, symbol)
        if prev:
            prev.status = "duplicate"
            return prev
        qty = self.round_qty(symbol, qty)
        ref = self.get_quote(symbol).mid
        if qty <= 0 or qty * ref < self.min_notional.get(symbol, 5.0):
            return OrderResult(client_order_id, symbol, side, qty, 0.0, ref, 0.0, "rejected", self.name,
                               message="cantidad bajo el lote o nocional mínimo")
        decimals = max(0, -int(math.floor(math.log10(self.lot_steps.get(symbol, 1e-5)))))
        o = self._signed("POST", "/api/v3/order", {"symbol": symbol, "side": side, "type": "MARKET",
                                                    "quantity": f"{qty:.{decimals}f}",
                                                    "newClientOrderId": client_order_id,
                                                    "newOrderRespType": "FULL"})
        return self._to_result(o)

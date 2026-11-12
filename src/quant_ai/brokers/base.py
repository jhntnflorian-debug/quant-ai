"""Interfaz común de brokers: el motor de IA no depende del proveedor concreto."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


class BrokerError(RuntimeError):
    """Error del broker/API (desconexión, rechazo, respuesta inválida)."""


class LiveTradingBlocked(PermissionError):
    """El modo LIVE está bloqueado hasta autorización expresa del propietario."""


@dataclass
class Quote:
    symbol: str
    bid: float
    ask: float
    time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread_bps(self) -> float:
        return (self.ask - self.bid) / self.mid * 1e4 if self.mid > 0 else float("inf")


@dataclass
class OrderResult:
    client_order_id: str
    symbol: str
    side: str
    qty_requested: float
    qty_filled: float
    avg_price: float
    fee: float
    status: str               # filled | partial | rejected | duplicate
    broker: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class Broker(ABC):
    name: str = "base"

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote: ...

    @abstractmethod
    def get_balances(self) -> dict:
        """{'cash': float, 'positions': {symbol: qty}}"""

    @abstractmethod
    def place_market_order(self, symbol: str, side: str, qty: float, client_order_id: str) -> OrderResult: ...

    @abstractmethod
    def get_order(self, client_order_id: str) -> OrderResult | None: ...

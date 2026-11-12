"""Selección de broker según el modo. LIVE está bloqueado por código."""
from __future__ import annotations

import os

from quant_ai.brokers.base import Broker, LiveTradingBlocked
from quant_ai.config import Config


def get_broker(cfg: Config) -> Broker:
    mode = cfg["project"]["mode"]
    if mode == "live":
        raise LiveTradingBlocked(
            "Modo LIVE bloqueado: requiere autorización expresa del propietario y una revisión de controles. "
            "No existe implementación de órdenes con dinero real en esta versión.")
    if mode not in ("paper", "backtest"):
        raise ValueError(f"Modo desconocido: {mode}")
    kind = cfg.get("execution", {}).get("broker", "local_paper")
    rules = cfg["instrument_rules"]
    if kind == "binance_testnet":
        from quant_ai.brokers.binance_testnet import BinanceSpotTestnetBroker
        return BinanceSpotTestnetBroker(lot_steps={k: r["lot_step"] for k, r in rules.items()},
                                        min_notional={k: r["min_notional"] for k, r in rules.items()})
    from quant_ai.brokers.paper import LocalPaperBroker
    return LocalPaperBroker(cfg.state_dir, cfg["backtest"]["initial_cash"], cfg["costs"]["fee_bps"],
                            cfg["costs"]["slippage_bps"], {s: r["lot_step"] for s, r in rules.items()},
                            {s: r["min_notional"] for s, r in rules.items()})


def assert_not_live() -> None:
    if os.environ.get("QUANT_AI_MODE", "paper") == "live":
        raise LiveTradingBlocked("QUANT_AI_MODE=live está bloqueado en esta versión.")

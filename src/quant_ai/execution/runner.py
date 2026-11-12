"""Ciclo de paper trading: datos → características → predicción → decisión → riesgo → broker → registro.

Diseñado para ejecutarse una vez por hora, justo después del cierre de cada vela (o con --once).
Todo queda registrado en ``state/`` para auditoría y para el dashboard.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from quant_ai.brokers.base import Broker, BrokerError
from quant_ai.brokers.factory import get_broker
from quant_ai.brokers.paper import LocalPaperBroker
from quant_ai.config import PROJECT_ROOT, Config
from quant_ai.data.store import load_all, update_all
from quant_ai.decision_engine.engine import decide
from quant_ai.decision_engine.policy import PolicyParams, stop_guard_start, stop_guard_step
from quant_ai.features.engineering import build_feature_set
from quant_ai.risk_management.engine import (MarketState, OrderProposal, PortfolioState, RiskEngine,
                                             RiskLimits)

FEATURE_TAIL = 2000  # velas necesarias para características (la ventana más larga es 720 h)


def load_model(cfg: Config) -> tuple[dict, str | None, str]:
    """Carga el modelo que usa paper trading, con versión FIJADA en ``state/paper/MODEL``.

    - Si ``state/paper/MODEL`` existe, se usa esa versión (solo una persona la cambia).
    - Si no existe: se usa ACTIVE (si pasó la puerta) o, en su defecto, el candidato más reciente,
      y se FIJA escribiéndolo en ``state/paper/MODEL``. Así un reentrenamiento posterior nunca
      reemplaza en silencio el modelo con el que se está operando.
    """
    reg = cfg.models_dir / "registry"
    if not reg.exists():
        return {}, None, "none"
    pin = cfg.state_dir / "paper" / "MODEL"
    active = (reg / "ACTIVE").read_text(encoding="utf-8").strip() if (reg / "ACTIVE").exists() else None
    if pin.exists():
        version = pin.read_text(encoding="utf-8").strip()
    elif active:
        version = active
    else:
        versions = sorted(p.name for p in reg.iterdir() if p.is_dir())
        if not versions:
            return {}, None, "none"
        version = versions[-1]
    if not (reg / version).is_dir():
        raise FileNotFoundError(f"La versión fijada {version} no existe en {reg}")
    if not pin.exists():
        pin.parent.mkdir(parents=True, exist_ok=True)
        pin.write_text(version, encoding="utf-8")
    status = "active" if version == active else "candidate"
    models = {p.stem: joblib.load(p) for p in (reg / version).glob("*.joblib")}
    return models, version, status


def _append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def _write_heartbeat(cfg: Config, status: str, message: str, last_bar: dict, t0: float) -> None:
    hb = {"last_cycle_at": datetime.now(timezone.utc).isoformat(), "status": status, "message": message,
          "last_bar": last_bar, "cycle_seconds": round(time.perf_counter() - t0, 2)}
    (cfg.state_dir / "heartbeat.json").write_text(json.dumps(hb, indent=2, ensure_ascii=False), encoding="utf-8")


class Book:
    """Libro de cartera del runner. Con el broker local usa su archivo; con otros brokers reconcilia saldos."""

    def __init__(self, broker: Broker, cfg: Config):
        self.broker, self.cfg = broker, cfg
        self.path = cfg.state_dir / "paper" / "portfolio.json"

    def load(self) -> dict:
        if isinstance(self.broker, LocalPaperBroker):
            return self.broker.load()
        pf = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {
            "mode": "paper", "positions": {}, "initial_cash": None, "peak_equity": 0.0, "day_start_equity": 0.0,
            "day": None}
        bal = self.broker.get_balances()  # reconciliación con el broker (fuente de verdad)
        pf["cash"] = bal["cash"]
        pos = {}
        for s in self.cfg.symbols:
            q = bal["positions"].get(s, 0.0)
            if q > 0:
                pos[s] = {**pf["positions"].get(s, {"avg_price": None, "stop_price": None, "opened_at": None}), "qty": q}
        pf["positions"] = pos
        return pf

    def save(self, pf: dict) -> None:
        if isinstance(self.broker, LocalPaperBroker):
            self.broker.save(pf)
        else:
            pf["updated_at"] = datetime.now(timezone.utc).isoformat()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(pf, indent=2, ensure_ascii=False), encoding="utf-8")

    def put(self, key: str, value) -> None:
        """Guarda solo ``key`` en el archivo de cartera sin consultar al broker (no depende de la API)."""
        if isinstance(self.broker, LocalPaperBroker):
            pf = self.broker.load()
        elif self.path.exists():
            pf = json.loads(self.path.read_text(encoding="utf-8"))
        else:  # sin libro previo no hay stop que proteger; lo escribirá el guardado del final del ciclo
            return
        pf[key] = value
        self.save(pf)


def run_cycle(cfg: Config, update_data: bool = True, broker: Broker | None = None, now: pd.Timestamp | None = None,
              log=print) -> dict:
    t0 = time.perf_counter()
    paper_dir = cfg.state_dir / "paper"
    broker = broker or get_broker(cfg)
    book = Book(broker, cfg)
    risk_engine = None
    last_bar: dict = {}
    try:
        if update_data:
            update_all(cfg, use_rest=True, log=lambda *_: None)
        raw = {s: df.tail(FEATURE_TAIL).reset_index(drop=True) for s, df in load_all(cfg).items()}
        feats = build_feature_set(raw, cfg["features"])
        models, version, status = load_model(cfg)
        if not models:
            _write_heartbeat(cfg, "error", "No hay modelo registrado. Ejecuta scripts/run_experiments.py --register",
                             {}, t0)
            return {"status": "error", "message": "sin modelo"}
        stats = next(iter(models.values())).feature_stats
        root = cfg.home if (cfg.home / cfg["risk"]["kill_switch_file"]).exists() else PROJECT_ROOT
        risk_engine = RiskEngine(RiskLimits.from_config(cfg["risk"]), feature_stats=stats,
                                 state_dir=cfg.state_dir, root=root)
        policy = PolicyParams.from_config(cfg["strategy"])
        # Enfriamiento tras stop en TIEMPO DE VELA (no en ciclos): un apagado del proceso también lo consume.
        cooldown = pd.Timedelta(cfg.interval) * int(cfg["strategy"].get("stop_cooldown_bars", 24))
        now = now or pd.Timestamp.now(tz="UTC")

        pf = book.load()
        pf["model_version"], pf["model_status"] = version, status
        # Estado de re-entrada tras stop por símbolo. Vive en este dict y se vuelve a poner en ``pf`` antes de
        # cada guardado, porque ``book.load()`` tras cada orden descarta los cambios en memoria.
        guard: dict = dict(pf.get("stop_guard") or {})
        quotes = {s: broker.get_quote(s) for s in cfg.symbols}
        risk_engine.record_api_success()

        def equity_of(p):
            return p["cash"] + sum(pos["qty"] * quotes[s].bid for s, pos in p["positions"].items() if s in quotes)

        equity = equity_of(pf)
        today = now.date().isoformat()
        if pf.get("day") != today:
            pf["day"], pf["day_start_equity"] = today, equity
        pf["peak_equity"] = max(pf.get("peak_equity") or equity, equity)
        decisions = []

        for s in cfg.symbols:
            f = feats[s]
            bar_time = f.index[-1]
            last_bar[s] = str(bar_time)
            close_time = raw[s]["close_time"].iloc[-1]
            pos = pf["positions"].get(s, {"qty": 0.0})
            price = quotes[s].mid
            cur_w = pos["qty"] * quotes[s].bid / equity if equity > 0 else 0.0
            model = models.get(s)
            if model is None:
                continue
            X_row = f.iloc[[-1]]
            d = decide(s, bar_time, X_row, model, policy, cur_w, pos["qty"], equity, price,
                       cfg["labeling"]["horizon_bars"], cfg["strategy"]["atr_stop_mult"], version)

            # Stop-loss protector: si el bid actual perfora el stop, se cierra
            stop_px = pos.get("stop_price")
            stop_hit = bool(pos["qty"] > 0 and stop_px and quotes[s].bid <= stop_px)
            if stop_hit:
                d.action, d.target_weight, d.size_qty = "CLOSE", 0.0, pos["qty"]
                d.notional = pos["qty"] * quotes[s].bid
                d.reasons.insert(0, f"stop-loss: bid {quotes[s].bid:.2f} <= stop {stop_px:.2f}")

            # Re-entrada tras stop (mismas reglas que run_backtest): enfriamiento y señal nueva
            prob = float(d.probabilities.get("p_ensemble", np.nan))
            g, blocks = stop_guard_step(guard.pop(s, None), bar_time, prob, policy)
            if g:
                guard[s] = g
            if blocks and d.action == "BUY":
                d.action, d.direction, d.target_weight = "HOLD", "flat", 0.0
                d.size_qty, d.notional, d.stop_loss = 0.0, 0.0, None
                d.reasons += blocks

            side = "BUY" if d.action == "BUY" else ("SELL" if d.action in ("SELL", "CLOSE") else None)
            if side and d.size_qty > 0:
                oid = f"qa-{s}-{bar_time:%Y%m%d%H}-{side}"
                proposal = OrderProposal(s, side, d.notional, d.target_weight, oid, price)
                mstate = MarketState(close_time, float(f["vol_24"].iloc[-1]),
                                     {k: float(v) for k, v in X_row.iloc[0].items() if pd.notna(v)},
                                     quotes[s].spread_bps)
                gross = sum(p["qty"] * quotes[k].bid for k, p in pf["positions"].items()) / equity if equity else 0
                pstate = PortfolioState(equity, pf["peak_equity"], pf["day_start_equity"],
                                        {k: p["qty"] * quotes[k].bid / equity for k, p in pf["positions"].items()},
                                        gross)
                # OOD por activo: cada símbolo se compara con SU distribución de entrenamiento
                risk_engine.feature_stats = model.feature_stats
                rd = risk_engine.evaluate(proposal, mstate, pstate, now)
                d.risk = rd.to_dict()
                if rd.approved:
                    try:
                        if isinstance(broker, LocalPaperBroker):
                            res = broker.place_market_order(s, side, d.size_qty, oid, stop_price=d.stop_loss)
                        else:
                            res = broker.place_market_order(s, side, d.size_qty, oid)
                        risk_engine.record_api_success()
                    except BrokerError as e:
                        risk_engine.record_api_error()
                        d.reasons.append(f"error del broker: {e}")
                        res = None
                    if res is not None:
                        d.executed = res.status in ("filled", "partial")
                        d.fill = {"price": res.avg_price, "qty": res.qty_filled, "fee": res.fee, "status": res.status}
                        if stop_hit and d.executed:
                            # Aproximación: el stop se revisa una vez por ciclo contra el bid tomado tras
                            # el cierre de la vela decidida b. Si perforó, también lo hizo el cierre de b
                            # (en cripto 24/7 la apertura de b+1 es el cierre de b): el backtest lo habría
                            # disparado en b o antes, así que b hace de vela i del stop y se bloquean las
                            # velas decididas b..b+N-1. Si el cruce ocurre entre el cierre de b y el ciclo,
                            # el backtest lo fecharía en b+1 y aquí el enfriamiento acaba una vela antes.
                            g = stop_guard_start(bar_time, prob, policy, cooldown)
                            guard.pop(s, None)
                            if g:
                                guard[s] = g
                            # Se guarda ya y sin pasar por el broker: ni un error de la API al reconciliar
                            # ni uno posterior del ciclo deben borrar el enfriamiento de una venta hecha.
                            book.put("stop_guard", guard)
                        pf = book.load()
                        pf["model_version"], pf["model_status"] = version, status
                        if not isinstance(broker, LocalPaperBroker) and side == "BUY" and d.executed:
                            pf["positions"].setdefault(s, {})["stop_price"] = d.stop_loss
                else:
                    d.reasons.append("bloqueada por riesgo: " + "; ".join(rd.reasons))
            else:
                d.risk = {"approved": None, "reasons": ["sin orden"], "checks": {}}
            decisions.append(d)
            _append_jsonl(paper_dir / "decisions.jsonl", d.to_dict())
            log(f"  {s} {bar_time} p={d.probabilities.get('p_ensemble', float('nan')):.3f} "
                f"unc={d.uncertainty:.3f} -> {d.action} riesgo={d.risk.get('approved')} ejecutada={d.executed}")

        equity = equity_of(pf)
        pf["equity"] = equity
        pf["peak_equity"] = max(pf.get("peak_equity") or equity, equity)
        pf["stop_guard"] = guard
        book.save(pf)
        gross = sum(p["qty"] * quotes[k].bid for k, p in pf["positions"].items()) / equity if equity else 0
        eq_path = paper_dir / "equity.csv"
        new = not eq_path.exists()
        with open(eq_path, "a", encoding="utf-8") as fh:
            if new:
                fh.write("time,equity,cash,gross_exposure\n")
            fh.write(f"{now.isoformat()},{equity:.6f},{pf['cash']:.6f},{gross:.6f}\n")
        risk_engine.save_state()
        status_hb = "kill_switch" if risk_engine.kill_switch_active() else (
            "emergency" if risk_engine.emergency_reason else "ok")
        _write_heartbeat(cfg, status_hb, risk_engine.emergency_reason or f"modelo {version} ({status})", last_bar, t0)
        return {"status": status_hb, "equity": equity, "decisions": [d.to_dict() for d in decisions]}
    except BrokerError as e:
        if risk_engine:
            risk_engine.record_api_error()
            risk_engine.save_state()
        _write_heartbeat(cfg, "error", f"broker/API: {e}", last_bar, t0)
        log(f"ERROR broker/API: {e}")
        return {"status": "error", "message": str(e)}
    except Exception as e:  # noqa: BLE001 — se registra y se reporta; el siguiente ciclo reintenta
        _write_heartbeat(cfg, "error", f"{type(e).__name__}: {e}", last_bar, t0)
        log(f"ERROR: {type(e).__name__}: {e}")
        raise

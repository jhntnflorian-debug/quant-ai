"""Entorno spot de un activo: decisión al cierre y ejecución a la siguiente apertura."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from quant_ai.backtesting.engine import CostModel, Rules


class TradingEnv(gym.Env):
    """Acciones 0/1/2 -> pesos objetivo 0/0.5/1 multiplicados por ``max_weight``.

    ``start`` y ``end`` son fechas UTC, intervalo [start, end). La primera fila
    es una decisión en efectivo; cada paso consume la vela siguiente del rango.
    La normalización debe proceder de ``fit_normalization`` sobre train completo;
    nunca se reajusta al reiniciar un episodio ni en evaluación.

    Recompensa = log(E[t+1]/E[t]) - dd_penalty * max(DD[t+1]-DD[t], 0)
    - turnover_penalty * abs(w_después_fill - w_antes_fill). DD es una pérdida
    positiva desde el máximo de capital. Ambos pesos se valoran en open[t+1],
    para no penalizar movimientos de mercado como operaciones. Los coeficientes
    adimensionales por defecto son 0.1 y 0.001; cero desactiva cada penalización.
    Los costes ya están en E: no se descuentan de nuevo de la recompensa.

    ``max_weight`` limita el objetivo, no la deriva posterior del precio. Sin
    crédito ni cortos, el peso efectivo permanece en [0, 1]. Se replica el motor
    sin stops: cantidad decidida al cierre, volumen previo, lote, mínimo y efectivo.
    Al acabar se marca a mercado, sin liquidación ficticia; es una truncación
    temporal, no un estado terminal de insolvencia.
    """

    metadata = {"render_modes": []}

    @staticmethod
    def fit_normalization(features: pd.DataFrame, start, end) -> dict:
        """Ajusta SOLO [start, end) de entrenamiento; std poblacional, constantes=1."""
        start, end = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)
        train = features.loc[(features.index >= start) & (features.index < end)]
        if train.empty or not start < end:
            raise ValueError("Tramo de entrenamiento vacío para normalización.")
        train = train.replace([np.inf, -np.inf], np.nan)
        mean = train.mean().fillna(0)
        std = train.std(ddof=0).replace(0, 1).fillna(1)
        return {"feature_names": list(features.columns), "mean": mean.tolist(),
                "std": std.tolist(), "train_start": start.isoformat(), "train_end": end.isoformat()}

    def __init__(self, features: pd.DataFrame, prices: pd.DataFrame, start, end, *,
                 normalization: dict, costs: CostModel | None = None, rules: Rules | None = None,
                 max_weight: float = 0.5, initial_cash: float = 10_000,
                 rebalance_band: float = 0.05, dd_penalty: float = 0.1,
                 turnover_penalty: float = 0.001, training: bool = False,
                 episode_length: int = 720, bars_scale: float = 168):
        super().__init__()
        self.start, self.end = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)
        self.training, self.episode_length = training, episode_length
        self.max_weight, self.initial_cash = float(max_weight), float(initial_cash)
        self.rebalance_band, self.bars_scale = float(rebalance_band), float(bars_scale)
        self.dd_penalty, self.turnover_penalty = float(dd_penalty), float(turnover_penalty)
        self.costs, self.rules = costs or CostModel(), rules or Rules()
        params = [max_weight, initial_cash, rebalance_band, bars_scale, dd_penalty, turnover_penalty,
                  self.costs.fee, self.costs.impact, self.costs.max_participation,
                  self.rules.lot_step, self.rules.min_notional]
        if not np.isfinite(params).all() or not 0 < max_weight <= 1 or initial_cash <= 0:
            raise ValueError("Parámetros inválidos: capital positivo y 0 < max_weight <= 1.")
        if (min(rebalance_band, dd_penalty, turnover_penalty, self.rules.min_notional) < 0
                or bars_scale <= 0 or self.rules.lot_step <= 0
                or not 0 <= self.costs.fee < 1 or not 0 <= self.costs.impact < 1
                or not 0 <= self.costs.max_participation <= 1):
            raise ValueError("Costes, reglas o penalizaciones inválidos.")
        if not isinstance(features.index, pd.DatetimeIndex) or not features.index.is_unique:
            raise ValueError("features requiere un DatetimeIndex open_time único.")
        f = features.copy()
        f.index = pd.to_datetime(f.index, utc=True)
        if not f.index.is_monotonic_increasing or not self.start < self.end:
            raise ValueError("Fechas desordenadas o rango inválido.")
        f = f.loc[(f.index >= self.start) & (f.index < self.end)]
        if len(f) < 2:
            raise ValueError("El rango requiere al menos dos velas de 1h.")
        steps_ok = (f.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).to_numpy()
        if not training and not steps_ok.all():
            raise ValueError("El rango de evaluación debe ser continuo (sin huecos de 1h).")
        # Los datos de Binance tienen huecos reales (27 desde 2018; el último el 2023-03-24).
        # En entrenamiento se permiten, pero ningún episodio cruza un hueco (ver valid_starts).
        self.segment = np.r_[0, np.cumsum(~steps_ok)]
        p = prices.copy()
        if "open_time" in p:
            p = p.set_index("open_time")
        p.index = pd.to_datetime(p.index, utc=True)
        if not p.index.is_unique:
            raise ValueError("Precios con open_time duplicado.")
        p = p.reindex(f.index)
        values = p[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
            raise ValueError("OHLCV faltantes o inválidos en el rango solicitado.")
        if list(f.columns) != normalization["feature_names"] or not len(f.columns):
            raise ValueError("Las columnas no coinciden con la normalización de entrenamiento.")
        fit_start = pd.to_datetime(normalization["train_start"], utc=True)
        fit_end = pd.to_datetime(normalization["train_end"], utc=True)
        if not fit_start < fit_end or (training and (fit_start != self.start or fit_end != self.end)):
            raise ValueError("La normalización debe ajustarse en el mismo rango de entrenamiento.")
        if not training and fit_end > self.start:
            raise ValueError("La normalización incluye datos posteriores al inicio de evaluación.")
        mean, std = np.asarray(normalization["mean"]), np.asarray(normalization["std"])
        if (mean.shape != (len(f.columns),) or std.shape != mean.shape
                or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any()):
            raise ValueError("Media/desviación inválidas en la normalización.")
        z = (f.to_numpy(dtype=float) - mean) / std
        self.features = np.clip(np.nan_to_num(z, nan=0, posinf=10, neginf=-10), -10, 10).astype(np.float32)
        self.index = f.index
        self.open_prices, self.close_prices = values[:, 0], values[:, 3]
        self.volumes = values[:, 4]
        if training and (not isinstance(episode_length, int) or not 1 <= episode_length < len(f)):
            raise ValueError("episode_length debe caber completo dentro de entrenamiento.")
        if training:
            starts = np.arange(len(f) - episode_length)
            # inicio válido: el episodio completo [i, i+episode_length] cae en un solo tramo continuo
            self.valid_starts = starts[self.segment[starts] == self.segment[starts + episode_length]]
            if not len(self.valid_starts):
                raise ValueError("Ningún tramo continuo del rango alcanza episode_length velas.")
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-10, 10, shape=(len(f.columns) + 4,), dtype=np.float32)
        self.done = True

    @property
    def equity(self) -> float:
        return float(self.cash + self.qty * self.close_prices[self.i])

    @property
    def weight(self) -> float:
        return float(self.qty * self.close_prices[self.i] / self.equity)

    def _observation(self):
        unrealized = ((self.qty * self.close_prices[self.i] - self.position_cost) / self.position_cost
                      if self.position_cost > 0 else 0.0)
        account = [self.weight, np.clip(unrealized, -10, 10), self.drawdown,
                   min(self.bars_since_trade / self.bars_scale, 1)]
        return np.concatenate([self.features[self.i], account]).astype(np.float32)

    def _info(self, step_cost=0.0, turnover=0.0):
        return {"time": self.index[self.i], "equity": self.equity, "weight": self.weight,
                "cash": self.cash, "qty": self.qty, "drawdown": self.drawdown,
                "costs_paid": self.costs_paid, "step_cost": step_cost,
                "operations": len(self.fills), "turnover": turnover}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.i = (int(self.valid_starts[self.np_random.integers(len(self.valid_starts))])
                  if self.training else 0)
        self.last_i = self.i + self.episode_length if self.training else len(self.index) - 1
        self.cash, self.qty, self.position_cost = self.initial_cash, 0.0, 0.0
        self.peak, self.drawdown, self.costs_paid = self.initial_cash, 0.0, 0.0
        self.bars_since_trade = 0
        self.fills, self.trades = [], []
        self.open_trade = None
        self.done = False
        return self._observation(), self._info()

    def step(self, action):
        if self.done:
            raise RuntimeError("Llame reset antes de actuar o después del fin del episodio.")
        if not self.action_space.contains(action):
            raise ValueError("Acción fuera de Discrete(3).")
        equity_before, dd_before = self.equity, self.drawdown
        target = int(action) * 0.5 * self.max_weight
        order = 0.0
        if target == 0 and self.qty > 0:
            order = -self.qty
        elif target > 0 and (self.qty == 0 or abs(target - self.weight) > self.rebalance_band):
            order = target * equity_before / self.close_prices[self.i] - self.qty
        # La cantidad y el volumen se fijan antes de acceder a la apertura siguiente.
        prev_volume = self.volumes[self.i]
        self.i += 1
        self.bars_since_trade += 1
        step_cost, turnover = self._fill(order, prev_volume) if order != 0 else (0.0, 0.0)
        self.peak = max(self.peak, self.equity)
        self.drawdown = max(0.0, 1 - self.equity / self.peak)
        reward = (np.log(self.equity / equity_before)
                  - self.dd_penalty * max(0.0, self.drawdown - dd_before)
                  - self.turnover_penalty * turnover)
        self.done = self.i >= self.last_i
        return self._observation(), float(reward), False, self.done, self._info(step_cost, turnover)

    def _fill(self, order, prev_volume):
        ref = self.open_prices[self.i]
        buying = order > 0
        px = ref * (1 + self.costs.impact if buying else 1 - self.costs.impact)
        q = min(abs(order), self.costs.max_participation * prev_volume)
        q = min(q, self.cash / (px * (1 + self.costs.fee))) if buying else min(q, self.qty)
        q = self.rules.round_qty(q) if buying or q < self.qty - 1e-12 else q
        notional = q * px
        if q <= 0 or (notional < self.rules.min_notional and not (not buying and q >= self.qty - 1e-12)):
            return 0.0, 0.0
        fee, impact = notional * self.costs.fee, abs(px - ref) * q
        weight_before = self.qty * ref / (self.cash + self.qty * ref)
        if buying:
            self.cash = max(0.0, self.cash - notional - fee)  # Solo residuo de redondeo flotante.
            self.qty += q
            self.position_cost += notional + fee
            if self.open_trade is None:
                self.open_trade = {"entry_time": self.index[self.i], "entry_value": 0.0,
                                   "exit_value": 0.0, "fees": 0.0}
            self.open_trade["entry_value"] += notional
        else:
            self.position_cost *= max(0.0, 1 - q / self.qty)
            self.cash += notional - fee
            self.qty -= q
            self.open_trade["exit_value"] += notional
            if self.qty < 1e-12:
                self.qty, self.position_cost = 0.0, 0.0
        self.open_trade["fees"] += fee
        if self.qty == 0:
            trade = self.open_trade
            trade["exit_time"] = self.index[self.i]
            trade["pnl"] = trade["exit_value"] - trade["entry_value"] - trade["fees"]
            trade["return"] = trade["pnl"] / trade["entry_value"]
            trade["bars_held"] = int((trade["exit_time"] - trade["entry_time"]) / pd.Timedelta(hours=1))
            self.trades.append(trade)
            self.open_trade = None
        self.fills.append({"time": self.index[self.i], "side": "BUY" if buying else "SELL",
                           "qty": q, "ref_price": ref, "fill_price": px, "notional": notional,
                           "fee": fee, "impact_cost": impact, "requested_qty": abs(order)})
        self.costs_paid += fee + impact
        self.bars_since_trade = 0
        weight_after = self.qty * ref / (self.cash + self.qty * ref)
        return fee + impact, abs(weight_after - weight_before)

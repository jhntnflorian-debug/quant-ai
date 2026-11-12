import numpy as np
import pandas as pd
import pytest


def make_ohlcv(n: int = 600, seed: int = 0, start: str = "2023-01-01", price: float = 100.0) -> pd.DataFrame:
    """Serie OHLCV sintética con la misma estructura que las velas de Binance (solo para pruebas)."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.01, n)
    close = price * np.exp(np.cumsum(r))
    open_ = np.r_[price, close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.005, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.005, n))
    vol = rng.uniform(100, 200, n)
    t = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open_time": t, "open": open_, "high": high, "low": low, "close": close, "volume": vol,
        "close_time": t + pd.Timedelta("1h") - pd.Timedelta("1ms"), "quote_volume": vol * close,
        "trades": rng.integers(100, 1000, n), "taker_buy_base": vol * rng.uniform(0.3, 0.7, n),
        "taker_buy_quote": vol * close * 0.5,
    })


@pytest.fixture
def ohlcv():
    return make_ohlcv()

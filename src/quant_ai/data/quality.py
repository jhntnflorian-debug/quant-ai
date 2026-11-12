"""Controles de calidad de velas OHLCV."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd


@dataclass
class QualityReport:
    symbol: str
    rows: int
    start: str | None
    end: str | None
    duplicates_removed: int
    missing_bars: int
    gap_count: int
    largest_gap_bars: int
    ohlc_violations: int
    nonpositive_prices: int
    off_grid_bars: int
    zero_volume_bars: int
    tz: str
    gaps: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def clean_klines(df: pd.DataFrame, symbol: str, freq: str = "1h") -> tuple[pd.DataFrame, QualityReport]:
    """Ordena, deduplica (conserva la última versión de cada open_time), valida y reporta huecos.

    NO rellena huecos: rellenar precios inventaría información. Los huecos se reportan y las
    características se calculan sobre el índice real (las ventanas cruzan el hueco y se documenta).
    """
    if df.empty:
        return df, QualityReport(symbol, 0, None, None, 0, 0, 0, 0, 0, 0, 0, 0, "UTC")
    if df["open_time"].dt.tz is None:
        raise ValueError("open_time debe tener zona horaria UTC")
    n0 = len(df)
    step = pd.Timedelta(freq)
    # Velas fuera de la grilla (p. ej. reanudación tras mantenimiento de Binance 2018-02-09 09:28:14):
    # se alinean al inicio de su intervalo. Si colisionan con otra vela, se conserva la última.
    off_grid = df["open_time"] != df["open_time"].dt.floor(step)
    df = df.assign(open_time=df["open_time"].dt.floor(step))
    out = df.sort_values("open_time").drop_duplicates("open_time", keep="last").reset_index(drop=True)
    dups = n0 - len(out)

    diffs = out["open_time"].diff()
    gap_mask = diffs > step
    gaps = []
    for idx in out.index[gap_mask]:
        prev, cur = out.at[idx - 1, "open_time"], out.at[idx, "open_time"]
        gaps.append({"from": str(prev), "to": str(cur), "missing_bars": int((cur - prev) / step) - 1})
    missing = sum(g["missing_bars"] for g in gaps)

    ohlc_bad = (
        (out["high"] < out[["open", "close", "low"]].max(axis=1)) |
        (out["low"] > out[["open", "close", "high"]].min(axis=1))
    )
    nonpos = (out[["open", "high", "low", "close"]] <= 0).any(axis=1)
    # Filas con precios imposibles se eliminan; las demás violaciones se reportan
    out = out[~nonpos].reset_index(drop=True)

    rep = QualityReport(
        symbol=symbol,
        rows=len(out),
        start=str(out["open_time"].iloc[0]),
        end=str(out["open_time"].iloc[-1]),
        duplicates_removed=int(dups),
        missing_bars=int(missing),
        gap_count=len(gaps),
        largest_gap_bars=max((g["missing_bars"] for g in gaps), default=0),
        ohlc_violations=int(ohlc_bad.sum()),
        nonpositive_prices=int(nonpos.sum()),
        off_grid_bars=int(off_grid.sum()),
        zero_volume_bars=int((out["volume"] == 0).sum()),
        tz=str(out["open_time"].dt.tz),
        gaps=gaps[:50],
    )
    return out, rep

"""Adquisición de velas spot de Binance.

Dos fuentes complementarias:
1. Archivos mensuales/diarios de https://data.binance.vision (históricos completos, con CHECKSUM SHA256).
2. API REST pública ``/api/v3/klines`` para completar incrementalmente hasta la última vela CERRADA.

Particularidades verificadas el 2026-11-08:
- Los CSV de data.binance.vision no traen encabezado.
- Desde 2025-01 los timestamps de spot vienen en MICROsegundos (antes en milisegundos).
  Normalizamos por magnitud, no por fecha.
- La API REST devuelve la vela en formación como último elemento: se descarta si close_time > ahora.
"""
from __future__ import annotations

import hashlib
import io
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

VISION_BASE = "https://data.binance.vision/data/spot"
REST_BASE = "https://api.binance.com"
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]
FLOAT_COLS = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]
INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
USER_AGENT = "quant-ai-research/0.1"


def _to_datetime_utc(values: pd.Series) -> pd.Series:
    """Convierte epoch en ms o µs (detectado por magnitud) a datetime UTC."""
    v = pd.to_numeric(values, errors="raise").astype("int64")
    # 1e14 µs ≈ 1973; 1e14 ms ≈ año 5138. Cualquier valor > 1e14 es microsegundos.
    us_mask = v > 100_000_000_000_000
    ms = v.where(~us_mask, v // 1000)
    return pd.to_datetime(ms, unit="ms", utc=True)


def parse_klines(rows: list[list] | pd.DataFrame) -> pd.DataFrame:
    """Normaliza filas de klines (REST o CSV) a un DataFrame tipado con timestamps UTC."""
    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows, columns=KLINE_COLUMNS)
    df.columns = KLINE_COLUMNS
    df["open_time"] = _to_datetime_utc(df["open_time"])
    df["close_time"] = _to_datetime_utc(df["close_time"])
    for c in FLOAT_COLS:
        df[c] = pd.to_numeric(df[c], errors="raise").astype("float64")
    df["trades"] = pd.to_numeric(df["trades"], errors="raise").astype("int64")
    return df.drop(columns=["ignore"])


# ----------------------------------------------------------------------------- vision
def _download(url: str, session: requests.Session, timeout: int = 60) -> bytes | None:
    r = session.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.content


def fetch_vision_file(symbol: str, interval: str, period: str, freq: str, raw_dir: Path,
                      session: requests.Session | None = None) -> tuple[Path | None, dict]:
    """Descarga (si no existe) un zip mensual (freq='monthly', period='2024-05') o diario
    (freq='daily', period='2026-11-03') y verifica su SHA256. Los zips son la copia original inmutable."""
    session = session or requests.Session()
    name = f"{symbol}-{interval}-{period}.zip"
    url = f"{VISION_BASE}/{freq}/klines/{symbol}/{interval}/{name}"
    dest = raw_dir / symbol / interval / freq / name
    meta = {"url": url, "file": str(dest), "symbol": symbol, "period": period, "freq": freq}
    if dest.exists():
        meta["sha256"] = hashlib.sha256(dest.read_bytes()).hexdigest()
        meta["cached"] = True
        return dest, meta
    content = _download(url, session)
    if content is None:
        return None, {**meta, "status": "not_found"}
    checksum_txt = _download(url + ".CHECKSUM", session)
    digest = hashlib.sha256(content).hexdigest()
    if checksum_txt:
        expected = checksum_txt.decode().split()[0].strip()
        if expected != digest:
            raise ValueError(f"Checksum inválido para {url}: esperado {expected}, obtenido {digest}")
        meta["checksum_verified"] = True
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    meta.update({"sha256": digest, "bytes": len(content), "cached": False,
                 "downloaded_at": datetime.now(timezone.utc).isoformat()})
    return dest, meta


def read_vision_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        raw = z.read(name).decode()
    first = raw.split("\n", 1)[0]
    header = 0 if first.startswith("open_time") else None  # algunos productos traen encabezado
    df = pd.read_csv(io.StringIO(raw), header=header, names=KLINE_COLUMNS if header is None else None)
    return parse_klines(df)


def month_range(start: str, end: pd.Timestamp) -> list[str]:
    end = end.tz_localize(None) if end.tzinfo is not None else end
    months = pd.period_range(pd.Timestamp(start).to_period("M"), end.to_period("M"), freq="M")
    return [str(p) for p in months]


# ----------------------------------------------------------------------------- REST
def server_time_ms(session: requests.Session | None = None) -> int:
    session = session or requests.Session()
    r = session.get(f"{REST_BASE}/api/v3/time", timeout=10, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    return int(r.json()["serverTime"])


def fetch_rest_klines(symbol: str, interval: str, start_ms: int, end_ms: int | None = None,
                      session: requests.Session | None = None, limit: int = 1000,
                      pause_s: float = 0.25) -> pd.DataFrame:
    """Pagina /api/v3/klines desde start_ms. Devuelve SOLO velas cerradas (close_time < serverTime)."""
    session = session or requests.Session()
    now_ms = server_time_ms(session)
    end_ms = min(end_ms or now_ms, now_ms)
    rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        params = {"symbol": symbol, "interval": interval, "startTime": cursor, "endTime": end_ms, "limit": limit}
        r = session.get(f"{REST_BASE}/api/v3/klines", params=params, timeout=20,
                        headers={"User-Agent": USER_AGENT})
        if r.status_code in (418, 429):  # rate limit: respetar Retry-After
            time.sleep(int(r.headers.get("Retry-After", "5")))
            continue
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        last_open = int(batch[-1][0])
        cursor = last_open + INTERVAL_MS[interval]
        if len(batch) < limit:
            break
        time.sleep(pause_s)
    if not rows:
        return parse_klines([])
    df = parse_klines(rows)
    return drop_unclosed(df, now_ms)


def drop_unclosed(df: pd.DataFrame, now_ms: int) -> pd.DataFrame:
    """Elimina velas cuyo cierre aún no ha ocurrido (vela en formación)."""
    now = pd.Timestamp(now_ms, unit="ms", tz="UTC")
    return df[df["close_time"] < now].reset_index(drop=True)


def fetch_exchange_info(symbols: list[str], session: requests.Session | None = None) -> dict:
    """Reglas del instrumento (LOT_SIZE, MIN_NOTIONAL/NOTIONAL, PRICE_FILTER)."""
    session = session or requests.Session()
    r = session.get(f"{REST_BASE}/api/v3/exchangeInfo", params={"symbols": str(symbols).replace("'", '"').replace(" ", "")},
                    timeout=15, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    out = {}
    for s in r.json()["symbols"]:
        f = {flt["filterType"]: flt for flt in s["filters"]}
        notional = f.get("NOTIONAL") or f.get("MIN_NOTIONAL") or {}
        out[s["symbol"]] = {
            "lot_step": float(f["LOT_SIZE"]["stepSize"]),
            "min_qty": float(f["LOT_SIZE"]["minQty"]),
            "tick_size": float(f["PRICE_FILTER"]["tickSize"]),
            "min_notional": float(notional.get("minNotional", 5.0)),
            "status": s["status"],
        }
    return out

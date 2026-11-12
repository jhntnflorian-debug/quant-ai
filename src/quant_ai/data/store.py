"""Almacenamiento versionado de datos de mercado.

Estructura en ``data/``:
- ``raw/binance_vision/<SYM>/<interval>/{monthly,daily}/*.zip``  copia original inmutable (con SHA256)
- ``raw/binance_rest/<SYM>_<interval>_<timestamp>.parquet``      lotes REST tal como llegaron
- ``clean/<SYM>_<interval>.parquet``                             serie limpia y deduplicada
- ``manifest.json``                                              procedencia: URLs, hashes, filas, calidad
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from quant_ai.config import Config
from quant_ai.data import binance
from quant_ai.data.quality import clean_klines


def clean_path(cfg: Config, symbol: str) -> Path:
    return cfg.data_dir / "clean" / f"{symbol}_{cfg.interval}.parquet"


def load_clean(cfg: Config, symbol: str) -> pd.DataFrame:
    p = clean_path(cfg, symbol)
    if not p.exists():
        raise FileNotFoundError(f"No existe {p}. Ejecuta: python scripts/download_data.py")
    return pd.read_parquet(p)


def load_all(cfg: Config) -> dict[str, pd.DataFrame]:
    return {s: load_clean(cfg, s) for s in cfg.symbols}


def _file_sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def update_symbol(cfg: Config, symbol: str, session: requests.Session | None = None,
                  use_rest: bool = True, log=print, rest_best_effort: bool = False) -> dict:
    """Actualiza incrementalmente un símbolo: zips mensuales faltantes + REST hasta la última vela cerrada.

    ``rest_best_effort``: si la API REST falla (p. ej. HTTP 451 cuando la máquina está en una región que
    Binance restringe, como ocurre con algunas VM de Colab), se avisa y se sigue con los archivos mensuales y
    lo ya guardado. Solo para entrenamiento: el ciclo de paper lo deja en False, porque operar exige datos al día.
    """
    session = session or requests.Session()
    interval = cfg.interval
    raw_dir = cfg.data_dir / "raw" / "binance_vision"
    now = pd.Timestamp.now(tz="UTC")
    existing = None
    cp = clean_path(cfg, symbol)
    if cp.exists():
        existing = pd.read_parquet(cp)

    files_meta: list[dict] = []
    frames: list[pd.DataFrame] = []
    if existing is not None:
        frames.append(existing)
    # Meses completos (excluye el mes en curso, que aún no está publicado como mensual)
    last_full_month = (now.tz_localize(None).to_period("M") - 1).to_timestamp().tz_localize("UTC")
    for period in binance.month_range(cfg["market"]["history_start"], last_full_month):
        month_start = pd.Timestamp(period + "-01", tz="UTC")
        month_end = month_start + pd.offsets.MonthBegin(1)
        if existing is not None and len(existing):
            # Los archivos mensuales son inmutables: si el mes ya tiene datos, no se relee
            # (los huecos genuinos del exchange quedan documentados en el reporte de calidad).
            have = existing[(existing["open_time"] >= month_start) & (existing["open_time"] < month_end)]
            if len(have) > 0:
                continue
        path, meta = binance.fetch_vision_file(symbol, interval, period, "monthly", raw_dir, session)
        files_meta.append(meta)
        if path is None:
            log(f"  {symbol} {period}: no publicado")
            continue
        frames.append(binance.read_vision_zip(path))

    df = pd.concat(frames, ignore_index=True) if frames else binance.parse_klines([])
    rest_rows, rest_error = 0, None
    if use_rest:
        start = df["open_time"].max() + pd.Timedelta(interval) if len(df) else pd.Timestamp(
            cfg["market"]["history_start"], tz="UTC")
        try:
            rest = binance.fetch_rest_klines(symbol, interval, int(start.timestamp() * 1000), session=session)
        except requests.RequestException as e:
            if not rest_best_effort or not len(df):
                raise
            rest, rest_error = binance.parse_klines([]), f"{type(e).__name__}: {e}"
            log(f"  AVISO {symbol}: la API REST no respondió ({rest_error}). Se continúa con los datos "
                f"guardados, que llegan hasta {df['open_time'].max()}.")
        if len(rest):
            rest_dir = cfg.data_dir / "raw" / "binance_rest"
            rest_dir.mkdir(parents=True, exist_ok=True)
            stamp = now.strftime("%Y%m%dT%H%M%SZ")
            rp = rest_dir / f"{symbol}_{interval}_{stamp}.parquet"
            rest.to_parquet(rp, index=False)
            files_meta.append({"source": "rest", "file": str(rp), "rows": len(rest), "sha256": _file_sha256(rp),
                               "from": str(rest["open_time"].min()), "to": str(rest["open_time"].max())})
            df = pd.concat([df, rest], ignore_index=True)
            rest_rows = len(rest)

    clean, rep = clean_klines(df, symbol, interval)
    clean = clean[clean["open_time"] >= pd.Timestamp(cfg["market"]["history_start"], tz="UTC")]
    cp.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(cp, index=False)
    result = {
        "symbol": symbol, "interval": interval, "clean_file": str(cp), "clean_sha256": _file_sha256(cp),
        "rows": len(clean), "rest_rows_added": rest_rows, "rest_error": rest_error, "new_files": [m for m in files_meta if not m.get("cached")],
        "quality": rep.to_dict(), "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    log(f"  {symbol}: {len(clean)} velas [{rep.start} → {rep.end}], huecos={rep.gap_count} "
        f"({rep.missing_bars} velas), duplicados={rep.duplicates_removed}, REST+{rest_rows}")
    return result


def update_all(cfg: Config, use_rest: bool = True, log=print, rest_best_effort: bool = False) -> dict:
    session = requests.Session()
    manifest_path = cfg.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"history": []}
    results = {}
    for s in cfg.symbols:
        results[s] = update_symbol(cfg, s, session=session, use_rest=use_rest, log=log,
                                   rest_best_effort=rest_best_effort)
    manifest["latest"] = results
    manifest["source"] = {
        "provider": "Binance spot",
        "bulk": binance.VISION_BASE, "rest": binance.REST_BASE + "/api/v3/klines",
        "license_note": "Datos públicos de Binance; ver términos de uso de Binance. No redistribuir.",
    }
    manifest["history"].append({s: {"rows": r["rows"], "sha256": r["clean_sha256"], "at": r["updated_at"]}
                                for s, r in results.items()})
    manifest["history"] = manifest["history"][-200:]
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return results

"""Descarga datos de otras plataformas (derivados, Coinbase, Deribit, sentimiento, macro, alts, on-chain).

Pensado para Colab (red rápida): python scripts/download_extended.py --workers 8 --cache-dir /content/qa_cache
En el PC local usar solo pruebas mínimas (p. ej. --sources fng,stable).

- Datos consolidados (Parquet + manifiesto) en <data_dir>/external/ (en Colab, Google Drive vía QUANT_AI_HOME).
- Originales en --cache-dir (en Colab conviene disco local: miles de archivos pequeños en Drive son lentos).
- Si una fuente falla (p. ej. bloqueo regional HTTP 451/403), se avisa y se continúa con las demás.
"""
import argparse
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quant_ai.config import load_config  # noqa: E402
from quant_ai.data.external import ALTS, Downloader  # noqa: E402

ALL = ["perp", "coinbase", "deribit", "fng", "stable", "fred", "alts", "onchain"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", default="all")
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    p.add_argument("--workers", type=int, default=8 if "google.colab" in sys.modules else 2)
    p.add_argument("--since", default="2018-01-01")
    p.add_argument("--data-dir", default=None, help="por defecto el data_dir de la configuración (QUANT_AI_HOME)")
    p.add_argument("--cache-dir", default=os.environ.get("QUANT_AI_CACHE"),
                   help="carpeta para los originales descargados (por defecto <data_dir>/external)")
    p.add_argument("--config", default=None)
    args = p.parse_args()
    sources = ALL if args.sources == "all" else args.sources.split(",")
    if set(sources) - set(ALL):
        p.error(f"Fuente desconocida; opciones: {ALL}")
    if args.workers < 1:
        p.error("--workers debe ser positivo")
    data_dir = Path(args.data_dir) if args.data_dir else load_config(args.config).data_dir
    print(f"Datos en {data_dir / 'external'} | caché {args.cache_dir or '(misma carpeta)'} | workers {args.workers}")
    d = Downloader(data_dir, max_workers=args.workers, cache_dir=args.cache_dir)
    failed = []
    for source in sources:
        symbols = (ALTS if source == "alts" else ["VIXCLS", "SP500", "DGS10", "DTWEXBGS"] if source == "fred"
                   else [None] if source in ("fng", "stable", "onchain") else args.symbols.split(","))
        for symbol in symbols:
            for kind in (["perp", "premium", "funding", "metrics"] if source == "perp" else [source]):
                print(f"Descargando {kind} {symbol or ''}", flush=True)
                try:
                    d.download(kind, symbol, args.since)
                except Exception as e:  # noqa: BLE001 — una fuente caída no detiene las demás
                    failed.append(f"{kind} {symbol or ''}: {type(e).__name__}: {str(e)[:160]}")
                    print(f"  FALLÓ {failed[-1]}", flush=True)
                    traceback.print_exc(limit=1)
    print("\nResumen de archivos (manifiesto):")
    man = d.root / "manifest.json"
    print(man.read_text(encoding="utf-8") if man.exists() else "Sin datos")
    for w in d.warnings:
        print("AVISO:", w)
    if failed:
        print("\nFuentes con fallo (sus características quedarán como NaN):")
        for f in failed:
            print("  -", f)


if __name__ == "__main__":
    main()

"""QUANT-AI: sistema de trading algorítmico con IA (modo PAPER por defecto)."""
import sys

# La consola de Windows usa cp1252 por defecto; forzamos UTF-8 para que los logs en español no fallen.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

__version__ = "0.1.0"

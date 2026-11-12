"""Paper trading con datos reales de Binance (sin dinero real).

Uso:
  python scripts/run_paper.py --once        # un ciclo (útil para probar o para cron/Task Scheduler)
  python scripts/run_paper.py               # bucle: un ciclo por hora, 90 s después del cierre de cada vela

Kill switch: crea un archivo llamado KILL_SWITCH en la raíz del proyecto para impedir nuevas compras.
"""
import argparse
import time

import pandas as pd

from quant_ai.brokers.factory import assert_not_live
from quant_ai.config import load_config
from quant_ai.execution.runner import run_cycle


def seconds_to_next_bar(offset_s: int = 90) -> float:
    now = pd.Timestamp.now(tz="UTC")
    nxt = now.floor("1h") + pd.Timedelta(hours=1) + pd.Timedelta(seconds=offset_s)
    return max(5.0, (nxt - now).total_seconds())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--no-update", action="store_true", help="no descargar datos nuevos antes del ciclo")
    a = ap.parse_args()
    assert_not_live()
    cfg = load_config(a.config)
    print(f"Modo: {cfg['project']['mode'].upper()} | símbolos {cfg.symbols}")
    while True:
        try:
            run_cycle(cfg, update_data=not a.no_update)
        except Exception as e:  # noqa: BLE001
            print(f"Ciclo falló: {e}; se reintenta en la próxima vela")
        if a.once:
            break
        wait = seconds_to_next_bar()
        print(f"Próximo ciclo en {wait/60:.1f} min")
        time.sleep(wait)

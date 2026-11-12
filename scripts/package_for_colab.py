"""Empaqueta el código del proyecto en dist/quant-ai.zip para subirlo a Google Drive.

Incluye solo código y configuración (src, scripts, configs, notebooks, tests, docs, dashboard,
pyproject.toml, README.md). Excluye datos, modelos, reportes, estado, entornos virtuales y secretos.
No requiere commits de Git.

Uso:  python scripts/package_for_colab.py              # Colab: dist/quant-ai.zip (código + estado_previo/)
      python scripts/package_for_colab.py --kaggle     # Kaggle: el mismo contenido en dist/quant-ai-kaggle.zip
      python scripts/package_for_colab.py --no-state   # solo código

``estado_previo/`` lleva los datos limpios (los mismos del PC) y el historial de intentos (summary.json de
corridas, backtest_summary.json de DL y eval_summary.json de RL). El cuaderno lo copia a QUANT_AI_HOME sin
sobrescribir nada: en Colab completa el historial que ya hay en Drive y en Kaggle lo aporta entero. Sin ese
historial el Deflated Sharpe contaría menos intentos de los que se han hecho y sería optimista.
Nunca incluye secretos, estado de paper, checkpoints ni predicciones.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE_DIRS = ["src", "scripts", "configs", "notebooks", "tests", "docs", "dashboard"]
INCLUDE_FILES = ["pyproject.toml", "README.md", ".env.example"]
EXCLUDE_PARTS = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache", ".venv", ".git"}
EXCLUDE_SUFFIXES = {".pyc", ".parquet", ".joblib", ".pt", ".ckpt", ".zip"}
EXCLUDE_NAMES = {".env", "KILL_SWITCH"}


def iter_files():
    for name in INCLUDE_FILES:
        p = ROOT / name
        if p.is_file():
            yield p
    for d in INCLUDE_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            if not p.is_file() or EXCLUDE_PARTS & set(p.relative_to(ROOT).parts):
                continue
            if p.suffix in EXCLUDE_SUFFIXES or p.name in EXCLUDE_NAMES:
                continue
            yield p


def iter_state():
    """Archivos de estado para la nube: (ruta local, ruta dentro de estado_previo/)."""
    patterns = ["data/clean/*.parquet", "data/manifest.json", "reports/runs/*/summary.json",
                "models/dl/**/backtest_summary.json", "models/rl/*/eval_summary.json"]
    seen = set()
    for pat in patterns:
        for p in sorted(ROOT.glob(pat)):
            rel = p.relative_to(ROOT)
            if p.is_file() and rel not in seen and not EXCLUDE_PARTS & set(rel.parts) and p.name not in EXCLUDE_NAMES:
                seen.add(rel)
                yield p, "estado_previo/" + rel.as_posix()


def main() -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle", action="store_true", help="escribir dist/quant-ai-kaggle.zip")
    ap.add_argument("--no-state", action="store_true", help="no incluir estado_previo/ (datos limpios e historial)")
    args = ap.parse_args()
    out = ROOT / "dist" / ("quant-ai-kaggle.zip" if args.kaggle else "quant-ai.zip")
    out.parent.mkdir(exist_ok=True)
    files = [(p, p.relative_to(ROOT).as_posix()) for p in iter_files()]
    state = [] if args.no_state else list(iter_state())
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p, arc in files + state:
            z.write(p, arc)
    print(f"{len(files)} archivos de código + {len(state)} de estado -> {out} ({out.stat().st_size / 1024:.0f} KB)")
    if args.kaggle:
        print("Súbelo a Kaggle como dataset privado (Create > New Dataset).")
    else:
        print("Súbelo a Google Drive en: Mi unidad/quant-ai/quant-ai.zip")
    return out


if __name__ == "__main__":
    main()

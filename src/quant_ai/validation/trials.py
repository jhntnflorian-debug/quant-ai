"""Registro acumulado de intentos para el Deflated Sharpe Ratio.

Cada estrategia evaluada sobre el periodo de desarrollo es un "intento". Si solo se contaran los de la
corrida actual, el DSR sería optimista: este tramo ya se miró en corridas anteriores (v1, DL, RL).
Se cuenta cada IDEA una vez: el mismo nombre de estrategia en varias corridas es el mismo intento;
cada semilla de RL y cada variante v2 son intentos distintos. Contar de más es conservador.

El Sharpe de cada idea (que alimenta la dispersión entre intentos) sale de la corrida COMPLETA más
reciente de ``reports/runs``; DL y RL solo lo aportan si ninguna corrida lo dio. Una prueba mínima
(smoke) cuenta como intento, pero nunca aporta Sharpe: 1 fold o 3 meses dan valores extremos que
inflarían la varianza. Las carpetas smoke de DL/RL comparten identidad con su versión real.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from quant_ai.config import Config

DERIVED_SUFFIXES = ("_cost2x", "_cost3x")  # escenarios de coste de una misma idea, no ideas nuevas
SMOKE_SUFFIX = "_smoke"


def _key(kind: str, name: str) -> str:
    return f"{kind}:{name}"


def _without_smoke(rel: str) -> str:
    """Quita el sufijo ``_smoke`` de cada parte de la ruta: la prueba mínima es la misma idea."""
    return "/".join(part.removesuffix(SMOKE_SUFFIX) for part in rel.split("/"))


def _is_smoke_folder(folder: Path, rel: str) -> bool:
    if _without_smoke(rel) != rel:
        return True
    for name in ("run_info.json", "experiment.json"):
        p = folder / name
        if p.is_file():
            try:
                if json.loads(p.read_text(encoding="utf-8")).get("smoke"):
                    return True
            except (OSError, ValueError, AttributeError):
                pass
    return False


def collect_trials(cfg: Config) -> dict[str, dict]:
    trials: dict[str, dict] = {}

    def add(kind, name, sharpe, source, usable, override=False):
        if name.endswith(DERIVED_SUFFIXES):
            return
        t = trials.setdefault(_key(kind, name), {"kind": kind, "name": name, "sharpe_annual": None,
                                                 "sharpe_source": None, "sources": []})
        t["sources"].append(source)
        finite = isinstance(sharpe, (int, float)) and not isinstance(sharpe, bool) and math.isfinite(sharpe)
        if usable and finite and (override or t["sharpe_annual"] is None):
            t["sharpe_annual"], t["sharpe_source"] = float(sharpe), source

    runs = cfg.reports_dir / "runs"
    # Orden cronológico: el nombre de cada corrida empieza con su marca de tiempo UTC.
    for sp in sorted(runs.glob("*/summary.json")) if runs.exists() else []:
        s = json.loads(sp.read_text(encoding="utf-8"))
        if s.get("final_holdout_used"):
            continue  # el holdout no forma parte de la búsqueda en desarrollo
        for name, m in (s.get("strategies") or {}).items():
            add("estrategia", name, m.get("sharpe"), sp.parent.name, not s.get("is_smoke"), override=True)
    dl = cfg.models_dir / "dl"
    for bp in sorted(dl.rglob("backtest_summary.json")) if dl.exists() else []:
        rel = bp.parent.relative_to(dl).as_posix()
        smoke, ident = _is_smoke_folder(bp.parent, rel), _without_smoke(rel)
        for name, m in (json.loads(bp.read_text(encoding="utf-8")).get("strategies") or {}).items():
            if name.startswith("dl_"):
                # Identidad por configuración, no por nombre: GRU v1 y GRU v2 son intentos distintos.
                # Mismos nombres que usa consolidate_v2 para no contar dos veces la misma idea.
                if ident.startswith("secuencias_v2"):
                    name = "dl_v2_" + ident.rsplit("/", 1)[-1]
                elif "chart_cnn" in ident:
                    name = "dl_" + ident.rsplit("/", 1)[-1]
                else:
                    name = f"{ident}:{name}"
            add("estrategia", name, m.get("sharpe"), rel, not smoke)
    rl = cfg.models_dir / "rl"
    for ep in sorted(rl.glob("*/eval_summary.json")) if rl.exists() else []:
        s = json.loads(ep.read_text(encoding="utf-8"))
        run = ep.parent.name
        smoke = _is_smoke_folder(ep.parent, run) or bool(s.get("smoke"))
        for seed, r in (s.get("seeds") or {}).items():
            add("rl", f"{_without_smoke(run)}:seed{seed}", r.get("metrics", {}).get("sharpe"), run, not smoke)
    return trials


def count_trials(cfg: Config, extra_names: list[str] | None = None) -> int:
    """Número total de ideas distintas probadas en desarrollo (incluye las de la corrida en curso)."""
    trials = collect_trials(cfg)
    for n in extra_names or []:
        if not n.endswith(DERIVED_SUFFIXES):
            trials.setdefault(_key("estrategia", n), {})
    return len(trials)


def write_registry(cfg: Config, path: Path | None = None) -> Path:
    trials = collect_trials(cfg)
    path = path or cfg.reports_dir / "trials.json"
    path.write_text(json.dumps({"n_trials": len(trials), "trials": trials}, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path

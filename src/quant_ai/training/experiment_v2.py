"""Experimento v2: ablación de grupos de características, meta-etiquetado, horizonte y volatility targeting.

Diseño (pre-registrado en configs/experiments_v2.yaml; revisión adversarial de Codex incorporada):
- Todas las variantes usan el MISMO walk-forward purgado y el MISMO tramo; los pesos ausentes dentro del
  tramo se convierten en efectivo explícito (nunca se "congela" una posición).
- Mismo tratamiento de stops para ML y reglas de tendencia (no para comprar y mantener).
- Meta-etiquetado: la señal primaria (cruce EMA 24/168) decide la DIRECCIÓN; el modelo solo decide si
  tomarla. Se entrena únicamente en velas con la señal primaria activa; sin señal, la política sale y
  reinicia su estado.
- DSR con TODOS los intentos acumulados sobre el periodo de desarrollo (número y dispersión).
- La puerta exige corrida completa, métricas finitas, referencias presentes y un bootstrap pareado.
- ``consolidate_v2`` re-evalúa juntos, con trato idéntico, todos los candidatos de la familia v2
  (variantes ML + GRU/TCN v2 + CNN de gráficos) tras terminar todas las etapas.
- Nunca usa el holdout final.
"""
from __future__ import annotations

import copy
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from quant_ai.backtesting.metrics import (classification_metrics, deflated_sharpe, paired_sharpe_diff_ci,
                                          probabilistic_sharpe)
from quant_ai.config import PROJECT_ROOT, Config
from quant_ai.decision_engine.policy import PolicyParams, vol_target_weights, weights_from_probs
from quant_ai.labeling.labels import make_labels
from quant_ai.models.baselines import buy_and_hold, ema_cross
from quant_ai.training.dataset import SymbolDataset
from quant_ai.training.experiment import run_strategy_backtests, walk_forward_predictions
from quant_ai.validation.splits import restrict_to_development
from quant_ai.validation.trials import collect_trials, write_registry

V2_CONFIG = PROJECT_ROOT / "configs" / "experiments_v2.yaml"
DERIVED = ("_cost2x", "_cost3x", "_risk")


def load_v2_config(path: str | Path | None = None) -> dict:
    with open(path or V2_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _with(cfg: Config, override: dict) -> Config:
    raw = copy.deepcopy(cfg.raw)

    def upd(a, b):
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                upd(a[k], v)
            else:
                a[k] = v
    upd(raw, override)
    return Config(raw=raw, home=cfg.home)


# ----------------------------------------------------------------------------- características y etiquetas
def build_feature_groups(cfg: Config, datasets: dict[str, SymbolDataset], needed: set[str], log=print
                         ) -> dict[str, dict[str, pd.DataFrame]]:
    """Devuelve {símbolo: {grupo: DataFrame}}. Falla con mensaje claro si falta un grupo requerido."""
    groups: dict[str, dict[str, pd.DataFrame]] = {s: {"base": ds.features[ds.feature_names]} for s, ds in datasets.items()}
    if "ta" in needed:
        try:
            from quant_ai.features.technical import build_technical_features
        except ImportError as e:
            raise RuntimeError("Falta quant_ai.features.technical (grupo 'ta').") from e
        for s, ds in datasets.items():
            ta = build_technical_features(ds.prices).reindex(ds.features.index)
            groups[s]["ta"] = ta[[c for c in ta.columns if c.startswith("ta_")]]
            log(f"  {s}: {groups[s]['ta'].shape[1]} características técnicas")
    if "ext" in needed:
        try:
            from quant_ai.features.external import build_external_features
        except ImportError as e:
            raise RuntimeError("Falta quant_ai.features.external (grupo 'ext').") from e
        for s, ds in datasets.items():
            ext = build_external_features(cfg.data_dir, s, ds.features.index)
            ext = ext[[c for c in ext.columns if c.startswith("ext_")]]
            if ext.empty or ext.notna().sum().sum() == 0:
                raise RuntimeError(f"{s}: no hay datos externos. Ejecuta scripts/download_extended.py primero.")
            groups[s]["ext"] = ext
            log(f"  {s}: {ext.shape[1]} características externas (cobertura media {ext.notna().mean().mean():.0%})")
    return groups


def primary_signal(features: pd.DataFrame, name: str) -> pd.Series:
    if name == "none":
        return pd.Series(True, index=features.index)
    if name == "ema_cross":  # EMA24 > EMA168  <=>  ema_dist_168 > ema_dist_24
        return (features["ema_dist_168"] > features["ema_dist_24"]).fillna(False)
    raise ValueError(f"Señal primaria desconocida: {name}")


def variant_datasets(cfg: Config, datasets: dict[str, SymbolDataset], groups: dict, spec: dict,
                     v2: dict | None = None) -> dict[str, SymbolDataset]:
    out = {}
    cost = cfg["labeling"]["cost_threshold_bps"] / 1e4
    for s, ds in datasets.items():
        feats = pd.concat([groups[s][g] for g in spec["features"]], axis=1)
        feats = feats.loc[:, ~feats.columns.duplicated()]
        if spec.get("label", "forward") == "triple_barrier":
            tb = (v2 or {}).get("triple_barrier", {"pt_atr_mult": 3.0, "sl_atr_mult": 3.0, "max_bars": spec["horizon"]})
            lab = make_labels(ds.prices, {"method": "triple_barrier", "triple_barrier": tb}, cost,
                              atr_pct=ds.features["atr_pct_14"], freq=cfg.interval)
        else:
            lab = make_labels(ds.prices, {"method": "forward_net_return", "horizon_bars": spec["horizon"]}, cost,
                              freq=cfg.interval)
        lab = restrict_to_development(lab, cfg.final_holdout_start)
        prim = primary_signal(ds.features, spec.get("primary", "none")).reindex(lab.index).fillna(False)
        lab.loc[~prim, "y"] = np.nan  # meta-etiquetado: solo se aprende donde la señal primaria está activa
        out[s] = SymbolDataset(s, ds.prices, feats, lab, list(feats.columns))
    return out


# ----------------------------------------------------------------------------- pesos
def variant_weights(cfg: Config, v2: dict, spec: dict, datasets: dict[str, SymbolDataset], preds: pd.DataFrame,
                    prob_col: str = "p_ensemble", unc_col: str | None = "uncertainty") -> dict[str, pd.Series]:
    pol = PolicyParams.from_config(cfg["strategy"])
    out = {}
    for s, ds in datasets.items():
        ps = preds[preds["symbol"] == s]
        prim = primary_signal(ds.features, spec.get("primary", "none")).reindex(ps.index).fillna(False)
        # Sin señal primaria la probabilidad se anula (NaN): la política sale y REINICIA su estado, así que
        # al reactivarse la señal solo entra con p >= enter_prob (sin arrastrar la histéresis).
        unc = ps[unc_col] if unc_col and unc_col in ps else None
        w = weights_from_probs(ps[prob_col].where(prim), unc, pol)
        if spec.get("sizing") == "vol_target":
            vt = vol_target_weights(ds.features["vol_168"].reindex(w.index), v2["sizing"]["vol_target_annual"],
                                    v2["sizing"]["max_weight"])
            w = (w > 0).astype(float) * vt
        out[s] = w
    return out


def rule_weights(cfg: Config, v2: dict, datasets: dict[str, SymbolDataset]) -> dict[str, dict[str, pd.Series]]:
    tw = cfg["strategy"]["target_weight"]
    out = {"buy_and_hold": {}, "ema_cross_24_168": {}, "ema_cross_vol_target": {}}
    for s, ds in datasets.items():
        f = ds.features
        out["buy_and_hold"][s] = buy_and_hold(f, tw)
        out["ema_cross_24_168"][s] = ema_cross(f, tw)
        vt = vol_target_weights(f["vol_168"], v2["sizing"]["vol_target_annual"], v2["sizing"]["max_weight"])
        out["ema_cross_vol_target"][s] = (ema_cross(f, 1.0) > 0).astype(float) * vt
    return {k: v for k, v in out.items() if k in v2.get("rule_benchmarks", out.keys())}


def complete_weights(weights: dict, datasets: dict[str, SymbolDataset], start, end) -> dict:
    """Cada estrategia tiene peso en TODAS las velas del tramo: lo ausente es efectivo (0), nunca
    'mantener lo anterior'. Evita posiciones congeladas en trimestres sin predicción."""
    out = {}
    for name, per in weights.items():
        out[name] = {}
        for s, ds in datasets.items():
            idx = ds.features.index[(ds.features.index >= start) & (ds.features.index <= end)]
            out[name][s] = per[s].reindex(idx).fillna(0.0)
    return out


# ----------------------------------------------------------------------------- evaluación común
def evaluate_candidates(cfg: Config, v2: dict, datasets: dict[str, SymbolDataset], weights: dict,
                        start, end, is_complete: bool, log=print) -> dict:
    """Backtests en todos los escenarios, DSR acumulado, bootstrap pareado y puerta. Trato idéntico."""
    weights = complete_weights(weights, datasets, start, end)
    timing = {k for k in weights if k != "buy_and_hold"}  # mismo stop para ML, DL y reglas de tendencia
    scenarios, equities, extra = {}, {}, {}
    for scen, costs in v2["cost_scenarios"].items():
        log(f"== Backtests escenario {scen} ({start} -> {end})")
        res, eq, trades, fills = run_strategy_backtests(_with(cfg, {"costs": costs}), datasets, weights, start,
                                                        log=log, end=end, stop_names=timing)
        scenarios[scen], equities[scen] = res, eq
        if scen == "taker_base":
            extra = {"trades": trades, "fills": fills}
    eq0 = equities["taker_base"]
    ann = cfg["backtest"]["annualization_bars"]
    names = [c for c in eq0.columns if not c.endswith(DERIVED)]
    rets = {c: eq0[c].pct_change().dropna().to_numpy() for c in names}
    # Dispersión y número de intentos: historial acumulado + esta evaluación (la actual reemplaza a la vieja)
    hist = collect_trials(cfg)
    # Solo None es "sin Sharpe" (p. ej. ideas vistas solo en pruebas mínimas); un Sharpe de 0.0 es legítimo.
    srs = {k: (np.nan if v.get("sharpe_annual") is None else v["sharpe_annual"]) / np.sqrt(ann)
           for k, v in hist.items()}
    for c in names:
        r = rets[c]
        srs[f"estrategia:{c}"] = r.mean() / r.std(ddof=1) if len(r) > 2 and r.std(ddof=1) > 0 else np.nan
    finite = [v for v in srs.values() if np.isfinite(v)]
    n_trials = len(srs)
    log(f"Intentos acumulados para el DSR: {n_trials} ({len(finite)} con Sharpe finito)")
    dsr = {}
    for c in names:
        d = deflated_sharpe(rets[c], finite, n_trials=n_trials) if len(finite) >= 2 else {"dsr": np.nan, "sr0": np.nan,
                                                                                         "n_trials": n_trials}
        dsr[c] = {**d, "psr": probabilistic_sharpe(rets[c])}
    boot = {}
    for c in names:
        if c.startswith(("ml_", "dl_")):
            boot[c] = {ref: paired_sharpe_diff_ci(rets[c], rets[ref], ann, block=v2["gate"].get("bootstrap_block", 168))
                       for ref in v2["gate"]["must_beat"] if ref in rets}
    gate = evaluate_gate(v2, scenarios, dsr, boot, is_complete)
    return {"scenarios": scenarios, "equity": eq0, "dsr": dsr, "bootstrap": boot, "gate": gate,
            "n_trials": n_trials, **extra}


def _finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and np.isfinite(x)


def evaluate_gate(v2: dict, scenarios: dict, dsr: dict, boot: dict | None = None, is_complete: bool = True) -> dict:
    g = v2["gate"]
    base = scenarios["taker_base"]
    stress = scenarios.get(g.get("positive_under", "stress_x2"))
    out = {}
    for name, m in base.items():
        if not name.startswith(("ml_", "dl_")) or name.endswith(DERIVED):
            continue
        reasons = []
        if not is_complete:
            reasons.append("evaluación incompleta (prueba mínima o folds limitados)")
        d = dsr.get(name, {}).get("dsr")
        if not _finite(d) or d < g["min_dsr"]:
            reasons.append(f"DSR {round(d, 3) if _finite(d) else 'no disponible'} < {g['min_dsr']}")
        for key in ("sharpe", "max_drawdown", "total_return"):
            if not _finite(m.get(key)):
                reasons.append(f"{key} no finito")
        for ref in g["must_beat"]:
            if ref not in base:
                reasons.append(f"falta la referencia {ref}")
                continue
            if not (_finite(m.get("sharpe")) and m["sharpe"] > base[ref]["sharpe"]):
                reasons.append(f"Sharpe {m.get('sharpe', float('nan')):.2f} <= {ref} {base[ref]['sharpe']:.2f}")
            b = (boot or {}).get(name, {}).get(ref)
            if b is None or not _finite(b.get("lo")) or b["lo"] <= 0:
                lo = b.get("lo") if b else None
                reasons.append(f"IC95 % de la ventaja de Sharpe frente a {ref} incluye 0 "
                               f"(inferior {round(lo, 2) if _finite(lo) else 'n/d'})")
        ref_dd = base.get(g["max_drawdown_not_worse_than"], {}).get("max_drawdown")
        if not _finite(ref_dd):
            reasons.append("falta el drawdown de referencia")
        elif _finite(m.get("max_drawdown")) and m["max_drawdown"] < ref_dd:
            reasons.append("drawdown peor que la referencia")
        if stress is None or name not in stress:
            reasons.append(f"falta el escenario {g.get('positive_under')}")
        elif not (_finite(stress[name].get("total_return")) and stress[name]["total_return"] > 0):
            reasons.append(f"retorno con estrés de costes <= 0")
        out[name] = {"passed": not reasons, "reasons": reasons}
    return out


# ----------------------------------------------------------------------------- orquestación
def _save(out: Path, summary: dict, ev: dict, preds: pd.DataFrame | None = None, imps: list | None = None):
    ev["equity"].to_parquet(out / "equity.parquet")
    if len(ev.get("trades", [])):
        ev["trades"].to_parquet(out / "trades.parquet")
    if len(ev.get("fills", [])):
        ev["fills"].to_parquet(out / "fills.parquet")
    if preds is not None and len(preds):
        preds.to_parquet(out / "predictions.parquet")
    if imps:
        fi = (pd.concat(imps).groupby(["variant", "symbol", "feature"])[["lgbm_gain_pct"]].mean()
              .reset_index().sort_values(["variant", "symbol", "lgbm_gain_pct"], ascending=[True, True, False]))
        fi.to_csv(out / "feature_importance.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "report.md").write_text(render_v2_report(summary), encoding="utf-8")


def run_experiment_v2(cfg: Config, datasets: dict[str, SymbolDataset], v2: dict | None = None,
                      only: list[str] | None = None, max_folds: int | None = None, tag: str = "v2",
                      log=print) -> Path:
    t0 = time.perf_counter()
    v2 = v2 or load_v2_config()
    variants = {k: v for k, v in v2["variants"].items() if not only or k in only}
    cfg_v2 = _with(cfg, {"splits": {"walk_forward": {"first_test_start": v2["test_start"]}}})
    needed = {g for spec in variants.values() for g in spec["features"]}
    log(f"Grupos requeridos: {sorted(needed)}")
    groups = build_feature_groups(cfg, datasets, needed, log=log)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{tag}"
    out = cfg.reports_dir / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)

    weights, all_preds, all_imps, clf, folds_info = {}, [], [], {}, {}
    for name, spec in variants.items():
        log(f"== Variante {name}: {spec}")
        vds = variant_datasets(cfg_v2, datasets, groups, spec, v2)
        cfg_var = _with(cfg_v2, {"model": {"missing_indicator": "ext" in spec["features"]}})
        preds, imps, folds = walk_forward_predictions(cfg_var, vds, max_folds=max_folds, log=log)
        folds_info[name] = folds
        weights[f"ml_{name}"] = variant_weights(cfg, v2, spec, datasets, preds)
        all_preds.append(preds.assign(variant=name))
        if len(imps):
            all_imps.append(imps.assign(variant=name))
        lab = preds[preds["y"].notna()]
        for s in datasets:
            ls = lab[lab["symbol"] == s]
            if len(ls) and ls["y"].nunique() > 1:
                m = classification_metrics(ls["y"].to_numpy(), ls["p_ensemble"].to_numpy())
                sel = ls["p_ensemble"] >= cfg["strategy"]["enter_prob"]
                m["precision_at_enter"] = float(ls.loc[sel, "y"].mean()) if sel.any() else None
                m["coverage_at_enter"] = float(sel.mean())
                clf[f"{name}:{s}"] = m
    preds_all = pd.concat(all_preds) if all_preds else pd.DataFrame()
    start = pd.Timestamp(v2["test_start"], tz="UTC")
    end = preds_all.index.max() if len(preds_all) else start
    weights.update(rule_weights(cfg, v2, datasets))
    is_complete = not max_folds
    ev = evaluate_candidates(cfg, v2, datasets, weights, start, end, is_complete, log=log)
    summary = {
        "run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(), "kind": "v2",
        "is_smoke": not is_complete, "final_holdout_used": False,
        "period": {"start": str(start), "end": str(end)}, "final_holdout_start": str(cfg.final_holdout_start),
        "symbols": list(datasets), "variants": variants, "strategies": ev["scenarios"]["taker_base"],
        "scenarios": ev["scenarios"], "dsr": ev["dsr"], "bootstrap": ev["bootstrap"], "gate": ev["gate"],
        "n_trials_cumulative": ev["n_trials"], "classification": clf,
        "folds": [dict(f, variant=v) for v, fs in folds_info.items() for f in fs],
        "v2_config": v2, "config": cfg.raw, "seconds": round(time.perf_counter() - t0, 1),
    }
    _save(out, summary, ev, preds_all, all_imps)
    write_registry(cfg)
    log(f"Listo en {summary['seconds']}s -> {out}")
    return out


def consolidate_v2(cfg: Config, datasets: dict[str, SymbolDataset], v2: dict | None = None, log=print) -> Path:
    """Evaluación final de la familia v2 con trato idéntico: variantes ML (última corrida v2 completa),
    GRU/TCN v2 (models/dl/secuencias_v2/*) y CNN (models/dl/chart_cnn_w*). Ejecutar al final de todo."""
    v2 = v2 or load_v2_config()
    runs = sorted(p for p in (cfg.reports_dir / "runs").glob("*_v2") if (p / "predictions.parquet").exists())
    runs = [p for p in runs if not json.loads((p / "summary.json").read_text(encoding="utf-8")).get("is_smoke")]
    if not runs:
        raise RuntimeError("No hay una corrida v2 completa (scripts/run_experiments_v2.py sin --max-folds).")
    src = runs[-1]
    s = json.loads((src / "summary.json").read_text(encoding="utf-8"))
    preds = pd.read_parquet(src / "predictions.parquet")
    weights, ranges = {}, []
    for name, spec in s["variants"].items():
        p = preds[preds["variant"] == name]
        weights[f"ml_{name}"] = variant_weights(cfg, v2, spec, datasets, p)
        ranges.append((p.index.min(), p.index.max()))
    dl_sources = {}
    dl_root = cfg.models_dir / "dl"
    for pp in sorted(dl_root.glob("secuencias_v2/*/predictions.parquet")) + sorted(dl_root.glob("chart_cnn_w*/predictions.parquet")):
        if "smoke" in str(pp.parent):
            continue
        name = "dl_v2_" + pp.parent.name if "secuencias_v2" in str(pp) else "dl_" + pp.parent.name
        dp = pd.read_parquet(pp).sort_index()
        weights[name] = variant_weights(cfg, v2, {"primary": "none", "sizing": "fixed"}, datasets, dp,
                                        prob_col="p_seq", unc_col=None)
        ranges.append((dp.index.min(), dp.index.max()))
        dl_sources[name] = str(pp.parent)
    start = max(r[0] for r in ranges)
    end = min(r[1] for r in ranges)
    log(f"Ventana común de evaluación: {start} -> {end} ({len(weights)} candidatos + referencias)")
    weights.update(rule_weights(cfg, v2, datasets))
    ev = evaluate_candidates(cfg, v2, datasets, weights, start, end, is_complete=True, log=log)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_v2final"
    out = cfg.reports_dir / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(), "kind": "v2_final",
        "is_smoke": False, "final_holdout_used": False, "source_v2_run": src.name, "dl_sources": dl_sources,
        "period": {"start": str(start), "end": str(end)}, "final_holdout_start": str(cfg.final_holdout_start),
        "symbols": list(datasets), "variants": s["variants"], "strategies": ev["scenarios"]["taker_base"],
        "scenarios": ev["scenarios"], "dsr": ev["dsr"], "bootstrap": ev["bootstrap"], "gate": ev["gate"],
        "n_trials_cumulative": ev["n_trials"], "classification": s.get("classification", {}), "v2_config": v2,
    }
    _save(out, summary, ev)
    write_registry(cfg)
    log(f"Consolidación lista -> {out}")
    return out


def render_v2_report(s: dict) -> str:
    def pct(x):
        return "—" if not _finite(x) else f"{x:+.1%}"

    def num(x, d=2):
        return "—" if not _finite(x) else f"{x:.{d}f}"

    L = [f"# Experimento v2 `{s['run_id']}` ({s['kind']})", "",
         f"- Tramo de prueba (walk-forward): {s['period']['start']} → {s['period']['end']}"
         + (" — **PRUEBA MÍNIMA, no válida**" if s.get("is_smoke") else ""),
         f"- Holdout final ({s['final_holdout_start']}): sin tocar",
         f"- Intentos contados en el DSR: {s.get('n_trials_cumulative')} acumulados sobre el periodo de desarrollo",
         "- Stop-loss ATR y regla de señal nueva tras un stop: iguales para ML, DL y reglas de tendencia; "
         "no aplican a comprar y mantener", ""]
    refs = s["v2_config"]["gate"]["must_beat"]
    L += ["| Estrategia | Retorno | Sharpe | Máx. DD | Operaciones | PSR | DSR | "
          + " | ".join(f"ΔSharpe vs {r} (IC95 %)" for r in refs) + " | Estrés ×2 | BNB 7.5 bps | Puerta |",
          "|---|---|---|---|---|---|---|" + "---|" * len(refs) + "---|---|---|"]
    base = s["scenarios"]["taker_base"]
    stress, bnb = s["scenarios"].get("stress_x2", {}), s["scenarios"].get("taker_bnb", {})
    for name, m in base.items():
        if name.endswith(DERIVED):
            continue
        d = s["dsr"].get(name, {})
        g = s["gate"].get(name)
        bcols = []
        for r in refs:
            b = s.get("bootstrap", {}).get(name, {}).get(r)
            bcols.append(f"{num(b['diff'])} ({num(b['lo'])}, {num(b['hi'])})" if b else "—")
        L.append(f"| {name} | {pct(m.get('total_return'))} | {num(m.get('sharpe'))} | {pct(m.get('max_drawdown'))} | "
                 f"{m.get('n_trades', 0)} | {num(d.get('psr'), 3)} | {num(d.get('dsr'), 3)} | " + " | ".join(bcols) +
                 f" | {pct(stress.get(name, {}).get('total_return'))} | {pct(bnb.get(name, {}).get('total_return'))} | "
                 f"{'PASA' if g and g['passed'] else ('no' if g else 'ref.')} |")
    if s.get("classification"):
        L += ["", "## Calidad predictiva fuera de muestra (variantes ML)", "",
              "| Variante:símbolo | n | Tasa base | AUC | Brier | Brier base | Precisión@entrada | Cobertura |",
              "|---|---|---|---|---|---|---|---|"]
        for k, m in s["classification"].items():
            L.append(f"| {k} | {m['n']} | {num(m['base_rate'], 3)} | {num(m['auc'], 3)} | {num(m['brier'], 4)} | "
                     f"{num(m['brier_baseline'], 4)} | "
                     f"{num(m['precision_at_enter'], 3) if m.get('precision_at_enter') is not None else '—'} | "
                     f"{pct(m['coverage_at_enter'])} |")
    failing = {k: v["reasons"] for k, v in s["gate"].items() if not v["passed"]}
    if failing:
        L += ["", "## Motivos de rechazo de la puerta", ""] + [f"- **{k}**: " + "; ".join(v) for k, v in failing.items()]
    L += ["", "## Lectura", "",
          "- Ablación: v1 → v2 (técnico) / v2b (externo) → v3 (ambos), mismo horizonte y tamaño.",
          "- Meta-etiquetado: v4, v5 y v6 frente a ema_cross_24_168 y ema_cross_vol_target (misma señal, sin ML).",
          "- DSR < 0.95 o IC95 % que incluye 0: no hay evidencia de ventaja frente a la suerte de haber probado muchas ideas.",
          "- Resultados simulados; no son beneficios reales.", ""]
    return "\n".join(L)

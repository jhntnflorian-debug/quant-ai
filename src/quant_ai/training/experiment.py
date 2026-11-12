"""Experimentos reproducibles: walk-forward purgado + backtests con costes + reportes.

Artefactos en ``reports/runs/<run_id>/`` (los consume el dashboard):
- summary.json          métricas por estrategia, clasificación por modelo, folds, configuración, hashes de datos
- equity.parquet        curva de capital por estrategia (index=open_time)
- trades.parquet        operaciones ida y vuelta (columna strategy)
- fills.parquet         ejecuciones individuales (columna strategy)
- predictions.parquet   predicciones fuera de muestra (OOS) por símbolo y fold
- feature_importance.csv, calibration.csv, by_year.csv, by_regime.csv
- report.md             reporte legible
"""
from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from quant_ai.backtesting.engine import CostModel, Rules, run_backtest
from quant_ai.backtesting.metrics import (calibration_table, classification_metrics, performance_metrics,
                                          period_stability)
from quant_ai.config import Config
from quant_ai.decision_engine.policy import PolicyParams, weights_from_probs
from quant_ai.models.baselines import RULE_STRATEGIES
from quant_ai.models.ml import EnsembleModel
from quant_ai.risk_management.engine import RiskLimits, backtest_risk_hook
from quant_ai.training.dataset import SymbolDataset
from quant_ai.validation.splits import purged_walk_forward


# ----------------------------------------------------------------------------- walk-forward ML
def walk_forward_predictions(cfg: Config, datasets: dict[str, SymbolDataset], max_folds: int | None = None,
                             log=print) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    wf = cfg["splits"]["walk_forward"]
    embargo = pd.Timedelta(cfg.interval) * cfg["splits"]["embargo_bars"]
    seed, n_jobs = cfg["project"]["seed"], cfg.get("compute", {}).get("n_jobs", 2)
    preds, imps, folds_info = [], [], []
    for sym, ds in datasets.items():
        fr = ds.frame()
        fr = fr[fr["y"].notna()]
        folds = purged_walk_forward(fr.index, fr["t1"], wf["first_test_start"], wf["test_months"],
                                    wf["min_train_months"], embargo, wf["calibration_fraction"],
                                    end=ds.features.index.max(), expanding=wf.get("expanding", True))
        if max_folds:
            folds = folds[:max_folds]
        for f in folds:
            t0 = time.perf_counter()
            fit, cal = fr.iloc[f.train_idx], fr.iloc[f.calib_idx]
            model = EnsembleModel(ds.feature_names, seed=seed, n_jobs=n_jobs,
                                  missing_indicator=cfg.get("model", {}).get("missing_indicator", False))
            model.fit(fit, fit["y"].to_numpy(), cal, cal["y"].to_numpy(), fit["fwd_ret"].to_numpy())
            # Se predice en TODAS las velas del tramo de prueba (también las que no tienen etiqueta)
            test_feats = ds.features[(ds.features.index >= f.test_start) & (ds.features.index < f.test_end)]
            p = model.predict(test_feats)
            p = p.join(ds.labels[["y", "fwd_ret"]], how="left")
            p["symbol"], p["fold"] = sym, f.k
            preds.append(p)
            imp = model.feature_importance()
            imp["symbol"], imp["fold"] = sym, f.k
            imps.append(imp)
            info = {"symbol": sym, "fold": f.k, "test_start": str(f.test_start), "test_end": str(f.test_end),
                    "n_fit": len(fit), "n_calib": len(cal), "n_test": len(test_feats),
                    "fit_start": str(fit.index.min()), "fit_end": str(fit.index.max()),
                    "calib_end": str(cal.index.max()), "seconds": round(time.perf_counter() - t0, 2)}
            folds_info.append(info)
            log(f"  {sym} fold {f.k:02d} test {f.test_start.date()}→{f.test_end.date()} "
                f"fit={len(fit)} cal={len(cal)} test={len(test_feats)} ({info['seconds']}s)")
    pred_df = pd.concat(preds).sort_index() if preds else pd.DataFrame()
    imp_df = pd.concat(imps) if imps else pd.DataFrame()
    return pred_df, imp_df, folds_info


# ----------------------------------------------------------------------------- regímenes
def market_regimes(btc_features: pd.DataFrame) -> pd.DataFrame:
    """Régimen causal por vela a partir de BTC: tendencia (retorno 30d) y volatilidad (percentil expansivo)."""
    r30 = btc_features["ret_1"].rolling(720, min_periods=720).sum()
    trend = pd.Series("lateral", index=btc_features.index)
    trend[r30 > 0.10] = "alcista"
    trend[r30 < -0.10] = "bajista"
    trend[r30.isna()] = "desconocido"
    vol = btc_features["vol_168"]
    q80 = vol.expanding(min_periods=24 * 90).quantile(0.8)
    vol_reg = np.where(vol > q80, "alta_vol", "normal_vol")
    return pd.DataFrame({"trend": trend, "vol": vol_reg}, index=btc_features.index)


def regime_breakdown(equity: pd.Series, regimes: pd.DataFrame, ann: int) -> list[dict]:
    r = equity.pct_change().dropna()
    reg = regimes.reindex(r.index)
    out = []
    for col in ("trend", "vol"):
        for name, grp in r.groupby(reg[col]):
            if len(grp) < 48:
                continue
            sd = grp.std()
            out.append({"dimension": col, "regime": name, "bars": int(len(grp)),
                        "ann_return": float(grp.mean() * ann),
                        "sharpe": float(grp.mean() / sd * np.sqrt(ann)) if sd > 0 else 0.0})
    return out


# ----------------------------------------------------------------------------- backtests
def _rules_from_cfg(cfg: Config) -> dict[str, Rules]:
    return {s: Rules(v["lot_step"], v["min_notional"]) for s, v in cfg["instrument_rules"].items()}


def strategy_weights(cfg: Config, datasets: dict[str, SymbolDataset], preds: pd.DataFrame | None
                     ) -> dict[str, dict[str, pd.Series]]:
    tw = cfg["strategy"]["target_weight"]
    out: dict[str, dict[str, pd.Series]] = {}
    for name, fn in RULE_STRATEGIES.items():
        out[name] = {s: fn(ds.features, tw) for s, ds in datasets.items()}
    if preds is not None and len(preds):
        pol = PolicyParams.from_config(cfg["strategy"])
        for col, name in (("p_logistic", "ml_logistic"), ("p_lightgbm", "ml_lightgbm"),
                          ("p_ensemble", "ml_ensemble")):
            out[name] = {}
            for s in datasets:
                ps = preds[preds["symbol"] == s]
                unc = ps["uncertainty"] if col == "p_ensemble" else None
                out[name][s] = weights_from_probs(ps[col], unc, pol)
    return out


def run_strategy_backtests(cfg: Config, datasets: dict[str, SymbolDataset], weights: dict, start: pd.Timestamp,
                           log=print, end: pd.Timestamp | None = None, stop_names: set[str] | None = None
                           ) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Backtest de todas las estrategias sobre el MISMO periodo [start, end] (comparación justa).

    ``stop_names``: estrategias que llevan stop-loss ATR. Si es None se conserva el criterio histórico
    (prefijos ml_/dl_). El experimento v2 lo pasa explícito para que ML y reglas de tendencia compitan
    con el MISMO tratamiento de stops.
    """
    rules = _rules_from_cfg(cfg)
    cash = cfg["backtest"]["initial_cash"]
    end = end or pd.Timestamp.max.tz_localize("UTC")
    prices = {s: ds.prices[(ds.prices["open_time"] >= start) & (ds.prices["open_time"] <= end)]
              [["open_time", "open", "high", "low", "close", "volume"]] for s, ds in datasets.items()}
    stops = {s: cfg["strategy"]["atr_stop_mult"] * ds.features["atr_pct_14"] for s, ds in datasets.items()}
    vol24 = {s: ds.features["vol_24"] for s, ds in datasets.items()}
    limits = RiskLimits.from_config(cfg["risk"])
    variants = []
    for name in weights:
        use_stop = (name in stop_names) if stop_names is not None else name.startswith(("ml_", "dl_"))
        variants.append((name, name, 1.0, use_stop, False))
    if "ml_ensemble" in weights:
        variants += [("ml_ensemble_risk", "ml_ensemble", 1.0, True, True),
                     ("ml_ensemble_cost2x", "ml_ensemble", 2.0, True, False),
                     ("ml_ensemble_cost3x", "ml_ensemble", 3.0, True, False)]
    if "ema_cross_24_168" in weights:
        variants += [("ema_cross_24_168_cost2x", "ema_cross_24_168", 2.0,
                      stop_names is not None and "ema_cross_24_168" in stop_names, False)]
    results, equities, trades_all, fills_all = {}, {}, [], []
    for label, wname, cmult, use_stop, use_risk in variants:
        w = {s: weights[wname][s][(weights[wname][s].index >= start) & (weights[wname][s].index <= end)]
             for s in datasets}
        # comprar y mantener: una sola compra, sin rebalanceo posterior
        band = np.inf if wname == "buy_and_hold" else 0.05
        res = run_backtest(prices, w, CostModel.from_config(cfg["costs"], cmult), rules, cash,
                           stops=stops if use_stop else None, rebalance_band=band,
                           risk_hook=backtest_risk_hook(limits, vol24) if use_risk else None)
        m = performance_metrics(res.equity["equity"], res.trades, res.fills, res.equity["gross_exposure"],
                                ann=cfg["backtest"]["annualization_bars"])
        m.update({"cost_multiplier": cmult, "stop_loss": use_stop, "risk_engine": use_risk,
                  "open_positions_at_end": res.meta["open_positions"]})
        results[label] = m
        equities[label] = res.equity["equity"]
        if len(res.trades):
            trades_all.append(res.trades.assign(strategy=label))
        if len(res.fills):
            fills_all.append(res.fills.assign(strategy=label))
        log(f"  {label:<22} ret={m['total_return']:+.1%} sharpe={m['sharpe']:.2f} "
            f"maxDD={m['max_drawdown']:.1%} trades={m.get('n_trades', 0)}")
    eq = pd.DataFrame(equities)
    tr = pd.concat(trades_all, ignore_index=True) if trades_all else pd.DataFrame()
    fl = pd.concat(fills_all, ignore_index=True) if fills_all else pd.DataFrame()
    return results, eq, tr, fl


# ----------------------------------------------------------------------------- orquestación
def run_experiment(cfg: Config, datasets: dict[str, SymbolDataset], skip_ml: bool = False,
                   max_folds: int | None = None, tag: str = "", final: bool = False, log=print) -> Path:
    t_start = time.perf_counter()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + (f"_{tag}" if tag else "")
    out = cfg.reports_dir / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    ann = cfg["backtest"]["annualization_bars"]
    start = pd.Timestamp(cfg["splits"]["walk_forward"]["first_test_start"], tz="UTC")
    if final:
        start = cfg.final_holdout_start

    preds, imps, folds = (None, pd.DataFrame(), [])
    end = None
    if not skip_ml:
        log("Walk-forward ML (fuera de muestra)…")
        preds, imps, folds = walk_forward_predictions(cfg, datasets, max_folds=max_folds, log=log)
        if len(preds):
            start, end = max(start, preds.index.min()), preds.index.max()
    log(f"Backtests {start} → {end or 'fin de datos'}…")
    weights = strategy_weights(cfg, datasets, preds)
    results, eq, trades, fills = run_strategy_backtests(cfg, datasets, weights, start, log=log, end=end)

    # Clasificación y calibración
    clf, calib_rows = {}, []
    if preds is not None and len(preds):
        lab = preds[preds["y"].notna()]
        for s in datasets:
            ls = lab[lab["symbol"] == s]
            for col in ("p_logistic", "p_lightgbm", "p_ensemble"):
                if not len(ls):
                    continue
                clf[f"{s}:{col}"] = classification_metrics(ls["y"].to_numpy(), ls[col].to_numpy())
                clf[f"{s}:{col}"]["precision_at_enter"] = float(
                    ls.loc[ls[col] >= cfg["strategy"]["enter_prob"], "y"].mean()) if (ls[col] >= cfg["strategy"]["enter_prob"]).any() else None
                clf[f"{s}:{col}"]["coverage_at_enter"] = float((ls[col] >= cfg["strategy"]["enter_prob"]).mean())
                ct = calibration_table(ls["y"].to_numpy(), ls[col].to_numpy())
                calib_rows.append(ct.assign(symbol=s, model=col))
        preds.to_parquet(out / "predictions.parquet")
    if len(imps):
        fi = (imps.groupby(["symbol", "feature"])[["lgbm_gain_pct", "logistic_coef"]].mean()
              .reset_index().sort_values(["symbol", "lgbm_gain_pct"], ascending=[True, False]))
        fi.to_csv(out / "feature_importance.csv", index=False)
    if calib_rows:
        pd.concat(calib_rows).to_csv(out / "calibration.csv", index=False)

    # Estabilidad por año y régimen
    btc = next((ds for s, ds in datasets.items() if s.startswith("BTC")), next(iter(datasets.values())))
    regimes = market_regimes(btc.features)
    by_year = pd.concat([period_stability(eq[c].dropna()).assign(strategy=c) for c in eq.columns])
    by_year.index = by_year.index.year
    by_year.rename_axis("year").reset_index().to_csv(out / "by_year.csv", index=False)
    by_regime = pd.DataFrame([{**r, "strategy": c} for c in eq.columns
                              for r in regime_breakdown(eq[c].dropna(), regimes, ann)])
    by_regime.to_csv(out / "by_regime.csv", index=False)

    eq.to_parquet(out / "equity.parquet")
    if len(trades):
        trades.to_parquet(out / "trades.parquet")
    if len(fills):
        fills.to_parquet(out / "fills.parquet")

    manifest_p = cfg.data_dir / "manifest.json"
    data_hashes = {}
    if manifest_p.exists():
        man = json.loads(manifest_p.read_text(encoding="utf-8"))
        data_hashes = {s: {"sha256": v.get("clean_sha256"), "rows": v.get("rows")} for s, v in man.get("latest", {}).items()}
    summary = {
        "run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(), "tag": tag,
        "kind": "final_holdout" if final else ("smoke" if max_folds else ("baselines_only" if skip_ml else "full")),
        "is_smoke": bool(max_folds), "final_holdout_used": final,
        "period": {"start": str(start), "end": str(eq.index.max()) if len(eq) else None},
        "final_holdout_start": str(cfg.final_holdout_start),
        "symbols": list(datasets), "strategies": results, "classification": clf, "folds": folds,
        "config": cfg.raw, "data": data_hashes,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "seconds": round(time.perf_counter() - t_start, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "report.md").write_text(render_report(summary, by_year, by_regime, out), encoding="utf-8")
    (cfg.reports_dir / "latest_run.txt").write_text(run_id, encoding="utf-8")
    log(f"Listo en {summary['seconds']}s → {out}")
    return out


def _pct(x):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.1%}"


def _num(x, d=2):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def render_report(s: dict, by_year: pd.DataFrame, by_regime: pd.DataFrame, out: Path) -> str:
    L = [f"# Reporte de experimento `{s['run_id']}`", "",
         f"- Tipo: **{s['kind']}**" + (" (prueba mínima: NO son resultados válidos)" if s["is_smoke"] else ""),
         f"- Periodo evaluado (fuera de muestra): {s['period']['start']} → {s['period']['end']}",
         f"- Holdout final ({s['final_holdout_start']} en adelante): "
         + ("**USADO**" if s["final_holdout_used"] else "sin tocar"),
         f"- Costes: comisión {s['config']['costs']['fee_bps']} bps + medio spread "
         f"{s['config']['costs']['half_spread_bps']} bps + slippage {s['config']['costs']['slippage_bps']} bps por lado",
         "", "## Estrategias (backtest con costes, señal al cierre → ejecución en la siguiente apertura)", "",
         "| Estrategia | Retorno | CAGR | Sharpe (IC95 %) | Sortino | Máx. DD | Operaciones | Win rate | Profit factor | Exposición | Costes/capital |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in s["strategies"].items():
        ci = m.get("sharpe_ci95") or [np.nan, np.nan]
        L.append(f"| {name} | {_pct(m['total_return'])} | {_pct(m.get('cagr'))} | {_num(m['sharpe'])} "
                 f"({_num(ci[0])}, {_num(ci[1])}) | {_num(m['sortino'])} | {_pct(m['max_drawdown'])} | "
                 f"{m.get('n_trades', 0)} | {_pct(m.get('win_rate')) if m.get('n_trades') else '—'} | "
                 f"{_num(m.get('profit_factor'))} | {_pct(m.get('avg_gross_exposure'))} | {_pct(m.get('costs_pct_initial'))} |")
    if s["classification"]:
        L += ["", "## Clasificación fuera de muestra (y=1: comprar supera costes en el horizonte)", "",
              "| Símbolo:modelo | Tasa base | AUC | Brier | Brier base | Bal. acc. | Precisión@entrada | Cobertura@entrada |",
              "|---|---|---|---|---|---|---|---|"]
        for k, m in s["classification"].items():
            L.append(f"| {k} | {_num(m['base_rate'], 3)} | {_num(m['auc'], 3)} | {_num(m['brier'], 4)} | "
                     f"{_num(m['brier_baseline'], 4)} | {_num(m['balanced_accuracy'], 3)} | "
                     f"{_num(m['precision_at_enter'], 3) if m['precision_at_enter'] is not None else '—'} | "
                     f"{_pct(m['coverage_at_enter'])} |")
    if len(by_year):
        piv = by_year.pivot_table(index="strategy", columns=by_year.index, values="return")
        L += ["", "## Retorno por año", "", "| Estrategia | " + " | ".join(str(c) for c in piv.columns) + " |",
              "|---|" + "---|" * len(piv.columns)]
        for st, row in piv.iterrows():
            L.append(f"| {st} | " + " | ".join(_pct(v) for v in row.values) + " |")
    if len(by_regime):
        L += ["", "## Por régimen de mercado (BTC; definido de forma causal)", "",
              "| Estrategia | Dimensión | Régimen | Velas | Retorno anualizado | Sharpe |", "|---|---|---|---|---|---|"]
        for _, r in by_regime.iterrows():
            L.append(f"| {r['strategy']} | {r['dimension']} | {r['regime']} | {r['bars']} | "
                     f"{_pct(r['ann_return'])} | {_num(r['sharpe'])} |")
    L += ["", "## Advertencias", "",
          "- Resultados simulados: no equivalen a beneficios reales.",
          "- Los IC95 % del Sharpe usan bootstrap de bloques de 168 h; son aproximados.",
          "- Los umbrales de la estrategia están pre-registrados en configs/base.yaml (no optimizados sobre estos folds).",
          ""]
    return "\n".join(L)


# ----------------------------------------------------------------------------- modelo de producción
def train_production_model(cfg: Config, datasets: dict[str, SymbolDataset], run_summary: dict | None = None,
                           log=print) -> Path:
    """Entrena con todo el periodo de desarrollo (ajuste 80 % / calibración 20 % cronológico) y registra
    el modelo como CANDIDATO. La promoción a ACTIVO exige pasar la puerta de validación."""
    import joblib

    version = datetime.now(timezone.utc).strftime("v%Y%m%dT%H%M%SZ")
    out = cfg.models_dir / "registry" / version
    out.mkdir(parents=True, exist_ok=True)
    frac = cfg["splits"]["walk_forward"]["calibration_fraction"]
    meta = {"version": version, "created_at": datetime.now(timezone.utc).isoformat(), "symbols": {},
            "labeling": cfg["labeling"], "strategy": cfg["strategy"], "status": "candidate"}
    for s, ds in datasets.items():
        fr = ds.frame()
        fr = fr[fr["y"].notna()]
        n_cal = int(len(fr) * frac)
        fit, cal = fr.iloc[:-n_cal], fr.iloc[-n_cal:]
        fit = fit[fit["t1"] < cal.index.min()]
        model = EnsembleModel(ds.feature_names, seed=cfg["project"]["seed"],
                              n_jobs=cfg.get("compute", {}).get("n_jobs", 2))
        model.fit(fit, fit["y"].to_numpy(), cal, cal["y"].to_numpy(), fit["fwd_ret"].to_numpy())
        joblib.dump(model, out / f"{s}.joblib")
        meta["symbols"][s] = {"train_start": str(fit.index.min()), "train_end": str(cal.index.max()),
                              "n_fit": len(fit), "n_calib": len(cal), "features": ds.feature_names}
        log(f"  modelo {s}: fit={len(fit)} cal={len(cal)}")
    gate = promotion_gate(run_summary) if run_summary else {"passed": False, "reasons": ["sin resumen de validación"]}
    meta["validation_run"] = run_summary.get("run_id") if run_summary else None
    meta["gate"] = gate
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if gate["passed"]:
        (cfg.models_dir / "registry" / "ACTIVE").write_text(version, encoding="utf-8")
        log(f"  PROMOVIDO a activo: {version}")
    else:
        log(f"  Registrado como candidato (no promovido): {gate['reasons']}")
    return out


def promotion_gate(summary: dict) -> dict:
    """Puerta de promoción (pre-registrada). Todas deben cumplirse en la validación walk-forward."""
    st = summary.get("strategies", {})
    reasons = []
    if summary.get("is_smoke"):
        reasons.append("la validación fue una prueba mínima (smoke)")
    ens, bh, ens2 = st.get("ml_ensemble"), st.get("buy_and_hold"), st.get("ml_ensemble_cost2x")
    if not ens or not bh or not ens2:
        reasons.append("faltan estrategias para comparar")
    else:
        if ens["sharpe"] <= bh["sharpe"]:
            reasons.append(f"Sharpe ensemble {ens['sharpe']:.2f} <= comprar y mantener {bh['sharpe']:.2f}")
        if ens2["total_return"] <= 0:
            reasons.append(f"retorno con costes x2 {ens2['total_return']:.1%} <= 0")
        if ens["max_drawdown"] < bh["max_drawdown"]:
            reasons.append("drawdown del ensemble peor que comprar y mantener")
        ci = ens.get("sharpe_ci95") or [np.nan]
        if not (np.isfinite(ci[0]) and ci[0] > 0):
            reasons.append("el IC95 % del Sharpe incluye 0")
    return {"passed": not reasons, "reasons": reasons}

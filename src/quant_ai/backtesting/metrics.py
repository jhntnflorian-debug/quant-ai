"""Métricas de rendimiento financiero y de clasificación, con intervalos por bootstrap de bloques."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, brier_score_loss, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)


def _max_drawdown(eq: np.ndarray) -> float:
    peak = np.maximum.accumulate(eq)
    return float(np.min(eq / peak - 1)) if len(eq) else 0.0


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def sharpe(r: np.ndarray, ann: int) -> float:
    sd = np.std(r, ddof=1)
    return float(np.mean(r) / sd * np.sqrt(ann)) if sd > 0 else 0.0


def sortino(r: np.ndarray, ann: int) -> float:
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    return float(np.mean(r) / downside * np.sqrt(ann)) if downside > 0 else 0.0


def block_bootstrap_ci(r: np.ndarray, stat, block: int = 168, n_boot: int = 500, alpha: float = 0.05,
                       seed: int = 0) -> tuple[float, float]:
    """Bootstrap de bloques circulares: preserva autocorrelación/volatilidad agrupada aproximadamente."""
    rng = np.random.default_rng(seed)
    n = len(r)
    if n < 2 * block:
        return (np.nan, np.nan)
    n_blocks = int(np.ceil(n / block))
    stats = np.empty(n_boot)
    rr = np.concatenate([r, r[:block]])
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        sample = np.concatenate([rr[s:s + block] for s in starts])[:n]
        stats[b] = stat(sample)
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def paired_sharpe_diff_ci(r_a: np.ndarray, r_b: np.ndarray, ann: int, block: int = 168, n_boot: int = 1000,
                          alpha: float = 0.05, seed: int = 0) -> dict:
    """IC por bootstrap de bloques circulares PAREADO de SR(a) - SR(b) (anualizados).

    Usa los mismos bloques temporales para ambas series: conserva su correlación y la dependencia serial
    (el PSR asume observaciones casi independientes y puede exagerar la evidencia)."""
    a, b = np.asarray(r_a, dtype=float), np.asarray(r_b, dtype=float)
    n = min(len(a), len(b))
    a, b = a[-n:], b[-n:]
    if n < 2 * block:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    aa, bb = np.concatenate([a, a[:block]]), np.concatenate([b, b[:block]])
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    offs = np.arange(block)[None, :]
    for k in range(n_boot):
        ix = (rng.integers(0, n, n_blocks)[:, None] + offs).ravel()[:n]
        diffs[k] = sharpe(aa[ix], ann) - sharpe(bb[ix], ann)
    return {"diff": float(sharpe(a, ann) - sharpe(b, ann)), "lo": float(np.quantile(diffs, alpha / 2)),
            "hi": float(np.quantile(diffs, 1 - alpha / 2))}


def probabilistic_sharpe(r: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """PSR (Bailey y López de Prado, 2012): P(SR verdadero > sr_benchmark) dado el SR observado,
    corrigiendo por asimetría y curtosis. SR por periodo (no anualizado)."""
    from scipy.stats import kurtosis, norm, skew

    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 3 or np.std(r, ddof=1) == 0:
        return float("nan")
    sr = np.mean(r) / np.std(r, ddof=1)
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    denom = 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2
    if denom <= 0:
        return float("nan")
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(len(r) - 1) / np.sqrt(denom)))


def deflated_sharpe(r: np.ndarray, trial_srs: np.ndarray, n_trials: int | None = None) -> dict:
    """DSR (Bailey y López de Prado, 2014): PSR contra el Sharpe máximo esperado por azar al probar
    N variantes. ``trial_srs``: SR por periodo de TODAS las variantes probadas (incluida la elegida).
    DSR > 0.95 sugiere que el resultado no se explica solo por la selección entre intentos."""
    from scipy.stats import norm

    trials = np.asarray([s for s in trial_srs if np.isfinite(s)], dtype=float)
    # n_trials permite contar intentos de corridas anteriores (registro acumulado); la varianza de los SR
    # se estima con las variantes disponibles en esta corrida.
    n = max(len(trials), n_trials or 0)
    if n < 2 or len(trials) < 2:
        return {"dsr": probabilistic_sharpe(r, 0.0), "sr0": 0.0, "n_trials": n}
    emc = 0.5772156649015329  # constante de Euler-Mascheroni
    sr0 = np.sqrt(np.var(trials, ddof=1)) * ((1 - emc) * norm.ppf(1 - 1 / n) + emc * norm.ppf(1 - 1 / (n * np.e)))
    return {"dsr": probabilistic_sharpe(r, sr0), "sr0": float(sr0), "n_trials": n}


def performance_metrics(equity: pd.Series, trades: pd.DataFrame | None = None, fills: pd.DataFrame | None = None,
                        gross_exposure: pd.Series | None = None, ann: int = 8760, ci: bool = True) -> dict:
    eq = equity.dropna().to_numpy(dtype=float)
    if len(eq) < 2:
        return {}
    r = np.diff(eq) / eq[:-1]
    years = len(r) / ann
    total = eq[-1] / eq[0] - 1
    m = {
        "start": str(equity.index[0]), "end": str(equity.index[-1]), "bars": int(len(eq)),
        "total_return": float(total),
        "cagr": float((eq[-1] / eq[0]) ** (1 / years) - 1) if years > 0 and eq[-1] > 0 else np.nan,
        "ann_volatility": float(np.std(r, ddof=1) * np.sqrt(ann)),
        "sharpe": sharpe(r, ann), "sortino": sortino(r, ann),
        "max_drawdown": _max_drawdown(eq),
    }
    m["calmar"] = float(m["cagr"] / abs(m["max_drawdown"])) if m["max_drawdown"] < 0 else np.nan
    if ci:
        lo, hi = block_bootstrap_ci(r, lambda x: sharpe(x, ann))
        m["sharpe_ci95"] = [lo, hi]
    if gross_exposure is not None:
        ge = gross_exposure.reindex(equity.index).fillna(0).to_numpy()
        m["exposure_time"] = float(np.mean(ge > 1e-6))
        m["avg_gross_exposure"] = float(np.mean(ge))
    if fills is not None and len(fills):
        avg_eq = float(np.mean(eq))
        m["n_fills"] = int(len(fills))
        m["turnover_annual"] = float(fills["notional"].sum() / avg_eq / years) if years > 0 else np.nan
        m["fees_paid"] = float(fills["fee"].sum())
        m["impact_paid"] = float(fills["impact_cost"].sum())
        m["costs_pct_initial"] = float((m["fees_paid"] + m["impact_paid"]) / eq[0])
        m["partial_fills"] = int((fills["qty"] < fills["requested_qty"] - 1e-9).sum())
    if trades is not None and len(trades):
        pnl = trades["pnl"].to_numpy()
        wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
        m["n_trades"] = int(len(pnl))
        m["win_rate"] = float(len(wins) / len(pnl))
        m["avg_win"] = float(wins.mean()) if len(wins) else 0.0
        m["avg_loss"] = float(losses.mean()) if len(losses) else 0.0
        m["payoff_ratio"] = float(m["avg_win"] / abs(m["avg_loss"])) if m["avg_loss"] < 0 else np.nan
        m["profit_factor"] = float(wins.sum() / abs(losses.sum())) if losses.sum() < 0 else np.nan
        m["expectancy"] = float(pnl.mean())
        m["expectancy_pct"] = float(trades["return"].mean())
        m["avg_bars_held"] = float(trades["bars_held"].mean())
    else:
        m["n_trades"] = 0
    return m


def period_stability(equity: pd.Series, freq: str = "YE") -> pd.DataFrame:
    """Retorno por periodo (año por defecto) para medir estabilidad entre periodos."""
    e = equity.resample(freq).last()
    first = equity.resample(freq).first()
    return pd.DataFrame({"return": e / first - 1})


def classification_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    prob = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    pred = (prob >= threshold).astype(int)
    out = {
        "n": int(len(y_true)), "base_rate": float(y_true.mean()),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "brier": float(brier_score_loss(y_true, prob)),
        "brier_baseline": float(brier_score_loss(y_true, np.full_like(prob, y_true.mean()))),
        "log_loss": float(log_loss(y_true, prob, labels=[0, 1])),
    }
    out["auc"] = float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else np.nan
    return out


def calibration_table(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y_true, dtype=float), "p": np.asarray(prob, dtype=float)})
    df["bin"] = pd.cut(df["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    t = df.groupby("bin", observed=True).agg(mean_pred=("p", "mean"), frac_pos=("y", "mean"), n=("y", "size"))
    return t.reset_index().assign(bin=lambda x: x["bin"].astype(str))

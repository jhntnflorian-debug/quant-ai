"""Modelos de ML tradicional con probabilidades calibradas.

- Regresión logística (con StandardScaler ajustado SOLO en el tramo de ajuste del fold).
- LightGBM clasificador (regularizado) + LightGBM regresor del retorno futuro (magnitud).
- Calibración Platt (sigmoide) sobre un tramo de calibración cronológicamente posterior al ajuste.
- Incertidumbre = dispersión entre miembros del ensemble (no es un intervalo de confianza formal).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", message="X does not have valid feature names")


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@dataclass
class PlattCalibrator:
    a: float = 1.0
    b: float = 0.0

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        if len(np.unique(y)) < 2:
            return self
        lr = LogisticRegression(C=1e6, max_iter=1000)
        lr.fit(_logit(p_raw).reshape(-1, 1), y.astype(int))
        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(self.a * _logit(p_raw) + self.b)))


def make_logistic(seed: int = 42, missing_indicator: bool = False):
    # missing_indicator: añade columnas binarias "dato faltante" (útil cuando una fuente externa aún no existía)
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=missing_indicator, keep_empty_features=True), StandardScaler(),
                         LogisticRegression(C=0.05, max_iter=2000, random_state=seed))


def make_lgbm_clf(seed: int = 42, n_jobs: int = 2):
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=300,
                              subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                              random_state=seed, verbose=-1, n_jobs=n_jobs)


def make_lgbm_reg(seed: int = 42, n_jobs: int = 2):
    return lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=15, min_child_samples=300,
                             subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                             random_state=seed, verbose=-1, n_jobs=n_jobs)


@dataclass
class EnsembleModel:
    """Ensemble LR + LGBM con calibración por miembro y regresor de magnitud."""

    features: list[str]
    seed: int = 42
    n_jobs: int = 2
    missing_indicator: bool = False
    members: dict = field(default_factory=dict)
    calibrators: dict = field(default_factory=dict)
    regressor: object = None
    feature_stats: dict = field(default_factory=dict)

    def fit(self, X_fit: pd.DataFrame, y_fit: np.ndarray, X_cal: pd.DataFrame, y_cal: np.ndarray,
            fwd_ret_fit: np.ndarray | None = None) -> "EnsembleModel":
        Xf = X_fit[self.features]
        self.members = {"logistic": make_logistic(self.seed, self.missing_indicator),
                        "lightgbm": make_lgbm_clf(self.seed, self.n_jobs)}
        for name, m in self.members.items():
            m.fit(Xf, y_fit.astype(int))
            p_cal = m.predict_proba(X_cal[self.features])[:, 1]
            self.calibrators[name] = PlattCalibrator().fit(p_cal, y_cal)
        if fwd_ret_fit is not None:
            self.regressor = make_lgbm_reg(self.seed, self.n_jobs)
            self.regressor.fit(Xf, np.clip(fwd_ret_fit, -0.2, 0.2))
        # estadísticos para detección de fuera de distribución (OOD) en inferencia
        self.feature_stats = {"mean": Xf.mean().to_dict(), "std": Xf.std().replace(0, np.nan).to_dict(),
                              "n_train": int(len(Xf))}
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        Xs = X[self.features]
        out = pd.DataFrame(index=X.index)
        probs = []
        for name, m in self.members.items():
            p = self.calibrators[name].transform(m.predict_proba(Xs)[:, 1])
            out[f"p_{name}"] = p
            probs.append(p)
        P = np.vstack(probs)
        out["p_ensemble"] = P.mean(axis=0)
        out["uncertainty"] = P.std(axis=0)
        if self.regressor is not None:
            out["exp_ret"] = self.regressor.predict(Xs)
        return out

    def feature_importance(self) -> pd.DataFrame:
        lgbm = self.members["lightgbm"]
        imp = pd.DataFrame({"feature": self.features,
                            "lgbm_gain": lgbm.booster_.feature_importance("gain")})
        lr = self.members["logistic"][-1]
        imp["logistic_coef"] = lr.coef_[0][:len(self.features)]  # sin las columnas indicadoras
        imp["lgbm_gain_pct"] = imp["lgbm_gain"] / imp["lgbm_gain"].sum()
        return imp.sort_values("lgbm_gain", ascending=False).reset_index(drop=True)

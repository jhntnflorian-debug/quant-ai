import numpy as np
import pandas as pd
import pytest

from quant_ai.validation.splits import (
    HoldoutViolation,
    guard_holdout,
    purged_walk_forward,
    restrict_to_development,
    split_fit_validation,
)


def _index_t1(n=24 * 365 * 3, h=12):
    idx = pd.date_range("2019-01-01", periods=n, freq="1h", tz="UTC")
    t1 = pd.Series(idx + pd.Timedelta(hours=h + 1), index=idx)
    return idx, t1


def test_walk_forward_purges_overlapping_labels():
    idx, t1 = _index_t1()
    folds = purged_walk_forward(idx, t1, "2020-07-01", 3, 12, pd.Timedelta(hours=24), 0.2)
    assert len(folds) == 6  # trimestres de 2020-07 a 2021-10 (los datos acaban el 2021-12-31)
    for f in folds:
        train_t1 = t1.iloc[np.r_[f.train_idx, f.calib_idx]]
        assert (train_t1 < f.test_start - pd.Timedelta(hours=24)).all(), "etiqueta de train cruza el test"
        assert idx[f.test_idx].min() >= f.test_start
        assert idx[f.test_idx].max() < f.test_end
        # el tramo de ajuste no puede resolverse dentro de calibración
        assert t1.iloc[f.train_idx].max() < idx[f.calib_idx].min()
        assert idx[f.train_idx].max() < idx[f.calib_idx].min()


def test_folds_are_chronological_and_disjoint():
    idx, t1 = _index_t1()
    folds = purged_walk_forward(idx, t1, "2020-07-01", 3, 12, pd.Timedelta(hours=24), 0.2)
    tests = [set(f.test_idx) for f in folds]
    for a in range(len(tests)):
        for b in range(a + 1, len(tests)):
            assert not tests[a] & tests[b]
    starts = [f.test_start for f in folds]
    assert starts == sorted(starts)


def test_holdout_guard():
    idx, t1 = _index_t1()
    hs = pd.Timestamp("2021-06-01", tz="UTC")
    with pytest.raises(HoldoutViolation):
        guard_holdout(idx, hs)
    guard_holdout(idx, hs, final=True)
    df = pd.DataFrame({"t1": t1}, index=idx)
    dev = restrict_to_development(df, hs)
    assert dev.index.max() < hs and dev["t1"].max() < hs
    guard_holdout(dev.index, hs)


def test_split_fit_validation_reproduce_la_particion_de_la_cnn():
    """Misma partición que el código que tenía build_chart_folds antes de extraer la función."""
    idx, t1 = _index_t1(h=24)
    embargo = pd.Timedelta(hours=24)
    for f in purged_walk_forward(idx, t1, "2020-07-01", 3, 12, embargo, 0.2):
        train = f.train_idx
        n_val = max(1, int(len(train) * .2))
        val_old, fit_old = train[-n_val:], train[:-n_val]
        fit_old = fit_old[(t1.iloc[fit_old] < idx[val_old[0]] - embargo).to_numpy()]
        fit, val = split_fit_validation(train, idx, t1, embargo, .2)
        np.testing.assert_array_equal(fit, fit_old)
        np.testing.assert_array_equal(val, val_old)


def test_split_fit_validation_cronologica_disjunta_y_purgada_horizonte_24():
    idx, t1 = _index_t1(h=24)   # etiqueta a 24 h: se resuelve en t + 25 h
    embargo = pd.Timedelta(hours=24)
    folds = purged_walk_forward(idx, t1, "2020-07-01", 3, 12, embargo, 0.2)
    assert folds
    for f in folds:
        fit, val = split_fit_validation(f.train_idx, idx, t1, embargo)
        parts = [fit, val, f.calib_idx, f.test_idx]
        for a, part in enumerate(parts):
            assert len(part) and (np.diff(part) > 0).all()
            for other in parts[a + 1:]:
                assert not set(part) & set(other)
                assert idx[part].max() < idx[other].min()     # ajuste < validación < calibración < test
        assert len(val) == int(len(f.train_idx) * .2)
        assert (t1.iloc[fit] < idx[val[0]] - embargo).all()   # purga respecto a validación
        assert (t1.iloc[val] < idx[f.calib_idx[0]]).all()     # validación purgada respecto a calibración
        assert (t1.iloc[f.calib_idx] < f.test_start - embargo).all()
        # Con datos horarios continuos se purgan exactamente las 49 h previas a la validación.
        assert len(f.train_idx) - len(val) - len(fit) == 49
    with pytest.raises(ValueError):
        split_fit_validation(np.array([3]), idx, t1, embargo)

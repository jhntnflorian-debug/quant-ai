"""Pruebas pequeñas de secuencias; requieren el torch disponible en Colab."""
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from quant_ai.models.sequence import build_model, CausalConv1d
from quant_ai.training.sequence_data import WindowDataset, build_sequence_folds, fit_standardizer, standardize
from quant_ai.training.trainer import SequenceTrainer, seed_everything


def test_ventana_causal_sin_copias():
    index = pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC")
    values = np.arange(40, dtype=np.float32).reshape(20, 2)
    ds = WindowDataset(values, np.zeros(20), index, [8], window=4)
    before = ds[0][0].clone()
    values[9:] = -100
    assert torch.equal(before, ds[0][0])
    np.testing.assert_array_equal(before.numpy(), values[5:9])
    assert np.shares_memory(ds.values, values)


def test_huecos_y_normalizacion_solo_ajuste():
    index = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC").delete(4)
    values = np.arange(18, dtype=np.float32).reshape(9, 2)
    values[0, 1] = np.nan
    stats = fit_standardizer(values, np.array([0, 1, 2]))
    changed = values.copy()
    changed[3:] = 9999
    assert stats == fit_standardizer(changed, np.array([0, 1, 2]))
    assert standardize(values, stats)[0, 1] == 0
    ds = WindowDataset(values, np.zeros(9), index, range(9), window=3)
    assert ds.positions.tolist() == [2, 3, 6, 7, 8]


@pytest.mark.parametrize("name", ["gru", "tcn", "transformer"])
def test_shapes(name):
    model = build_model(name, 5, hidden=8, layers=2, dropout=0, window=7)
    assert model(torch.randn(3, 7, 5)).shape == (3,)
    assert model(torch.randn(1, 7, 5)).shape == (1,)


def test_convolucion_causal():
    conv = CausalConv1d(2, 3, kernel_size=3, dilation=2)
    x = torch.randn(2, 2, 12)
    before = conv(x).detach()
    x[:, :, 7:] = 100
    torch.testing.assert_close(before[:, :, :7], conv(x)[:, :, :7])


def test_folds_conservan_velas_sin_etiqueta_y_purga(tmp_path):
    from quant_ai.config import Config
    from quant_ai.training.dataset import SymbolDataset

    index = pd.date_range("2023-01-01", "2023-05-31 23:00", freq="h", tz="UTC", name="open_time")
    values = pd.DataFrame({"feature": np.arange(len(index), dtype=float)}, index=index)
    labels = pd.DataFrame({"y": np.arange(len(index)) % 2, "fwd_ret": 0.01,
                           "t1": index + pd.Timedelta(hours=12)}, index=index)
    missing = index[-30]
    labels.loc[missing, ["y", "fwd_ret", "t1"]] = [np.nan, np.nan, pd.NaT]
    cfg = Config({"market": {"interval": "1h"}, "splits": {
        "final_holdout_start": "2023-06-01", "embargo_bars": 24,
        "walk_forward": {"first_test_start": "2023-03-01", "test_months": 1,
                         "min_train_months": 1, "calibration_fraction": 0.2}}}, tmp_path)
    ds = SymbolDataset("BTCUSDT", pd.DataFrame(), values, labels, ["feature"])
    all_folds = list(build_sequence_folds(cfg, ds, window=8))
    last = list(build_sequence_folds(cfg, ds, window=8, last_folds=1))[0]
    assert last.fold.k == all_folds[-1].fold.k
    assert missing in last.test.times
    assert last.test.times.max() == index[-1]
    assert labels.loc[last.fit.times, "t1"].max() < last.calibration.times.min()
    assert labels.loc[last.calibration.times, "t1"].max() < last.fold.test_start - pd.Timedelta(hours=24)
    frame = ds.frame()
    frame = frame.loc[frame.y.notna() & (frame.t1 < cfg.final_holdout_start)]
    np.testing.assert_allclose(last.normalization["mean"], frame.iloc[last.fold.train_idx][["feature"]].mean())


def test_validacion_separada_de_calibracion(tmp_path):
    from quant_ai.config import Config
    from quant_ai.training.dataset import SymbolDataset
    from quant_ai.validation.splits import split_fit_validation

    index = pd.date_range("2023-01-01", "2023-05-31 23:00", freq="h", tz="UTC", name="open_time")
    values = pd.DataFrame({"feature": np.arange(len(index), dtype=float)}, index=index)
    # Etiqueta a 24 h: se resuelve en t + 25 h.
    labels = pd.DataFrame({"y": np.arange(len(index)) % 2, "fwd_ret": 0.01,
                           "t1": index + pd.Timedelta(hours=25)}, index=index)
    cfg = Config({"market": {"interval": "1h"}, "splits": {
        "final_holdout_start": "2023-06-01", "embargo_bars": 24,
        "walk_forward": {"first_test_start": "2023-03-01", "test_months": 1,
                         "min_train_months": 1, "calibration_fraction": 0.2}}}, tmp_path)
    ds = SymbolDataset("BTCUSDT", pd.DataFrame(), values, labels, ["feature"])
    v1 = list(build_sequence_folds(cfg, ds, window=8))
    v2 = list(build_sequence_folds(cfg, ds, window=8, separate_validation=True))
    assert len(v1) == len(v2) > 0
    embargo = pd.Timedelta(hours=24)
    frame = ds.frame()
    frame = frame.loc[frame.y.notna() & (frame.t1 < cfg.final_holdout_start)]
    for a, b in zip(v1, v2):
        assert a.validation is a.calibration                  # sin la bandera, todo como en v1
        assert b.validation is not b.calibration
        assert a.calibration.positions.tolist() == b.calibration.positions.tolist()
        assert a.test.positions.tolist() == b.test.positions.tolist()
        assert b.fit.times.max() < b.validation.times.min() < b.validation.times.max() < b.calibration.times.min()
        assert labels.loc[b.fit.times, "t1"].max() < b.validation.times.min() - embargo
        assert labels.loc[b.validation.times, "t1"].max() < b.calibration.times.min()
        assert len(b.fit) < len(a.fit)
        # El estandarizador se ajusta solo con el ajuste resultante, no con la validación.
        fit_idx, _ = split_fit_validation(b.fold.train_idx, frame.index, frame.t1, embargo)
        np.testing.assert_allclose(b.normalization["mean"], frame.iloc[fit_idx][["feature"]].mean())
    smoke = list(build_sequence_folds(cfg, ds, window=8, smoke=True, separate_validation=True))
    assert all(0 < len(f.validation) <= 512 and len(f.fit) <= 2048 for f in smoke)


def test_acumulacion_grupo_incompleto_equivale_a_batch(tmp_path):
    seed_everything(7)
    index = pd.date_range("2023-01-01", periods=7, freq="h", tz="UTC")
    x = np.arange(14, dtype=np.float32).reshape(7, 2) / 14
    ds = WindowDataset(x, np.arange(7) % 2, index, range(7), 1)
    original = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(2, 1), torch.nn.Flatten(0))
    state = {k: v.clone() for k, v in original.state_dict().items()}
    models = []
    for batch, accum in [(4, 1), (2, 2)]:
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(2, 1), torch.nn.Flatten(0))
        model.load_state_dict(state)
        trainer = SequenceTrainer(model, tmp_path / str(batch), model_name="test", hyperparams={},
                                  epochs=2, batch_size=batch, accum=accum, num_workers=0, device="cpu")
        trainer.prepare(len(ds))
        trainer._train_epoch(trainer.loader(ds))
        models.append(model)
    for a, b in zip(models[0].parameters(), models[1].parameters()):
        torch.testing.assert_close(a, b)


def synthetic():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(256, 2)).astype("float32")
    y = (x[:, 0] > 0).astype("float32")
    index = pd.date_range("2024-01-01", periods=256, freq="h", tz="UTC")
    return (WindowDataset(x, y, index, range(192), 1),
            WindowDataset(x, y, index, range(192, 256), 1))


def make_trainer(out, epochs=12):
    return SequenceTrainer(build_model("gru", 2, hidden=8, layers=1, dropout=0),
                           out, model_name="gru", hyperparams={"hidden": 8},
                           epochs=epochs, batch_size=32, lr=0.03, patience=epochs,
                           num_workers=0, device="cpu", seed=42)


def test_smoke_reduce_perdida(tmp_path):
    seed_everything(42)
    train, val = synthetic()
    trainer = make_trainer(tmp_path)
    trainer.fit(train, val, normalization={})
    metrics = pd.read_csv(tmp_path / "metrics.csv")
    assert metrics.val_loss.min() < metrics.val_loss.iloc[0] * 0.8
    assert json.loads((tmp_path / "run_info.json").read_text())["status"] == "finished"


def test_checkpoint_resume_restaura_epoca_pesos_y_rng(tmp_path):
    seed_everything(42)
    train, val = synthetic()
    trainer = make_trainer(tmp_path, epochs=3)
    # Una interrupción al comenzar la segunda época deja una época completa persistida.
    original = trainer._train_epoch
    count = 0

    def interrupt(loader):
        nonlocal count
        count += 1
        if count == 2:
            raise KeyboardInterrupt
        return original(loader)

    trainer._train_epoch = interrupt
    with pytest.raises(KeyboardInterrupt):
        trainer.fit(train, val, normalization={})
    checkpoint = torch.load(tmp_path / "checkpoints/last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 1
    resumed = make_trainer(tmp_path, epochs=3)
    resumed.prepare(len(train))
    state = resumed.load_checkpoint(tmp_path / "checkpoints/last.pt")
    assert state["epoch"] == resumed.epoch == 1
    for name, value in resumed.model.state_dict().items():
        torch.testing.assert_close(value, checkpoint["model"][name])
    assert torch.equal(torch.get_rng_state(), checkpoint["rng"]["torch"])
    resumed.fit(train, val, normalization={}, resume=True)
    assert pd.read_csv(tmp_path / "metrics.csv").epoch.tolist() == [1, 2, 3]
    # La reanudación debe coincidir con la trayectoria ininterrumpida en CPU.
    seed_everything(42)
    direct = make_trainer(tmp_path / "direct", epochs=3)
    direct.fit(train, val, normalization={})
    for name, value in direct.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[name])


def test_resume_rechaza_cambios(tmp_path):
    seed_everything(42)
    train, val = synthetic()
    trainer = make_trainer(tmp_path, epochs=1)
    trainer.fit(train, val, normalization={})
    with pytest.raises(ValueError, match="configuración"):
        make_trainer(tmp_path, epochs=2).fit(train, val, normalization={}, resume=True)
    train.values[0, 0] += 10
    with pytest.raises(ValueError, match="datos cambiaron"):
        make_trainer(tmp_path, epochs=1).fit(train, val, normalization={}, resume=True)

"""Pruebas del renderizado y de la CNN opcional."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from quant_ai.training.chart_images import attach_image_cache, prerender_windows, render_window


def test_render_causal_shape_and_high():
    c = np.arange(1,101,dtype=float)
    def image():
        return render_window(c[:48],c[:48]+1,c[:48]-1,c[:48],c[:48])
    before = image()
    c[48:] = 9999
    np.testing.assert_array_equal(before,image())
    assert before.shape == (1,80,144)
    assert before.dtype == np.uint8
    assert before[0,0,142] == 255


def test_constant_zero_volume():
    image = render_window(*([np.ones(48)]*4),np.zeros(48))
    assert not image[:,64:].any()


@pytest.mark.parametrize("window",[48,96])
def test_model_logits(window):
    torch = pytest.importorskip("torch")
    from quant_ai.models.chart_cnn import ChartCNN
    model = ChartCNN(window).eval()
    assert sum(p.numel() for p in model.parameters()) < 1000000
    with torch.no_grad():
        assert model(torch.zeros(2,1,80,window*3)).shape == (2,)


def test_synthetic_training_reduces_loss():
    torch = pytest.importorskip("torch")
    from quant_ai.models.chart_cnn import ChartCNN
    torch.manual_seed(42)
    model = ChartCNN()
    x = torch.zeros(8,1,80,144)
    x[4:] = 1
    y = torch.tensor([0.]*4+[1.]*4)
    optimizer = torch.optim.Adam(model.parameters(),lr=.003)
    criterion = torch.nn.BCEWithLogitsLoss()
    model.train()
    losses = []
    for _ in range(12):
        optimizer.zero_grad()
        loss = criterion(model(x),y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    assert np.mean(losses[-3:]) < np.mean(losses[:3])


def test_dataset_excludes_gaps_and_future():
    pytest.importorskip("torch")
    import pandas as pd
    from quant_ai.training.chart_images import ChartDataset
    index = pd.date_range("2021-01-01",periods=120,freq="h",tz="UTC").delete(60)
    values = np.ones((119,5),dtype=np.float32)
    ds = ChartDataset(values,np.zeros(119),index,range(119),48)
    assert 59 in ds.positions
    assert 60 not in ds.positions
    assert 106 not in ds.positions
    assert 107 in ds.positions
    before = ds[0][0].clone()
    ds.values[48:] *= 2
    assert (before == ds[0][0]).all()


def _ohlcv(n, seed=0):
    """Velas válidas en float32, como las que recibe ChartDataset."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, .01, n)))
    open_ = np.r_[100, close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, .005, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, .005, n))
    volume = rng.uniform(0, 200, n)
    volume[rng.random(n) < .05] = 0
    return np.column_stack([open_, high, low, close, volume]).astype(np.float32)


def _assert_cache_matches(values, positions, rows, images, window=48):
    for end in positions:
        expected = render_window(*values[end-window+1:end+1].T)
        assert images[rows[end]].dtype == np.uint8
        np.testing.assert_array_equal(images[rows[end]], expected)


def test_prerender_identico_a_render_y_serie_igual_a_paralelo():
    values = _ohlcv(1500)
    rng = np.random.default_rng(1)
    positions = rng.choice(np.arange(47, 1500), 400, replace=False)
    rows, images = prerender_windows(values, np.r_[positions, positions[:50]], 48, workers=1, chunk=64)
    assert images.shape == (400, 1, 80, 144) and rows.shape == (1500,)
    assert (rows[np.setdiff1d(np.arange(1500), positions)] == -1).all()
    _assert_cache_matches(values, positions, rows, images)
    rows2, images2 = prerender_windows(values, positions, 48, workers=2, chunk=64)
    np.testing.assert_array_equal(rows, rows2)
    np.testing.assert_array_equal(images, images2)
    with pytest.raises(ValueError):
        prerender_windows(values, [10], 48)


def test_prerender_identico_en_datos_reales():
    from quant_ai.config import load_config
    path = load_config().data_dir / "clean" / "BTCUSDT_1h.parquet"
    if not path.is_file():
        pytest.skip("Sin data/clean/BTCUSDT_1h.parquet")
    prices = pd.read_parquet(path).sort_values("open_time").iloc[30000:33000]
    values = prices[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float32)
    positions = np.random.default_rng(2).choice(np.arange(47, len(values)), 300, replace=False)
    rows, images = prerender_windows(values, positions, 48, workers=1, chunk=100)
    _assert_cache_matches(values, positions, rows, images)


def test_cache_compartida_entre_tramos():
    values = _ohlcv(400)
    parts = [SimpleNamespace(values=values, window=48, positions=np.arange(a, b)) for a, b in
             [(47, 200), (150, 260), (300, 400)]]
    n, nbytes = attach_image_cache(parts)
    assert n == 153 + 60 + 100 and nbytes == n * 80 * 144 + 400 * 4
    assert all(p.images is parts[0].images and p.rows is parts[0].rows for p in parts)
    _assert_cache_matches(values, np.r_[47:260, 300:400], parts[0].rows, parts[0].images)
    with pytest.raises(ValueError):
        attach_image_cache(parts + [SimpleNamespace(values=values.copy(), window=48, positions=np.arange(60, 70))])


def test_dataset_con_cache_devuelve_lo_mismo():
    torch = pytest.importorskip("torch")
    from quant_ai.training.chart_images import ChartDataset
    index = pd.date_range("2021-01-01", periods=300, freq="h", tz="UTC").delete(120)
    values = _ohlcv(299)
    labels = (np.arange(299) % 2).astype(np.float32)
    lazy = ChartDataset(values, labels, index, range(299), 48)
    cached = [ChartDataset(values, labels, index, p, 48) for p in (range(150), range(100, 299))]
    attach_image_cache(cached)
    assert lazy.rows is None
    for ds in cached:
        for item in range(len(ds)):
            x, y = ds[item]
            x0, y0 = lazy[int(np.flatnonzero(lazy.positions == ds.positions[item])[0])]
            assert torch.equal(x, x0) and torch.equal(y, y0)


def test_folds_cnn_cronologicos_purgados_y_cache_identica(tmp_path):
    torch = pytest.importorskip("torch")
    from conftest import make_ohlcv
    from quant_ai.config import Config
    from quant_ai.training.chart_images import ChartDataset, build_chart_folds
    prices = make_ohlcv(24 * 200, seed=5, start="2023-01-01")
    cfg = Config({"market": {"interval": "1h"}, "splits": {
        "final_holdout_start": "2023-08-01", "embargo_bars": 24,
        "walk_forward": {"first_test_start": "2023-04-01", "min_train_months": 1,
                         "calibration_fraction": 0.2}}}, tmp_path)
    ds = SimpleNamespace(symbol="BTCUSDT", prices=prices)
    folds = list(build_chart_folds(cfg, ds, window=48, last_folds=2))
    assert len(folds) == 2
    embargo = pd.Timedelta(hours=24)
    for f in folds:
        parts = [f.fit, f.validation, f.calibration, f.test]
        for a, b in zip(parts, parts[1:]):
            assert a.times.max() < b.times.min()
        # etiqueta a 24 h sobre velas continuas: se resuelve en t + 25 h
        assert (f.fit.times + pd.Timedelta(hours=25)).max() < f.validation.times.min() - embargo
        assert (f.calibration.times + pd.Timedelta(hours=25)).max() < f.fold.test_start - embargo
    n, _ = attach_image_cache(p for f in folds for p in (f.fit, f.validation, f.calibration, f.test))
    assert n == len(np.unique(np.concatenate([p.positions for f in folds
                                              for p in (f.fit, f.validation, f.calibration, f.test)])))
    for part in (folds[0].validation, folds[1].test):
        lazy = ChartDataset(part.values, part.labels, part.index, part.positions, 48)
        assert lazy.rows is None and part.rows is not None
        for item in np.linspace(0, len(part) - 1, 25, dtype=int):
            assert torch.equal(part[item][0], lazy[item][0])
    smoke = next(build_chart_folds(cfg, ds, window=48, last_folds=1, smoke=True))
    assert all(0 < len(p) <= 512 for p in (smoke.fit, smoke.validation, smoke.calibration, smoke.test))

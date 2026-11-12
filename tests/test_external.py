import json
import pandas as pd
import pytest
from quant_ai.data.external import parse_source
from quant_ai.features.external import align_point_in_time, build_external_features


def test_formats():
    c = parse_source('coinbase', json.loads('[[1609459200,1,5,2,4,9]]'))
    assert c.iloc[0]['open']==2 and c.iloc[0]['close']==4
    assert c.available_at.iloc[0]==pd.Timestamp('2021-01-01T01:00Z')
    f = parse_source('fred', 'DATE,VIXCLS\n2021-01-01,.\n2021-01-02,20\n', 'VIXCLS')
    assert pd.isna(f.value.iloc[0])
    assert f.available_at.iloc[0]==pd.Timestamp('2021-01-03T00:00Z')
    d = parse_source('funding', 'calc_time,funding_interval_hours,last_funding_rate\n1609459200000,8,0.001\n')
    assert d.available_at.iloc[0]==pd.Timestamp('2021-01-01T00:01Z')
    m = parse_source('metrics', 'create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio\n2021-01-01 00:00:00,BTCUSDT,1,2,3,4,5,6\n')
    assert m.available_at.iloc[0]==pd.Timestamp('2021-01-01T00:05Z')


@pytest.mark.parametrize('source,payload,symbol,delay', [
    ('deribit', {'result':{'data':[[1609459200000,1,2,0,1]]}}, None, '1h'),
    ('fng', {'data':[{'timestamp':'1609459200','value':'50','value_classification':'Neutral'}]}, None, '2h'),
    ('stable', [{'date':1609459200,'totalCirculatingUSD':{'peggedUSD':42}}], None, '25h'),
    ('onchain', {'data':[{'time':'2021-01-01T00:00:00Z','asset':'btc','AdrActCnt':'10','TxCnt':'20'}]}, None, '26h'),
    ('fred', 'DATE,DTWEXBGS\n2021-01-01,100\n', 'DTWEXBGS', '8d'),
    *[(s, [[1609459200000,1,2,0,1,10,1609462799999,10,1,5,5,0]], None, '1h') for s in ['perp','premium','alts']],
])
def test_other_formats(source,payload,symbol,delay):
    d = parse_source(source,payload,symbol)
    assert d.available_at.iloc[0]-d.time.iloc[0]==pd.Timedelta(delay)


def test_causality_and_partial(tmp_path):
    folder = tmp_path/'external'
    folder.mkdir()
    d = parse_source('funding', 'calc_time,funding_interval_hours,last_funding_rate\n1609459200000,8,0.001\n1609488000000,8,0.002\n')
    path = folder/'funding_BTCUSDT.parquet'
    d.to_parquet(path)
    idx = pd.date_range('2020-12-31T23:00Z', periods=12, freq='h')
    with pytest.warns(UserWarning):
        before = build_external_features(tmp_path, 'BTCUSDT', idx)
    assert pd.isna(before.ext_funding_rate.iloc[0])
    assert before.ext_funding_rate.iloc[1]==0.001
    cutoff = pd.Timestamp('2021-01-01T08:00Z')
    d.loc[d.available_at>cutoff,'last_funding_rate']=999
    d.to_parquet(path)
    with pytest.warns(UserWarning):
        after = build_external_features(tmp_path, 'BTCUSDT', idx)
    pd.testing.assert_frame_equal(before.loc[idx+pd.Timedelta('1h')<=cutoff], after.loc[idx+pd.Timedelta('1h')<=cutoff])
    a = align_point_in_time(d, idx)
    assert (a.available_at.dropna()<=a.decision_time[a.available_at.notna()]).all()


def test_manifest_resume_and_closed_only(tmp_path):
    from quant_ai.data.external import Downloader
    import hashlib
    d = Downloader(tmp_path, now='2021-01-01T02:00Z')
    frame = parse_source('coinbase', [[1609459200,1,5,2,4,9], [1609466400,1,5,2,4,9]])
    d.save('coinbase','BTCUSDT',[frame],'fixture')
    d.save('coinbase','BTCUSDT',[frame],'fixture')
    path = tmp_path/'external/coinbase_BTCUSDT.parquet'
    assert len(pd.read_parquet(path))==1
    manifest = json.loads((tmp_path/'external/manifest.json').read_text())
    assert manifest[path.name]['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest[path.name]['rows']==1


def test_retry_after(monkeypatch):
    import requests
    from quant_ai.data.external import RetrySession
    responses = []
    for status in [429,418,200]:
        response = requests.Response()
        response.status_code = status
        response.headers['Retry-After']='3'
        responses.append(response)
    monkeypatch.setattr(requests.Session,'get',lambda *a,**kw:responses.pop(0))
    sleeps = []
    monkeypatch.setattr('quant_ai.data.external.time.sleep',sleeps.append)
    assert RetrySession().get('https://fixture.invalid').status_code==200
    assert sleeps==[3,3]


def test_align_handles_mixed_datetime_resolution():
    """pandas 3 produce índices en µs y fuentes en ns: merge_asof exige la misma resolución."""
    import pandas as pd
    from quant_ai.features.external import align_point_in_time

    decisions = pd.date_range("2026-01-01 01:00", periods=3, freq="h", tz="UTC").astype("datetime64[us, UTC]")
    series = pd.DataFrame({"available_at": pd.to_datetime(["2026-01-01 00:30", "2026-01-01 02:00"], utc=True)
                           .astype("datetime64[ns, UTC]"), "v": [1.0, 2.0]})
    out = align_point_in_time(series, decisions)
    assert out["v"].tolist() == [1.0, 2.0, 2.0]

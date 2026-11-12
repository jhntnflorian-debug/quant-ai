"""Características externas causales. Los retrasos son supuestos, no vintages históricos."""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd


def align_point_in_time(series, decision_times):
    times = pd.DatetimeIndex(pd.to_datetime(decision_times, utc=True))
    # misma resolución en ambas claves: pandas 3 mezcla µs y ns y merge_asof exige tipos idénticos
    left = pd.DataFrame({'decision_time':times.astype('datetime64[ns, UTC]'), '_order':range(len(times))})
    right = series.copy()
    right['available_at'] = pd.to_datetime(right.available_at, utc=True).astype('datetime64[ns, UTC]')
    result = pd.merge_asof(left.sort_values('decision_time'), right.sort_values('available_at'), left_on='decision_time', right_on='available_at', direction='backward')
    result = result.sort_values('_order').drop(columns='_order')
    result.index = times
    return result


def _z(s, window):
    r = s.rolling(window)
    return (s-r.mean())/r.std().replace(0, np.nan)


def build_external_features(data_dir, symbol, bar_index):
    root = Path(data_dir)
    original = pd.DatetimeIndex(bar_index)
    idx = pd.DatetimeIndex(pd.to_datetime(bar_index, utc=True))
    if not len(idx):
        return pd.DataFrame(index=original)
    grid = pd.date_range(idx.min(), idx.max(), freq='h')
    decisions = grid+pd.Timedelta(hours=1)
    f = pd.DataFrame(index=grid)
    def read(name, clean=False):
        path = root/('clean' if clean else 'external')/(name+'.parquet')
        if not path.exists():
            warnings.warn(f'Fuente ausente: {path}', UserWarning, stacklevel=2)
            return None
        d = pd.read_parquet(path)
        if clean:
            if 'open_time' not in d:
                d = d.reset_index()
            d['time'] = pd.to_datetime(d.open_time, utc=True)
            d['available_at'] = d.time+pd.Timedelta(hours=1)
        return d.sort_values('available_at').drop_duplicates('available_at', keep='last')
    def aligned(d, exact=False):
        a = align_point_in_time(d, decisions)
        a.index = grid
        if exact:
            cols = a.columns.difference(['available_at','decision_time','time'])
            a.loc[a.time != grid, cols] = np.nan
        return a
    def add(d, cols):
        a = aligned(d)
        for c in cols:
            f['ext_'+c] = a[c]
    spot = read(symbol+'_1h', True)
    s = aligned(spot, True) if spot is not None else None
    funding = read('funding_'+symbol)
    if funding is not None:
        funding['funding_rate'] = funding.last_funding_rate
        funding['funding_mean_3d'] = funding.funding_rate.rolling(9).mean()
        funding = funding.set_index('available_at', drop=False)
        funding['funding_z_30d'] = _z(funding.funding_rate, '30d')
        add(funding.reset_index(drop=True), ['funding_rate','funding_mean_3d','funding_z_30d'])
    premium = read('premium_'+symbol)
    if premium is not None:
        p = aligned(premium, True).close
        f['ext_premium_close'] = p
        f['ext_premium_mean_24h'] = p.rolling(24).mean()
    perp = read('perp_'+symbol)
    if perp is not None:
        p = aligned(perp, True)
        f['ext_perp_taker_buy_ratio_24h'] = p.taker_buy_base.rolling(24).sum()/p.volume.rolling(24).sum()
        if s is not None:
            f['ext_basis_perp_spot'] = p.close/s.close-1
            f['ext_perp_spot_volume_ratio_24h'] = p.volume.rolling(24).sum()/s.volume.rolling(24).sum()
    metrics = read('metrics_'+symbol)
    if metrics is not None:
        m = aligned(metrics)
        oi = np.log(m.sum_open_interest.where(m.sum_open_interest>0))
        f['ext_oi_change_1h'], f['ext_oi_change_24h'] = oi.diff(), oi.diff(24)
        f['ext_oi_value_z_30d'] = _z(m.sum_open_interest_value, 720)
        for raw, name in [('count_toptrader_long_short_ratio','toptrader_ls_ratio'),('count_long_short_ratio','global_ls_ratio'),('sum_taker_long_short_vol_ratio','taker_ls_vol_ratio_24h')]:
            if raw in m:
                f['ext_'+name] = m[raw].rolling(24).mean() if name.endswith('24h') else m[raw]
    cb = read('coinbase_'+symbol)
    if cb is not None and s is not None:
        f['ext_coinbase_premium'] = aligned(cb, True).close/s.close-1
        f['ext_coinbase_premium_mean_24h'] = f.ext_coinbase_premium.rolling(24).mean()
    dv = read('deribit_'+symbol)
    if dv is not None:
        v = aligned(dv, True).close
        f['ext_dvol'], f['ext_dvol_change_24h'] = v, v.diff(24)
        if s is not None:
            f['ext_dvol_minus_realized'] = v/100-np.log(s.close).diff().rolling(24).std()*np.sqrt(365*24)
    for source in ['fng','stable']:
        d = read(source)
        if d is None:
            continue
        d = d.set_index('time', drop=False).asfreq('D')
        if source=='fng':
            d['fng'] = d.value/100
            d['fng_change_7d'] = d.fng.diff(7)
            cols = ['fng','fng_change_7d']
        else:
            cols = ['stablecoin_growth_7d','stablecoin_growth_30d']
            for n,c in zip([7,30],cols):
                d[c] = d.value/d.value.shift(n)-1
        add(d.dropna(subset=['available_at']).reset_index(drop=True), cols)
    for series, name, n in [('VIXCLS','vix',5),('SP500','spx',5),('DGS10','dgs10',5),('DTWEXBGS','dxy',20)]:
        d = read('fred_'+series)
        if d is None:
            continue
        d = d.set_index('time', drop=False).asfreq('D')
        # Calendar-day lag; carry only already observed values across holidays.
        v = d.value.ffill()
        col = f'{name}_ret_{n}d' if name=='spx' else f'{name}_change_{n}d'
        d[col] = v/v.shift(n)-1 if name=='spx' else v.diff(n)
        cols = [col]
        if name=='vix':
            d['vix'] = d.value
            cols.append('vix')
        add(d.dropna(subset=['available_at']).reset_index(drop=True), cols)
    above, returns = [], []
    for alt in ['BNBUSDT','SOLUSDT','XRPUSDT','ADAUSDT','DOGEUSDT']:
        d = read('alts_'+alt)
        if d is not None:
            c = aligned(d, True).close
            ema = c.ewm(span=168, adjust=False, min_periods=168).mean()
            above.append((c>ema).astype(float).where(c.notna() & ema.notna()))
            returns.append(c/c.shift(24)-1)
    if above:
        f['ext_breadth_above_ema168'] = pd.concat(above, axis=1).mean(axis=1)
        btc = s if symbol=='BTCUSDT' else None
        if btc is None:
            b = read('BTCUSDT_1h', True)
            btc = aligned(b, True) if b is not None else None
        if btc is not None:
            f['ext_alt_minus_btc_ret_24h'] = pd.concat(returns, axis=1).mean(axis=1)-(btc.close/btc.close.shift(24)-1)
    d = read('onchain')
    if d is not None:
        # Filter before collapsing dates: BTC and ETH share timestamps.
        d = pd.read_parquet(root/'external/onchain.parquet')
        d = d[d.asset==symbol.replace('USDT','').lower()].sort_values('available_at')
        d = d.set_index('available_at', drop=False)
        for raw, name in [('AdrActCnt','active_addr_z_30d'),('TxCnt','tx_count_z_30d')]:
            d[name] = _z(d[raw], '30d')
        add(d.reset_index(drop=True), ['active_addr_z_30d','tx_count_z_30d'])
    f = f.reindex(idx).replace([np.inf,-np.inf], np.nan)
    f.index = original
    return f


def external_coverage(df):
    return pd.DataFrame({'first_valid':{c:df[c].first_valid_index() for c in df}, 'coverage_pct':df.notna().mean()*100})

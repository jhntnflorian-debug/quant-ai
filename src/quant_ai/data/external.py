"""Fuentes externas; caché original y disponibilidad conservadora en UTC."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
import hashlib
import io
import json
import time
import zipfile
import pandas as pd
import requests
from . import binance

BASE = 'https://data.binance.vision/data'
URLS = dict(coinbase='https://api.exchange.coinbase.com/products', deribit='https://www.deribit.com/api/v2/public/get_volatility_index_data', fng='https://api.alternative.me/fng/?limit=0&format=json', stable='https://stablecoins.llama.fi/stablecoincharts/all', fred='https://fred.stlouisfed.org/graph/fredgraph.csv', onchain='https://community-api.coinmetrics.io/v4/timeseries/asset-metrics')
ALTS = ['BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT']

class RetrySession(requests.Session):
    def get(self, url, **kwargs):
        for attempt in range(6):
            try:
                r = super().get(url, **kwargs)
                if r.status_code not in (418, 429) and r.status_code < 500:
                    return r
                retry = r.headers.get('Retry-After', '')
                try:
                    delay = float(retry)
                except ValueError:
                    delay = max(0, (parsedate_to_datetime(retry).timestamp()-time.time())) if retry else 0
                if attempt == 5:
                    r.raise_for_status()
            except requests.RequestException:
                if attempt == 5:
                    raise
                delay = 0
            time.sleep(max(2 ** attempt, delay))


def parse_source(source, payload, symbol=None):
    """Normaliza originales; nunca estima disponibilidad usando la fecha de descarga."""
    if source in ('perp', 'premium', 'alts'):
        d = binance.parse_klines(payload)
        t, delay = d.open_time, '1h'
    elif source == 'funding':
        d = pd.read_csv(io.StringIO(payload)) if isinstance(payload, str) else pd.DataFrame(payload).rename(columns={'fundingTime':'calc_time', 'fundingRate':'last_funding_rate'})
        t, delay = pd.to_datetime(d.calc_time, unit='ms', utc=True), '1min'
    elif source == 'metrics':
        d = pd.read_csv(io.StringIO(payload)) if isinstance(payload, str) else pd.DataFrame(payload)
        t, delay = pd.to_datetime(d.create_time, utc=True), '5min'
    elif source == 'coinbase':
        d = pd.DataFrame(payload, columns=['time','low','high','open','close','volume'])
        t, delay = pd.to_datetime(d.time, unit='s', utc=True), '1h'
    elif source == 'deribit':
        d = pd.DataFrame(payload['result']['data'], columns=['time','open','high','low','close'])
        t, delay = pd.to_datetime(d.time, unit='ms', utc=True), '1h'
    elif source == 'fng':
        d = pd.DataFrame(payload['data'])
        t, delay = pd.to_datetime(pd.to_numeric(d.timestamp), unit='s', utc=True), '2h'
    elif source == 'stable':
        d = pd.DataFrame([{'date':r['date'], 'value':r['totalCirculatingUSD']['peggedUSD']} for r in payload])
        t, delay = pd.to_datetime(pd.to_numeric(d.date), unit='s', utc=True), '25h'
    elif source == 'fred':
        d = pd.read_csv(io.StringIO(payload), na_values='.')
        t, delay = pd.to_datetime(d.iloc[:,0], utc=True), '8d' if symbol == 'DTWEXBGS' else '2d'
        d = d.rename(columns={d.columns[1]:'value'})
    elif source == 'onchain':
        d = pd.DataFrame(payload['data'])
        t, delay = pd.to_datetime(d.time, utc=True), '26h'
    else:
        raise ValueError(source)
    d['time'] = t
    d['available_at'] = t + pd.Timedelta(delay)
    for c in d.columns.difference(['time','available_at','open_time','close_time','create_time','asset','symbol','value_classification','DATE','observation_date']):
        d[c] = pd.to_numeric(d[c], errors='coerce')
    return d.sort_values('available_at').reset_index(drop=True)


class Downloader:
    def __init__(self, data_dir, max_workers=2, now=None, cache_dir=None):
        # root: Parquet consolidados + manifiesto (en Colab, Drive). cache: miles de originales pequeños
        # (en Colab, disco local /content: escribir miles de archivos en Drive es muy lento).
        self.root = Path(data_dir)/'external'
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache = Path(cache_dir) if cache_dir else self.root
        self.max_workers = max_workers
        self.now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz='UTC')
        self.warnings = []

    def warn(self, msg):
        self.warnings.append(msg)
        print(f'AVISO: {msg}', flush=True)

    def raw(self, source, url, params=None, refresh=False):
        key = hashlib.sha256((url+json.dumps(params, sort_keys=True)).encode()).hexdigest()
        path = self.cache/source/key
        if path.exists() and not refresh:
            return path.read_bytes()
        r = RetrySession().get(url, params=params, timeout=60)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp')
        tmp.write_bytes(r.content)
        tmp.replace(path)
        return r.content

    def save(self, source, symbol, frames, base):
        path = self.root/(source + ('_'+symbol if symbol else '')+'.parquet')
        if path.exists():
            frames = [pd.read_parquet(path)] + frames
        frames = [f for f in frames if not f.empty]
        if not frames:
            return None
        d = pd.concat(frames, ignore_index=True)
        if 'create_time' in d:  # archivos traen texto y la API REST trae Timestamp: unificar tipo
            d['create_time'] = pd.to_datetime(d['create_time'], utc=True)
        keys = ['time'] + (['asset'] if 'asset' in d else [])
        d = d.drop_duplicates(keys, keep='last').sort_values('available_at')
        d = d[d.available_at <= self.now]
        tmp = path.with_suffix('.tmp')
        d.to_parquet(tmp, index=False)
        tmp.replace(path)
        manifest = self.root/'manifest.json'
        m = json.loads(manifest.read_text()) if manifest.exists() else {}
        m[path.name] = dict(base_url=base, rows=len(d), start=str(d.time.min()), end=str(d.time.max()), downloaded_at=str(self.now), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        tmp = manifest.with_suffix('.tmp')
        tmp.write_text(json.dumps(m, indent=2))
        tmp.replace(manifest)
        print(f'{path.name}: {len(d)} filas, {d.time.min()} — {d.time.max()}')
        return d

    def vision(self, source, symbol, since):
        daily = source == 'metrics'
        starts = {'perp':'2019-09-01' if symbol=='BTCUSDT' else '2019-11-01', 'premium':'2019-12-01' if symbol=='BTCUSDT' else '2020-01-01', 'funding':'2020-01-01', 'metrics':'2020-09-01' if symbol=='BTCUSDT' else '2021-09-01', 'alts':'2018-01-01'}
        start = max(pd.Timestamp(since), pd.Timestamp(starts[source]))
        cutoff = self.now.tz_localize(None).normalize() if daily else self.now.tz_localize(None).to_period('M').start_time
        periods = pd.date_range(start.normalize(), cutoff-pd.Timedelta(days=1), freq='D') if daily else pd.period_range(start.to_period('M'), (cutoff-pd.Timedelta(days=1)).to_period('M'), freq='M')
        def one(p):
            period = str(p.date()) if daily else str(p)
            if source == 'alts':
                path, _ = binance.fetch_vision_file(symbol, '1h', period, 'monthly', self.cache/source, RetrySession())
                if path is None:
                    return None
                d = binance.read_vision_zip(path)
                d['time'] = d.open_time
                d['available_at'] = d.open_time+pd.Timedelta(hours=1)
                return d
            kind = dict(perp='klines', premium='premiumIndexKlines', funding='fundingRate', metrics='metrics')[source]
            suffix = f'{symbol}-{kind}-{period}.zip' if source in ('funding','metrics') else f'{symbol}-1h-{period}.zip'
            url = f'{BASE}/futures/um/{"daily" if daily else "monthly"}/{kind}/{symbol}/' + ('' if source in ('funding','metrics') else '1h/') + suffix
            raw = self.raw(source, url)
            if raw is None:
                return None
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                body = z.read(z.namelist()[0]).decode()
            if source in ('perp','premium'):
                body = pd.read_csv(io.StringIO(body), header=0 if body.startswith('open_time') else None)
            return parse_source(source, body)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            frames = [frame for frame in pool.map(one, periods) if frame is not None]
        if frames:  # una sola escritura por fuente (antes: una por archivo, O(n^2) sobre Drive)
            self.save(source, symbol, frames, BASE)
        # El mes/día en curso por REST es opcional: entrenar y evaluar no necesitan nada posterior al
        # 2025-10-31, cubierto por los archivos. Una VM bloqueada por región (HTTP 451/403) solo avisa.
        try:
            self.rest(source, symbol, max(start, cutoff).tz_localize('UTC'))
        except (requests.RequestException, ValueError, TypeError) as e:
            self.warn(f'{source} {symbol}: complemento REST omitido ({type(e).__name__}: {str(e)[:120]})')

    def rest(self, source, symbol, start):
        cursor = max(1, int(start.timestamp()*1000))
        end = int(self.now.timestamp()*1000)
        if source == 'metrics':
            cursor = max(cursor, end-29*86400000)
            endpoints = [('openInterestHist', {'sumOpenInterest':'sum_open_interest','sumOpenInterestValue':'sum_open_interest_value'}), ('topLongShortAccountRatio', {'longShortRatio':'count_toptrader_long_short_ratio'}), ('globalLongShortAccountRatio', {'longShortRatio':'count_long_short_ratio'}), ('takerlongshortRatio', {'buySellRatio':'sum_taker_long_short_vol_ratio'})]
            frames = []
            while cursor < end:
                stop = min(end, cursor+499*300000)
                merged = None
                for endpoint, mapping in endpoints:
                    url = 'https://fapi.binance.com/futures/data/'+endpoint
                    body = self.raw(source, url, dict(symbol=symbol, period='5m', startTime=cursor, endTime=stop, limit=500), True)
                    rows = json.loads(body) if body else []
                    d = pd.DataFrame(rows)
                    if d.empty:
                        continue
                    d = d[['timestamp']+list(mapping)].rename(columns=mapping).set_index('timestamp')
                    merged = d if merged is None else merged.join(d, how='outer')
                if merged is not None:
                    merged['create_time'] = pd.to_datetime(merged.index, unit='ms', utc=True)
                    frames.append(parse_source('metrics', merged.reset_index()))
                cursor = stop+1
            self.save(source, symbol, frames, 'https://fapi.binance.com')
            return
        endpoint = dict(perp='klines', premium='premiumIndexKlines', funding='fundingRate', alts='klines')[source]
        url = ('https://api.binance.com/api/v3/' if source=='alts' else 'https://fapi.binance.com/fapi/v1/')+endpoint
        while cursor < end:
            params = dict(symbol=symbol, startTime=cursor, endTime=end, limit=1000)
            if source != 'funding':
                params['interval'] = '1h'
            body = self.raw(source, url, params, True)
            rows = json.loads(body) if body else []
            if not rows:
                break
            self.save(source, symbol, [parse_source(source, rows)], url)
            last = int(rows[-1]['fundingTime'] if source=='funding' else rows[-1][0])
            if last < cursor:
                raise ValueError('Paginación sin avance')
            cursor = last+1

    def download(self, source, symbol=None, since='2018-01-01'):
        if source in ('perp','premium','funding','metrics','alts'):
            return self.vision(source, symbol, since)
        url = URLS[source]
        start = pd.Timestamp(since, tz='UTC')
        if source in ('coinbase','deribit'):
            path = self.root/f'{source}_{symbol}.parquet'
            if path.exists():
                start = max(start, pd.read_parquet(path).time.max())
            frames = []
            try:
                while start < self.now:
                    stop = min(start+pd.Timedelta(hours=299), self.now)
                    if source=='coinbase':
                        target = url+'/'+symbol.replace('USDT','-USD')+'/candles'
                        params = dict(granularity=3600, start=start.isoformat(), end=stop.isoformat())
                    else:
                        target = url
                        params = dict(currency=symbol.replace('USDT',''), start_timestamp=int(start.timestamp()*1000), end_timestamp=int(stop.timestamp()*1000), resolution=3600)
                    body = self.raw(source, target, params, stop==self.now)
                    if body:
                        frames.append(parse_source(source, json.loads(body)))
                    if len(frames) >= 50:  # guardado por lotes: reanudable sin reescribir en cada ventana
                        self.save(source, symbol, frames, url)
                        frames = []
                    start = stop
            except (requests.RequestException, ValueError, KeyError) as e:
                # p. ej. Deribit restringe EE. UU.: la fuente queda incompleta y las características serán NaN
                self.warn(f'{source} {symbol}: descarga interrumpida ({type(e).__name__}: {str(e)[:120]})')
            if frames:
                self.save(source, symbol, frames, url)
            return
        params = {'id':symbol} if source=='fred' else (dict(assets='btc,eth', metrics='AdrActCnt,TxCnt', frequency='1d', start_time=since, page_size=10000) if source=='onchain' else None)
        while url:
            raw = self.raw(source, url, params, True)
            payload = raw.decode() if source=='fred' else json.loads(raw)
            d = parse_source(source, payload, symbol)
            self.save(source, symbol, [d[d.time>=start]], URLS[source])
            url = payload.get('next_page_url') if source=='onchain' else None
            params = None

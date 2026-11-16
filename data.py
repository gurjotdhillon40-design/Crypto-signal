"""Download and cache daily prices (Coinbase public API) and the Fear & Greed index."""
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"
COINBASE = "https://api.exchange.coinbase.com/products/{pair}/candles"
FNG_URL = "https://api.alternative.me/fng/?limit=0"
EARLIEST = datetime(2015, 1, 1, tzinfo=timezone.utc)
HEADERS = {"User-Agent": "crypto-signals"}


def _get(url, params=None):
    for attempt in range(4):
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            r.raise_for_status()
        time.sleep(2 ** attempt)
    r.raise_for_status()


def fetch_prices(coin, refresh=True):
    """Daily open/high/low/close/volume in USD, completed days only."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"{coin}.csv"
    old = pd.read_csv(path, parse_dates=["date"]) if path.exists() else None
    if old is not None and not refresh:
        return old
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = EARLIEST if old is None else old["date"].max().to_pydatetime().replace(tzinfo=timezone.utc)
    rows = []
    while start < today:
        end = min(start + timedelta(days=299), today)
        rows += _get(COINBASE.format(pair=f"{coin}-USD"),
                     {"granularity": 86400, "start": start.isoformat(), "end": end.isoformat()})
        start = end + timedelta(days=1)
        time.sleep(0.15)
    new = pd.DataFrame(rows, columns=["ts", "low", "high", "open", "close", "volume"])
    new["date"] = pd.to_datetime(new["ts"], unit="s")
    new = new[["date", "open", "high", "low", "close", "volume"]]
    df = new if old is None else pd.concat([old, new])
    df = df[df["date"] < today.replace(tzinfo=None)]          # drop today's unfinished candle
    df = df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


def fetch_fng(refresh=True):
    """Crypto Fear & Greed index, one value per day (starts 2018-02-01)."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / "fng.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=["date"])
    raw = _get(FNG_URL)["data"]
    df = pd.DataFrame({"date": pd.to_datetime([int(d["timestamp"]) for d in raw], unit="s").normalize(),
                       "fng": [int(d["value"]) for d in raw]})
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    df.to_csv(path, index=False)
    return df

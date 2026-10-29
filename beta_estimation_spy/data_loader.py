"""
data_loader.py - download once, cache, align once.

`load_market_data` is the place that touches the network. It returns a
single DataFrame of daily log-returns (one column per ticker plus the
benchmark) aligned on common dates. Every experiment receives the same frame.
"""
import os

import numpy as np
import pandas as pd

from config import BENCHMARK, CACHE_FILE, END_DATE, START_DATE, TICKERS

_TOL = pd.Timedelta(days=10)


def _cache_is_valid(cached: pd.DataFrame, cols, start, end) -> bool:
    if not set(cols) <= set(cached.columns) or cached.empty:
        return False
    end_eff = min(pd.Timestamp(end), pd.Timestamp.today().normalize())
    return (cached.index.min() <= pd.Timestamp(start) + _TOL
            and cached.index.max() >= end_eff - _TOL)


def _download(cols, start, end) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:                                  # pragma: no cover
        raise RuntimeError("yfinance is required to download data: pip install yfinance") from exc

    data = yf.download(cols, start=start, end=end, auto_adjust=True, progress=False)
    if data is None or len(data) == 0:
        raise RuntimeError("Download failed (no data returned). Check the network connection.")
    closes = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
    closes = closes[cols].dropna()                              # one alignment for all columns
    return np.log(closes / closes.shift(1)).dropna()


def load_market_data(tickers=None, benchmark=None, start=None, end=None,
                     cache_path=None, refresh=False) -> pd.DataFrame:
    """Daily log-returns of `tickers` + `benchmark`, from cache when possible."""
    tickers = list(TICKERS if tickers is None else tickers)
    benchmark = BENCHMARK if benchmark is None else benchmark
    start = START_DATE if start is None else start
    end = END_DATE if end is None else end
    cache_path = CACHE_FILE if cache_path is None else cache_path
    cols = tickers + [benchmark]

    if not refresh and os.path.exists(cache_path):
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if _cache_is_valid(cached, cols, start, end):
            return cached[cols]
        print(f"[data_loader] cache {cache_path} does not cover {cols} / {start}..{end}: downloading")

    log_rets = _download(cols, start, end)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    log_rets.to_csv(cache_path)
    return log_rets

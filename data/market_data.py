"""
Market environment data for SJM features (Phase 0).
Fetches VIX, 2Y Treasury, 10Y Treasury from FRED (fredapi) or yfinance fallback.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Load .env from project root (parent of data/), regardless of cwd
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env_path)
except ImportError:
    pass

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Factor ETFs (6 factors, exclude market)
FACTOR_ETFS = ['QUAL', 'MTUM', 'USMV', 'VLUE', 'SIZE', 'IWF']
MARKET_ETF = 'SPY'

# FRED series IDs
_FRED_SERIES = [
    ('VIXCLS', 'vix'),
    ('DGS2', 'yield_2y'),
    ('DGS10', 'yield_10y'),
]


def fetch_market_data(start_date, end_date):
    """
    Fetch VIX, 2Y Treasury yield, 10Y Treasury yield.
    Tries FRED (fredapi) first, falls back to yfinance.
    Set FRED_API_KEY env var for FRED (free at https://fred.stlouisfed.org/docs/api/api_key.html).
    Returns DataFrame with columns: vix, yield_2y, yield_10y, index aligned to trading dates.
    """
    data = _fetch_market_data_fred(start_date, end_date)
    if data is not None and not data.empty:
        return data
    return _fetch_market_data_yfinance(start_date, end_date)


def _fetch_market_data_fred(start_date, end_date):
    """Fetch from FRED via fredapi. Returns None on failure."""
    try:
        from fredapi import Fred
        api_key = (os.environ.get('FRED_API_KEY') or '').strip()
        if not api_key:
            logger.info("FRED_API_KEY not set, using yfinance for market data")
            return None
        fred = Fred(api_key=api_key)
        series_dict = {}
        for sid, col in _FRED_SERIES:
            s = fred.get_series(sid, observation_start=start_date, observation_end=end_date)
            if s is not None and len(s) > 0:
                series_dict[col] = s
        if len(series_dict) < 2:
            logger.warning("FRED returned insufficient series")
            return None
        df = pd.DataFrame(series_dict)
        df = df.ffill().dropna()
        if df.empty:
            return None
        logger.info("Fetched market data from FRED: %d days" % len(df))
        return df
    except ImportError:
        logger.debug("fredapi not installed, using yfinance")
        return None
    except Exception as e:
        logger.warning("FRED fetch failed (%s), using yfinance fallback" % str(e))
        return None


def _fetch_market_data_yfinance(start_date, end_date):
    """Fallback: VIX and Treasury yields via yfinance."""
    import yfinance as yf
    # ^VIX, ^IRX (13w), ^TNX (10Y) - use ^IRX as short-rate proxy for 2Y
    series_list = []
    for sym, col in [('^VIX', 'vix'), ('^IRX', 'yield_2y'), ('^TNX', 'yield_10y')]:
        d = yf.download(sym, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if not d.empty:
            c = d['Close'].squeeze() if 'Close' in d.columns else d.iloc[:, 0]
            s = c.rename(col) if hasattr(c, 'rename') else pd.Series(c, name=col)
            series_list.append(s)
    if not series_list:
        return pd.DataFrame()
    df = pd.concat(series_list, axis=1)
    df = df.ffill().dropna()
    logger.info("Fetched market data from yfinance: %d days" % len(df))
    return df

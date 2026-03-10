"""
Market environment data for SJM features (Phase 0).
Fetches VIX, 2Y Treasury, 10Y Treasury from FRED or yfinance fallback.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Factor ETFs (6 factors, exclude market)
FACTOR_ETFS = ['QUAL', 'MTUM', 'USMV', 'VLUE', 'SIZE', 'IWF']
MARKET_ETF = 'SPY'


def fetch_market_data(start_date, end_date):
    """
    Fetch VIX, 2Y Treasury yield, 10Y Treasury yield.
    Tries FRED first, falls back to yfinance.
    Returns DataFrame with columns: vix, yield_2y, yield_10y, index aligned to trading dates.
    """
    try:
        import pandas_datareader.data as web
        fred_data = web.DataReader(
            ['VIXCLS', 'DGS2', 'DGS10'],
            'fred',
            start=start_date,
            end=end_date
        )
        fred_data.columns = ['vix', 'yield_2y', 'yield_10y']
        fred_data = fred_data.ffill().dropna()
        logger.info("Fetched market data from FRED: %d days" % len(fred_data))
        return fred_data
    except Exception as e:
        logger.warning("FRED fetch failed (%s), using yfinance fallback" % str(e))
        return _fetch_market_data_yfinance(start_date, end_date)


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

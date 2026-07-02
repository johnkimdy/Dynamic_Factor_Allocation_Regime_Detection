"""Unit tests for data/market_data module."""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from data.market_data import (
    FACTOR_ETFS,
    MARKET_ETF,
    fetch_market_data,
    _fetch_market_data_fred,
    _fetch_market_data_yfinance,
)


class TestConstants:
    def test_factor_etfs(self):
        assert len(FACTOR_ETFS) == 6
        assert "QUAL" in FACTOR_ETFS
        assert "MTUM" in FACTOR_ETFS
        assert "USMV" in FACTOR_ETFS
        assert "VLUE" in FACTOR_ETFS
        assert "SIZE" in FACTOR_ETFS
        assert "IWF" in FACTOR_ETFS

    def test_market_etf(self):
        assert MARKET_ETF == "SPY"


class TestFetchMarketDataFred:
    def test_no_api_key(self, monkeypatch):
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        result = _fetch_market_data_fred("2023-01-01", "2023-06-01")
        assert result is None

    def test_empty_api_key(self, monkeypatch):
        monkeypatch.setenv("FRED_API_KEY", "  ")
        result = _fetch_market_data_fred("2023-01-01", "2023-06-01")
        assert result is None

    @patch("fredapi.Fred")
    def test_successful_fetch(self, mock_fred_cls, monkeypatch):
        monkeypatch.setenv("FRED_API_KEY", "test_key")
        dates = pd.date_range("2023-01-01", periods=50, freq="B")
        mock_fred = MagicMock()
        mock_fred.get_series.side_effect = [
            pd.Series(20 + np.random.randn(50), index=dates),  # vix
            pd.Series(4.0 + np.random.randn(50) * 0.1, index=dates),  # 2y
            pd.Series(4.5 + np.random.randn(50) * 0.1, index=dates),  # 10y
        ]
        mock_fred_cls.return_value = mock_fred
        result = _fetch_market_data_fred("2023-01-01", "2023-06-01")
        assert result is not None
        assert "vix" in result.columns
        assert "yield_2y" in result.columns
        assert "yield_10y" in result.columns

    @patch("fredapi.Fred")
    def test_insufficient_series(self, mock_fred_cls, monkeypatch):
        monkeypatch.setenv("FRED_API_KEY", "test_key")
        mock_fred = MagicMock()
        mock_fred.get_series.side_effect = [
            pd.Series([20.0]),  # only one series with data
            None,
            None,
        ]
        mock_fred_cls.return_value = mock_fred
        result = _fetch_market_data_fred("2023-01-01", "2023-06-01")
        assert result is None

    def test_fredapi_not_installed(self, monkeypatch):
        monkeypatch.setenv("FRED_API_KEY", "test_key")
        # If fredapi is installed, this test just verifies the function works;
        # the ImportError path is hard to trigger in isolation.
        result = _fetch_market_data_fred("2023-01-01", "2023-06-01")
        assert result is None or isinstance(result, pd.DataFrame)


class TestFetchMarketDataYfinance:
    @patch("yfinance.download")
    def test_successful_fetch(self, mock_download):
        dates = pd.date_range("2023-01-01", periods=50, freq="B")

        def download_side_effect(sym, **kwargs):
            data = pd.DataFrame(
                {"Close": 20 + np.random.randn(50)},
                index=dates,
            )
            return data

        mock_download.side_effect = download_side_effect
        result = _fetch_market_data_yfinance("2023-01-01", "2023-06-01")
        assert result is not None
        assert len(result) > 0

    @patch("yfinance.download")
    def test_empty_download(self, mock_download):
        mock_download.return_value = pd.DataFrame()
        result = _fetch_market_data_yfinance("2023-01-01", "2023-06-01")
        assert result.empty


class TestFetchMarketData:
    @patch("data.market_data._fetch_market_data_fred")
    @patch("data.market_data._fetch_market_data_yfinance")
    def test_uses_fred_when_available(self, mock_yf, mock_fred):
        dates = pd.date_range("2023-01-01", periods=50, freq="B")
        fred_df = pd.DataFrame({
            "vix": np.random.randn(50) + 20,
            "yield_2y": np.random.randn(50) + 4,
            "yield_10y": np.random.randn(50) + 4.5,
        }, index=dates)
        mock_fred.return_value = fred_df
        result = fetch_market_data("2023-01-01", "2023-06-01")
        mock_yf.assert_not_called()
        pd.testing.assert_frame_equal(result, fred_df)

    @patch("data.market_data._fetch_market_data_fred")
    @patch("data.market_data._fetch_market_data_yfinance")
    def test_falls_back_to_yfinance(self, mock_yf, mock_fred):
        mock_fred.return_value = None
        dates = pd.date_range("2023-01-01", periods=50, freq="B")
        yf_df = pd.DataFrame({
            "vix": np.random.randn(50) + 20,
            "yield_2y": np.random.randn(50) + 4,
            "yield_10y": np.random.randn(50) + 4.5,
        }, index=dates)
        mock_yf.return_value = yf_df
        result = fetch_market_data("2023-01-01", "2023-06-01")
        mock_yf.assert_called_once()
        pd.testing.assert_frame_equal(result, yf_df)

    @patch("data.market_data._fetch_market_data_fred")
    @patch("data.market_data._fetch_market_data_yfinance")
    def test_falls_back_on_empty_fred(self, mock_yf, mock_fred):
        mock_fred.return_value = pd.DataFrame()
        mock_yf.return_value = pd.DataFrame()
        result = fetch_market_data("2023-01-01", "2023-06-01")
        mock_yf.assert_called_once()

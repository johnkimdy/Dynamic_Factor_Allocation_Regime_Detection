"""Unit tests for utility functions in analyze_strategy.py."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from analyze_strategy import (
    _format_duration,
    _progress_bar,
    _sortino,
    apply_er_drag,
    metrics_from_returns,
    _serialize_period_for_export,
)


# ---------------------------------------------------------------------------
# _format_duration
# ---------------------------------------------------------------------------

class TestFormatDuration:
    def test_seconds(self):
        assert _format_duration(30) == "30s"

    def test_minutes_seconds(self):
        assert _format_duration(90) == "1m 30s"

    def test_minutes_exact(self):
        assert _format_duration(120) == "2m"

    def test_hours_minutes(self):
        assert _format_duration(3720) == "1h 2m"

    def test_hours_exact(self):
        assert _format_duration(3600) == "1h"

    def test_zero(self):
        assert _format_duration(0) == "0s"


# ---------------------------------------------------------------------------
# _progress_bar
# ---------------------------------------------------------------------------

class TestProgressBar:
    def test_empty(self):
        bar = _progress_bar(0, 10, width=10)
        assert bar == "[░░░░░░░░░░]"

    def test_full(self):
        bar = _progress_bar(10, 10, width=10)
        assert bar == "[██████████]"

    def test_half(self):
        bar = _progress_bar(5, 10, width=10)
        assert bar == "[█████░░░░░]"

    def test_zero_total(self):
        bar = _progress_bar(0, 0, width=10)
        assert bar == "[░░░░░░░░░░]"


# ---------------------------------------------------------------------------
# _sortino
# ---------------------------------------------------------------------------

class TestSortino:
    def test_positive_returns_only(self):
        returns = pd.Series([0.01, 0.02, 0.03, 0.01, 0.02])
        assert _sortino(returns) == 0.0

    def test_mixed_returns(self):
        np.random.seed(42)
        returns = pd.Series(np.random.randn(500) * 0.01)
        s = _sortino(returns)
        assert isinstance(s, float)
        assert np.isfinite(s)

    def test_empty_returns(self):
        returns = pd.Series([], dtype=float)
        # _sortino on empty -> no downside -> 0
        assert _sortino(returns) == 0.0


# ---------------------------------------------------------------------------
# apply_er_drag
# ---------------------------------------------------------------------------

class TestApplyErDrag:
    def test_reduces_returns(self):
        returns = pd.Series([0.001, 0.002, -0.001, 0.0])
        net = apply_er_drag(returns, 0.001)
        daily_drag = 0.001 / 252.0
        expected = returns - daily_drag
        pd.testing.assert_series_equal(net, expected)

    def test_zero_er(self):
        returns = pd.Series([0.01, -0.01])
        pd.testing.assert_series_equal(apply_er_drag(returns, 0.0), returns)


# ---------------------------------------------------------------------------
# metrics_from_returns
# ---------------------------------------------------------------------------

class TestMetricsFromReturns:
    def test_normal_returns(self):
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01)
        m = metrics_from_returns(returns)
        assert "total_return" in m
        assert "sharpe" in m
        assert "sortino" in m
        assert "vol" in m
        assert "max_dd" in m
        assert m["max_dd"] <= 0

    def test_empty_returns(self):
        m = metrics_from_returns(pd.Series([], dtype=float))
        assert m["total_return"] == 0
        assert m["sharpe"] == 0

    def test_constant_returns(self):
        returns = pd.Series([0.001] * 100)
        m = metrics_from_returns(returns)
        assert m["total_return"] > 0
        # std is extremely small but not exactly 0 due to floating point,
        # so sharpe may be huge; just check it's finite
        assert np.isfinite(m["sharpe"])

    def test_max_dd_negative(self):
        returns = pd.Series([0.01, 0.01, -0.05, -0.05, 0.01])
        m = metrics_from_returns(returns)
        assert m["max_dd"] < 0


# ---------------------------------------------------------------------------
# _serialize_period_for_export
# ---------------------------------------------------------------------------

class TestSerializePeriodForExport:
    def _make_results(self, n=100):
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        pv = pd.Series(100.0 * (1 + np.random.randn(n).cumsum() * 0.001), index=dates)
        return {
            "portfolio_values": pv,
            "weights_history": [(dates[0], pd.Series({"SPY": 0.5, "QUAL": 0.5}))],
            "rebalance_dates": [dates[0]],
            "total_return": 0.05,
            "annualized_volatility": 0.12,
            "sharpe_ratio": 0.8,
            "max_drawdown": -0.03,
            "n_rebalances": 1,
        }

    def test_basic_serialization(self):
        np.random.seed(42)
        results = self._make_results()
        out = _serialize_period_for_export("2023-01-01", "2023-06-01", "H1 2023", results, None, None)
        assert out["period"] == "H1 2023"
        assert out["start_date"] == "2023-01-01"
        assert "metrics" in out
        assert out["metrics"]["total_return"] == 0.05
        assert len(out["portfolio_values"]) > 0
        assert out["ew7_values"] == []
        assert out["spy_values"] == []

    def test_with_benchmarks(self):
        np.random.seed(42)
        results = self._make_results()
        dates = results["portfolio_values"].index
        ew7 = {
            "total_return": 0.04,
            "sharpe_ratio": 0.7,
            "sortino_ratio": 0.9,
            "annualized_volatility": 0.11,
            "max_drawdown": -0.02,
            "portfolio_values": pd.Series(100 + np.arange(len(dates)) * 0.1, index=dates),
        }
        spy = {
            "total_return": 0.06,
            "sharpe_ratio": 0.9,
            "sortino_ratio": 1.0,
            "annualized_volatility": 0.13,
            "max_drawdown": -0.04,
            "portfolio_values": pd.Series(100 + np.arange(len(dates)) * 0.12, index=dates),
        }
        out = _serialize_period_for_export("2023-01-01", "2023-06-01", "test", results, ew7, spy)
        assert out["benchmarks"]["ew7"]["total_return"] == 0.04
        assert out["benchmarks"]["spy"]["total_return"] == 0.06
        assert len(out["ew7_values"]) > 0
        assert len(out["spy_values"]) > 0

    def test_with_asym_results(self):
        np.random.seed(42)
        results = self._make_results()
        asym = self._make_results()
        out = _serialize_period_for_export("2023-01-01", "2023-06-01", "test", results, None, None, asym_results=asym)
        assert out["benchmarks"]["helix_asym"] is not None
        assert len(out["asym_values"]) > 0

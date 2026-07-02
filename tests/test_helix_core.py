"""Unit tests for untested parts of helix_factor_strategy.py.

Covers: compute_active_returns, helper functions (_rsi, _stoch_k, _macd,
_compute_bcss, _soft_thres_l2_normalized, _solve_lasso,
_symmetric_jump_penalty_matrix), BlackLittermanOptimizer,
HelixFactorStrategy methods (generate_expected_returns, should_rebalance,
validate_regime_inference, _ewma_covariance, _calculate_max_drawdown,
check_covariance_matrix, calculate_returns, optimize_portfolio).
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from helix_factor_strategy import (
    BlackLittermanOptimizer,
    HelixFactorStrategy,
    SparseJumpModel,
    compute_active_returns,
    _rsi,
    _stoch_k,
    _macd,
    _compute_bcss,
    _soft_thres_l2_normalized,
    _solve_lasso,
    _symmetric_jump_penalty_matrix,
    MARKET_ETF,
)


# ---------------------------------------------------------------------------
# compute_active_returns
# ---------------------------------------------------------------------------

class TestComputeActiveReturns:
    def test_basic(self):
        df = pd.DataFrame({
            "SPY": [0.01, 0.02, -0.01],
            "QUAL": [0.02, 0.01, 0.00],
            "MTUM": [0.00, 0.03, -0.02],
        })
        active = compute_active_returns(df, market_col="SPY")
        assert "SPY" not in active.columns
        assert list(active.columns) == ["QUAL", "MTUM"]
        np.testing.assert_allclose(active["QUAL"].values, [0.01, -0.01, 0.01])
        np.testing.assert_allclose(active["MTUM"].values, [-0.01, 0.01, -0.01])

    def test_missing_market_col_raises(self):
        df = pd.DataFrame({"QUAL": [0.01], "MTUM": [0.02]})
        with pytest.raises(ValueError, match="Market column"):
            compute_active_returns(df, market_col="SPY")


# ---------------------------------------------------------------------------
# Technical indicator helpers
# ---------------------------------------------------------------------------

class TestTechnicalIndicators:
    def test_rsi_range(self):
        series = pd.Series(np.random.randn(100).cumsum())
        rsi_vals = _rsi(series, 14)
        assert len(rsi_vals) == 100
        assert np.all(np.isfinite(rsi_vals))
        # RSI should be in [0, 100]
        assert np.all(rsi_vals >= 0)
        assert np.all(rsi_vals <= 100)

    def test_stoch_k_range(self):
        series = pd.Series(np.random.randn(100).cumsum())
        sk = _stoch_k(series, 14)
        assert len(sk) == 100

    def test_macd(self):
        series = pd.Series(np.random.randn(100).cumsum())
        m = _macd(series, 8, 21)
        assert len(m) == 100
        assert np.all(np.isfinite(m))


# ---------------------------------------------------------------------------
# BCSS and sparsity helpers
# ---------------------------------------------------------------------------

class TestBCSSAndSparsity:
    def test_compute_bcss_basic(self):
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        proba = np.array([[1, 0], [1, 0], [0, 1]], dtype=float)
        bcss = _compute_bcss(X, proba)
        assert len(bcss) == 2
        assert np.all(bcss >= 0)

    def test_soft_thres_l2_normalized(self):
        x = np.array([3.0, 1.0, 0.5])
        result = _soft_thres_l2_normalized(x, 1.0)
        # After thresholding: [2.0, 0.0, 0.0] -> L2 normalized: [1.0, 0.0, 0.0]
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0])

    def test_soft_thres_all_zero(self):
        x = np.array([0.5, 0.3])
        result = _soft_thres_l2_normalized(x, 1.0)
        # All values below threshold -> zero vector returned
        np.testing.assert_allclose(result, [0.0, 0.0])

    def test_solve_lasso_uniform(self):
        a = np.array([1.0, 1.0, 1.0, 1.0])
        w = _solve_lasso(a, norm_ub=2.0)
        assert len(w) == 4
        # L2 norm should be 1
        np.testing.assert_allclose(np.linalg.norm(w), 1.0, atol=1e-6)

    def test_solve_lasso_small_norm_ub(self):
        a = np.array([5.0, 3.0, 1.0])
        w = _solve_lasso(a, norm_ub=0.5)
        # norm_ub < 1 is clipped to 1
        np.testing.assert_allclose(np.linalg.norm(w), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# _symmetric_jump_penalty_matrix
# ---------------------------------------------------------------------------

class TestSymmetricJumpPenaltyMatrix:
    def test_basic(self):
        L = _symmetric_jump_penalty_matrix(2, 50.0)
        expected = np.array([[0.0, 50.0], [50.0, 0.0]])
        np.testing.assert_array_equal(L, expected)

    def test_3_regimes(self):
        L = _symmetric_jump_penalty_matrix(3, 10.0)
        assert L.shape == (3, 3)
        assert np.all(np.diag(L) == 0)
        off_diag = L[~np.eye(3, dtype=bool)]
        assert np.all(off_diag == 10.0)


# ---------------------------------------------------------------------------
# SparseJumpModel additional tests
# ---------------------------------------------------------------------------

class TestSparseJumpModelExtras:
    def test_invalid_jump_penalty_matrix_shape(self):
        with pytest.raises(ValueError, match="jump_penalty_matrix has shape"):
            SparseJumpModel(n_regimes=2, jump_penalty_matrix=np.zeros((3, 3)))

    def test_predict_regime_unfitted(self):
        model = SparseJumpModel()
        with pytest.raises(ValueError, match="must be fitted"):
            model.predict_regime(pd.Series([0.01, 0.02]))

    def test_predict_regime_fitted(self):
        np.random.seed(42)
        ar = pd.Series(
            0.001 * np.random.randn(200),
            index=pd.date_range("2020-01-01", periods=200, freq="B"),
        )
        model = SparseJumpModel(n_regimes=2, jump_penalty=50.0, sparsity_param=9.5)
        model.fit(ar)
        pred = model.predict_regime(ar[:10])
        assert len(pred) == 10

    def test_infer_regime_online_unfitted(self):
        model = SparseJumpModel()
        with pytest.raises(ValueError, match="must be fitted"):
            model.infer_regime_online(np.zeros((1, 4)))

    def test_infer_regime_online_dim_mismatch(self):
        np.random.seed(42)
        ar = pd.Series(
            0.001 * np.random.randn(200),
            index=pd.date_range("2020-01-01", periods=200, freq="B"),
        )
        model = SparseJumpModel(n_regimes=2, jump_penalty=50.0)
        model.fit(ar)
        with pytest.raises(ValueError, match="Feature dim mismatch"):
            model.infer_regime_online(np.zeros((1, 99)))

    def test_fit_empty_raises(self):
        ar = pd.Series([], dtype=float)
        model = SparseJumpModel()
        with pytest.raises(ValueError):
            model.fit(ar)

    def test_loss_curve_recorded(self):
        np.random.seed(42)
        ar = pd.Series(
            0.001 * np.random.randn(200),
            index=pd.date_range("2020-01-01", periods=200, freq="B"),
        )
        model = SparseJumpModel(record_loss_curve=True, max_iter=10)
        model.fit(ar)
        assert len(model.loss_curve_) > 0
        # Loss should generally be non-increasing (allowing small floating point noise)
        for i in range(1, len(model.loss_curve_)):
            assert model.loss_curve_[i] <= model.loss_curve_[0] * 1.01

    def test_infer_regime_online_1d_input(self):
        np.random.seed(42)
        ar = pd.Series(
            0.001 * np.random.randn(200),
            index=pd.date_range("2020-01-01", periods=200, freq="B"),
        )
        model = SparseJumpModel(n_regimes=2, jump_penalty=50.0)
        model.fit(ar)
        n_feat = model.centroids_.shape[1]
        regime = model.infer_regime_online(np.zeros(n_feat))
        assert regime in (0, 1)


# ---------------------------------------------------------------------------
# BlackLittermanOptimizer
# ---------------------------------------------------------------------------

class TestBlackLittermanOptimizer:
    def _make_cov(self, assets):
        n = len(assets)
        np.random.seed(42)
        A = np.random.randn(n, n) * 0.01
        cov = A @ A.T + np.eye(n) * 0.0001
        return pd.DataFrame(cov, index=assets, columns=assets)

    def test_basic_optimize(self):
        assets = ["SPY", "QUAL", "MTUM", "USMV", "VLUE", "SIZE", "IWF"]
        cov = self._make_cov(assets)
        expected = {"QUAL": 0.001, "MTUM": -0.001, "USMV": 0.0005,
                    "VLUE": 0.0, "SIZE": 0.0002, "IWF": -0.0003}
        opt = BlackLittermanOptimizer(risk_aversion=2.5, tau=0.025)
        weights = opt.optimize(expected, cov)
        assert len(weights) == 7
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)
        assert (weights >= 0).all()  # long-only

    def test_no_views_returns_benchmark(self):
        assets = ["SPY", "QUAL"]
        cov = self._make_cov(assets)
        opt = BlackLittermanOptimizer()
        weights = opt.optimize({}, cov)
        # Should return benchmark (equal weights)
        np.testing.assert_allclose(weights.values, [0.5, 0.5], atol=0.01)

    def test_custom_market_weights(self):
        assets = ["SPY", "QUAL", "MTUM"]
        cov = self._make_cov(assets)
        mw = pd.Series([0.6, 0.2, 0.2], index=assets)
        opt = BlackLittermanOptimizer()
        weights = opt.optimize({"QUAL": 0.001}, cov, market_weights=mw)
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)

    def test_optimize_with_te_target(self):
        assets = ["SPY", "QUAL", "MTUM", "USMV", "VLUE", "SIZE", "IWF"]
        cov = self._make_cov(assets)
        expected = {"QUAL": 0.001, "MTUM": -0.001, "USMV": 0.0005,
                    "VLUE": 0.0, "SIZE": 0.0002, "IWF": -0.0003}
        opt = BlackLittermanOptimizer(risk_aversion=2.5, tau=0.025)
        weights = opt.optimize_with_te_target(expected, cov, target_te_annual=0.02)
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)

    def test_calculate_single_factor_lambda(self):
        assets = ["SPY", "QUAL", "MTUM"]
        cov = self._make_cov(assets)
        opt = BlackLittermanOptimizer()
        lam, P_j, eta_j = opt.calculate_single_factor_lambda(
            0.001, "QUAL", "SPY", cov
        )
        assert isinstance(lam, float)
        assert np.isfinite(lam)
        assert eta_j > 0


# ---------------------------------------------------------------------------
# HelixFactorStrategy
# ---------------------------------------------------------------------------

class TestHelixFactorStrategy:
    def _make_strategy_with_data(self, n=500):
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        cols = ["SPY", "QUAL", "MTUM", "USMV", "VLUE", "SIZE", "IWF"]
        data = pd.DataFrame(
            100 + np.random.randn(n, 7).cumsum(axis=0) * 0.5,
            index=dates,
            columns=cols,
        )
        strategy = HelixFactorStrategy(lookback_days=252)
        strategy.data = data
        strategy.market_data = None
        return strategy

    def test_calculate_returns(self):
        strategy = self._make_strategy_with_data()
        returns = strategy.calculate_returns()
        assert len(returns) == len(strategy.data) - 1
        assert list(returns.columns) == list(strategy.data.columns)

    def test_calculate_returns_no_data(self):
        strategy = HelixFactorStrategy()
        with pytest.raises(ValueError, match="No data available"):
            strategy.calculate_returns()

    def test_generate_expected_returns(self):
        strategy = self._make_strategy_with_data()
        strategy.regime_means = {
            "QUAL": {0: 0.02, 1: -0.01},
            "MTUM": {0: 0.01, 1: 0.03},
        }
        strategy.regime_models = {"QUAL": None, "MTUM": None}
        current_regimes = {"QUAL": 0, "MTUM": 1}
        expected = strategy.generate_expected_returns(current_regimes)
        assert "QUAL" in expected
        assert "MTUM" in expected
        np.testing.assert_allclose(expected["QUAL"], 0.02 / 252, atol=1e-8)
        np.testing.assert_allclose(expected["MTUM"], 0.03 / 252, atol=1e-8)

    def test_generate_expected_returns_missing_regime(self):
        strategy = self._make_strategy_with_data()
        strategy.regime_means = {"QUAL": {0: 0.02}}
        strategy.regime_models = {"QUAL": None}
        expected = strategy.generate_expected_returns({"QUAL": 5})  # regime 5 doesn't exist
        assert expected["QUAL"] == 0.0 / 252  # default 0.0

    def test_should_rebalance_no_weights(self):
        strategy = HelixFactorStrategy()
        assert strategy.should_rebalance(pd.Series([0.5, 0.5])) is True

    def test_should_rebalance_daily(self):
        strategy = HelixFactorStrategy(apply_daily_weights=True)
        strategy.current_weights = pd.Series([0.5, 0.5])
        assert strategy.should_rebalance(pd.Series([0.5, 0.5])) is True

    def test_should_rebalance_threshold(self):
        strategy = HelixFactorStrategy(apply_daily_weights=False, rebalance_threshold=0.01)
        strategy.current_weights = pd.Series([0.5, 0.5], index=["A", "B"])
        # Small change below threshold
        assert not strategy.should_rebalance(pd.Series([0.505, 0.495], index=["A", "B"]))
        # Large change above threshold
        assert strategy.should_rebalance(pd.Series([0.6, 0.4], index=["A", "B"]))

    def test_validate_regime_inference_ok(self):
        strategy = HelixFactorStrategy()
        regimes = {"QUAL": 0, "MTUM": 1}
        expected = {"QUAL": 0.0001, "MTUM": -0.0001}
        ok, msg = strategy.validate_regime_inference(regimes, expected)
        assert ok is True
        assert msg == "OK"

    def test_validate_regime_inference_no_regimes(self):
        strategy = HelixFactorStrategy()
        ok, msg = strategy.validate_regime_inference(None)
        assert ok is False

    def test_validate_regime_inference_out_of_range(self):
        strategy = HelixFactorStrategy()
        ok, msg = strategy.validate_regime_inference({"QUAL": 5})
        assert ok is False
        assert "out of range" in msg

    def test_validate_regime_inference_nan_expected(self):
        strategy = HelixFactorStrategy()
        ok, msg = strategy.validate_regime_inference(
            {"QUAL": 0}, {"QUAL": float("nan")}
        )
        assert ok is False
        assert "NaN" in msg

    def test_validate_regime_inference_exceeds_cap(self):
        strategy = HelixFactorStrategy()
        ok, msg = strategy.validate_regime_inference(
            {"QUAL": 0}, {"QUAL": 0.10}  # 0.10 * 252 >> 5%
        )
        assert ok is False
        assert "exceeds" in msg

    def test_ewma_covariance(self):
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        returns = pd.DataFrame(
            np.random.randn(100, 3) * 0.01,
            index=dates,
            columns=["A", "B", "C"],
        )
        strategy = HelixFactorStrategy()
        cov = strategy._ewma_covariance(returns)
        assert cov.shape == (3, 3)
        # Should be symmetric
        np.testing.assert_allclose(cov.values, cov.values.T, atol=1e-10)
        # Diagonal should be positive
        assert np.all(np.diag(cov.values) > 0)

    def test_calculate_max_drawdown(self):
        strategy = HelixFactorStrategy()
        pv = pd.Series([100, 110, 105, 95, 100, 120])
        dd = strategy._calculate_max_drawdown(pv)
        # Max drawdown: 110 -> 95 = -15/110 ≈ -0.1364
        assert dd < 0
        np.testing.assert_allclose(dd, (95 - 110) / 110, atol=1e-4)

    def test_calculate_max_drawdown_empty(self):
        strategy = HelixFactorStrategy()
        dd = strategy._calculate_max_drawdown(pd.Series([], dtype=float))
        assert dd == 0.0

    def test_check_covariance_matrix_psd(self):
        np.random.seed(42)
        A = np.random.randn(3, 3)
        cov = pd.DataFrame(A @ A.T + np.eye(3) * 0.01, columns=["A", "B", "C"], index=["A", "B", "C"])
        diag = HelixFactorStrategy.check_covariance_matrix(cov, "test")
        assert diag["is_psd"] is True
        assert diag["diag_nonpositive_count"] == 0
        assert diag["shape"] == (3, 3)

    def test_check_covariance_matrix_not_psd(self):
        # Construct a matrix that is NOT positive semi-definite
        cov = np.array([[1, 0, 2], [0, 1, 0], [2, 0, 1]], dtype=float)
        diag = HelixFactorStrategy.check_covariance_matrix(cov, "test_not_psd")
        assert diag["is_psd"] is False

    def test_optimize_portfolio(self):
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=300, freq="B")
        cols = ["SPY", "QUAL", "MTUM", "USMV", "VLUE", "SIZE", "IWF"]
        returns = pd.DataFrame(
            np.random.randn(300, 7) * 0.01,
            index=dates,
            columns=cols,
        )
        strategy = HelixFactorStrategy()
        expected_active = {"QUAL": 0.001, "MTUM": -0.001, "USMV": 0.0005,
                           "VLUE": 0.0, "SIZE": 0.0002, "IWF": -0.0003}
        weights = strategy.optimize_portfolio(returns, expected_active)
        assert len(weights) == 7
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)

    def test_optimize_portfolio_with_te_target(self):
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=300, freq="B")
        cols = ["SPY", "QUAL", "MTUM", "USMV", "VLUE", "SIZE", "IWF"]
        returns = pd.DataFrame(
            np.random.randn(300, 7) * 0.01,
            index=dates,
            columns=cols,
        )
        strategy = HelixFactorStrategy(target_tracking_error=0.02)
        expected_active = {"QUAL": 0.001, "MTUM": -0.001, "USMV": 0.0005,
                           "VLUE": 0.0, "SIZE": 0.0002, "IWF": -0.0003}
        weights = strategy.optimize_portfolio(returns, expected_active)
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)

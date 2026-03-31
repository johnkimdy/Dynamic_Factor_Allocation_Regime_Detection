"""
JOH-8: Integration tests for SJM hyperparameter convergence.
Asserts that (λ, κ²) affect regime detection and feature sparsity.

Run from project root: python tests/test_sjm_hyperparameters.py
"""

import logging
import os
import sys

# Ensure project root is on path
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import itertools
import numpy as np
import pandas as pd

from helix_factor_strategy import SparseJumpModel


def _brute_optimal_regimes(X, centroids, weights, L):
    """Exhaustive search; validates DP for small n_obs × n_regimes."""
    n_obs = X.shape[0]
    K = centroids.shape[0]
    best_cost = np.inf
    best_r = None
    for path in itertools.product(range(K), repeat=n_obs):
        path = np.array(path, dtype=int)
        cost = 0.0
        for t in range(n_obs):
            d = weights * (X[t] - centroids[path[t]])
            cost += 0.5 * np.sum(d ** 2)
        for t in range(1, n_obs):
            cost += L[path[t - 1], path[t]]
        if cost < best_cost:
            best_cost = cost
            best_r = path.copy()
    return best_r


def test_regimes_differ_with_lambda_and_kappa():
    """Different (jump_penalty, sparsity_param) must produce different regime sequences."""
    np.random.seed(42)
    n = 500
    ar = pd.Series(
        0.001 * np.random.randn(n),
        index=pd.date_range('2020-01-01', periods=n, freq='B')
    )

    configs = [(5.0, 2.0), (50.0, 9.5), (150.0, 18.0)]
    regimes_list = []
    for lam, k2 in configs:
        model = SparseJumpModel(n_regimes=2, jump_penalty=lam, sparsity_param=k2)
        model.fit(ar)
        regimes_list.append(model.regimes_)

    r0, r1, r2 = regimes_list
    diff_01 = (r0 != r1).sum()
    diff_12 = (r1 != r2).sum()
    diff_02 = (r0 != r2).sum()

    assert diff_01 > 0 or diff_12 > 0 or diff_02 > 0, (
        "JOH-8: All (λ, κ²) configs produced identical regimes. "
        "Hyperparameters have no effect on SJM output."
    )


def test_sparsity_param_affects_feature_weights():
    """Lower sparsity_param must yield fewer non-zero feature weights (BCSS+Lasso)."""
    np.random.seed(42)
    # Use 20 features so L1 bound sqrt(kappa²) has room to vary (legacy has only 4)
    n, n_feat = 500, 20
    dates = pd.date_range('2020-01-01', periods=n, freq='B')
    ar = pd.Series(0.001 * np.random.randn(n), index=dates)
    # Inject precomputed features via a model that produces many features
    from helix_factor_strategy import HelixFactorStrategy, compute_active_returns
    ret = pd.DataFrame(
        0.0002 * np.random.randn(n, 7),
        index=dates,
        columns=['SPY', 'QUAL', 'MTUM', 'USMV', 'VLUE', 'SIZE', 'IWF']
    )
    md = pd.DataFrame({
        'vix': 20 + 5 * np.random.randn(n),
        'yield_2y': 2 + 0.5 * np.random.randn(n),
        'yield_10y': 3 + 0.5 * np.random.randn(n),
    }, index=dates)
    strategy = HelixFactorStrategy()
    strategy.data = ret
    strategy.market_data = md
    active_returns = compute_active_returns(ret, market_col='SPY')
    market_returns = ret['SPY']

    m_lo = SparseJumpModel(jump_penalty=50, sparsity_param=2)
    m_hi = SparseJumpModel(jump_penalty=50, sparsity_param=18)
    m_lo.fit(active_returns['QUAL'], market_data=md, market_returns=market_returns, factor_name='QUAL')
    m_hi.fit(active_returns['QUAL'], market_data=md, market_returns=market_returns, factor_name='QUAL')

    nz_lo = (np.abs(m_lo.feature_weights_) > 1e-6).sum()
    nz_hi = (np.abs(m_hi.feature_weights_) > 1e-6).sum()

    assert nz_lo <= nz_hi, (
        "JOH-8: sparsity_param inverted: κ²=2 gave {} non-zero, κ²=18 gave {} (expected fewer for lower κ²)"
        .format(nz_lo, nz_hi)
    )


def test_sjm_config_passed_to_model():
    """Verify sjm_config flow: fit_regime_models applies per-factor config."""
    from helix_factor_strategy import HelixFactorStrategy, compute_active_returns, MARKET_ETF

    # Use synthetic data to avoid network
    np.random.seed(42)
    dates = pd.date_range('2000-01-01', periods=3000, freq='B')
    n = len(dates)
    ret = pd.DataFrame(
        0.0002 * np.random.randn(n, 7),
        index=dates,
        columns=['SPY', 'QUAL', 'MTUM', 'USMV', 'VLUE', 'SIZE', 'IWF']
    )
    ret['SPY'] += 0.0001  # slight market drift

    strategy = HelixFactorStrategy(lookback_days=252 * 5)
    strategy.data = ret  # Inject data without fetch
    strategy.market_data = None  # Use legacy feature path

    active_returns = compute_active_returns(ret, market_col=MARKET_ETF)
    market_returns = ret[MARKET_ETF]

    cfg = {'QUAL': {'jump_penalty': 99.0, 'sparsity_param': 7.0}}
    strategy.fit_regime_models(active_returns, market_returns, sjm_config=cfg)

    model = strategy.regime_models.get('QUAL')
    assert model is not None
    assert model.jump_penalty == 99.0
    assert model.sparsity_param == 7.0


def test_symmetric_matrix_equivalent_to_scalar_jump_penalty():
    """Explicit Λ with equal off-diagonals must match scalar λ (JOH-12 backward compat)."""
    np.random.seed(42)
    n = 400
    ar = pd.Series(
        0.001 * np.random.randn(n),
        index=pd.date_range('2020-01-01', periods=n, freq='B')
    )
    lam, k2 = 50.0, 9.5
    m_scalar = SparseJumpModel(n_regimes=2, jump_penalty=lam, sparsity_param=k2)
    m_scalar.fit(ar)
    L = np.array([[0.0, lam], [lam, 0.0]])
    m_mat = SparseJumpModel(n_regimes=2, jump_penalty_matrix=L, sparsity_param=k2, jump_penalty=lam)
    m_mat.fit(ar)
    assert (m_scalar.regimes_.values == m_mat.regimes_.values).all(), (
        "Scalar λ and symmetric jump_penalty_matrix should yield identical regimes"
    )


def test_optimize_regimes_dp_matches_bruteforce_asymmetric():
    """Dynamic programming with asymmetric Λ matches exhaustive search."""
    np.random.seed(0)
    n_obs, n_feat, K = 7, 4, 2
    X = np.random.randn(n_obs, n_feat)
    centroids = np.random.randn(K, n_feat)
    weights = np.abs(np.random.randn(n_feat))
    weights /= weights.sum()
    L = np.array([[0.0, 1.5], [4.0, 0.0]])
    model = SparseJumpModel(n_regimes=K, jump_penalty_matrix=L)
    dp_path = model._optimize_regimes(X, centroids, weights, jump_penalty_matrix=L)
    bf_path = _brute_optimal_regimes(X, centroids, weights, L)
    np.testing.assert_array_equal(dp_path, bf_path)


def test_sjm_config_jump_penalty_matrix():
    """fit_regime_models passes asymmetric Λ into SparseJumpModel."""
    from helix_factor_strategy import HelixFactorStrategy, compute_active_returns, MARKET_ETF

    np.random.seed(42)
    dates = pd.date_range('2000-01-01', periods=800, freq='B')
    n = len(dates)
    ret = pd.DataFrame(
        0.0002 * np.random.randn(n, 7),
        index=dates,
        columns=['SPY', 'QUAL', 'MTUM', 'USMV', 'VLUE', 'SIZE', 'IWF']
    )
    strategy = HelixFactorStrategy(lookback_days=252 * 3)
    strategy.data = ret
    strategy.market_data = None

    active_returns = compute_active_returns(ret, market_col=MARKET_ETF)
    market_returns = ret[MARKET_ETF]
    L = [[0.0, 80.0], [20.0, 0.0]]
    cfg = {'QUAL': {'jump_penalty_matrix': L, 'sparsity_param': 9.5}}
    strategy.fit_regime_models(active_returns, market_returns, sjm_config=cfg)

    model = strategy.regime_models.get('QUAL')
    assert model is not None
    np.testing.assert_allclose(model.jump_penalty_matrix, np.array(L, dtype=float))
    assert model.regimes_ is not None and len(model.regimes_) == n


def _run_all():
    """Run tests without pytest."""
    logging.disable(logging.CRITICAL)
    for _n in ('helix_factor_strategy', 'data.market_data', 'yfinance'):
        logging.getLogger(_n).setLevel(logging.WARNING)

    test_regimes_differ_with_lambda_and_kappa()
    print("test_regimes_differ_with_lambda_and_kappa: OK")
    test_sparsity_param_affects_feature_weights()
    print("test_sparsity_param_affects_feature_weights: OK")
    test_sjm_config_passed_to_model()
    print("test_sjm_config_passed_to_model: OK")
    test_symmetric_matrix_equivalent_to_scalar_jump_penalty()
    print("test_symmetric_matrix_equivalent_to_scalar_jump_penalty: OK")
    test_optimize_regimes_dp_matches_bruteforce_asymmetric()
    print("test_optimize_regimes_dp_matches_bruteforce_asymmetric: OK")
    test_sjm_config_jump_penalty_matrix()
    print("test_sjm_config_jump_penalty_matrix: OK")
    print("All JOH-8 tests passed.")


if __name__ == '__main__':
    _run_all()

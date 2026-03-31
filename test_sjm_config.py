#!/usr/bin/env python3
"""Debug JOH-8: Verify SJM hyperparameters affect regime outputs.

Uses synthetic data - no network required.
"""

import numpy as np
import pandas as pd

# Silence logs
import logging
logging.basicConfig(level=logging.WARNING)
for _n in ('helix_factor_strategy', 'data.market_data', 'yfinance'):
    logging.getLogger(_n).setLevel(logging.WARNING)

# Import SparseJumpModel directly to test in isolation
from helix_factor_strategy import SparseJumpModel


def make_synthetic_returns(n=500, seed=42):
    """Synthetic active returns with some regime structure."""
    np.random.seed(seed)
    dates = pd.date_range('2020-01-01', periods=n, freq='B')
    # Two regimes: low mean vs high mean
    regime = np.zeros(n)
    regime[200:400] = 1
    mu = np.where(regime == 0, -0.0002, 0.0003)
    ret = mu + 0.01 * np.random.randn(n)
    return pd.Series(ret, index=dates)


def main():
    print("=== JOH-8: SJM hyperparameter convergence debug ===\n")

    ar = make_synthetic_returns(500)

    # Test 1: Different (lam, k2) should produce different regime sequences
    configs = [
        (5.0, 2.0, "low lambda, low kappa"),
        (50.0, 9.5, "paper defaults"),
        (150.0, 18.0, "high lambda, high kappa"),
    ]

    regimes_list = []

    for lam, k2, label in configs:
        model = SparseJumpModel(
            n_regimes=2,
            jump_penalty=lam,
            sparsity_param=k2,
            max_iter=100,
        )
        model.fit(ar)  # Uses legacy features (no market data)
        regimes = model.regimes_
        regimes_list.append((label, regimes))
        n0, n1 = (regimes == 0).sum(), (regimes == 1).sum()
        n_changes = (regimes.diff() != 0).sum()
        print(f"Config {label}: λ={lam}, κ²={k2}")
        print(f"  Regime 0: {n0}, Regime 1: {n1}, Changes: {n_changes}")
        print(f"  Feature weights sum: {model.feature_weights_.sum():.4f}")
        print()

    # Check if regimes differ
    r0, r1, r2 = [x[1] for x in regimes_list]
    diff_01 = (r0 != r1).sum()
    diff_12 = (r1 != r2).sum()
    diff_02 = (r0 != r2).sum()

    print("=== Regime sequence differences ===")
    print(f"  Config 0 vs 1: {diff_01} differing observations")
    print(f"  Config 1 vs 2: {diff_12} differing observations")
    print(f"  Config 0 vs 2: {diff_02} differing observations")

    if diff_01 == 0 and diff_12 == 0 and diff_02 == 0:
        print("\n*** BUG: All configs produce identical regimes! Hyperparams have no effect. ***")
        return 1
    else:
        print("\n*** OK: Different hyperparams produce different regimes. ***")

    # Test 2: Verify sparsity_param affects feature weights
    print("\n=== Verifying sparsity_param affects weights ===")
    m_lo = SparseJumpModel(jump_penalty=50, sparsity_param=2)
    m_hi = SparseJumpModel(jump_penalty=50, sparsity_param=18)
    m_lo.fit(ar)
    m_hi.fit(ar)
    nz_lo = (np.abs(m_lo.feature_weights_) > 1e-6).sum()
    nz_hi = (np.abs(m_hi.feature_weights_) > 1e-6).sum()
    print(f"  sparsity_param=2:  {nz_lo} non-zero weights")
    print(f"  sparsity_param=18: {nz_hi} non-zero weights")
    if nz_lo == nz_hi and nz_lo == m_lo.feature_weights_.shape[0]:
        print("  *** BUG: sparsity_param has no effect on weight sparsity! ***")
    else:
        print("  *** OK: sparsity_param affects feature selection. ***")

    return 0


if __name__ == '__main__':
    exit(main())

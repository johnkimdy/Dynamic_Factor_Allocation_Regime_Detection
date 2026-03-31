#!/usr/bin/env python3
"""
Diagnose SJM training loss curves and covariance matrix.

Usage:
  python scripts/diagnose_training.py [--plot-dir OUT_DIR] [--start YYYY-MM-DD] [--end YYYY-MM-DD]

- Fits SJMs with record_loss_curve=True and optionally plots loss curves per factor.
- Runs one optimize_portfolio call with diagnose_cov=True and prints covariance diagnostics.
"""

import argparse
import os
import sys

if os.path.dirname(os.path.abspath(__file__)) != os.getcwd():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from helix_factor_strategy import (
    HelixFactorStrategy,
    compute_active_returns,
    MARKET_ETF,
)


def main():
    ap = argparse.ArgumentParser(description="SJM loss curves and covariance diagnostics")
    ap.add_argument("--plot-dir", default=None, help="Directory to save loss-curve plots (default: no plot)")
    ap.add_argument("--start", default="2020-01-01", help="Start date for data")
    ap.add_argument("--end", default="2024-12-31", help="End date for data")
    args = ap.parse_args()

    strategy = HelixFactorStrategy(
        diagnose_cov=True,
        record_sjm_loss_curve=True,
    )
    strategy.fetch_data(args.start, args.end)
    returns = strategy.calculate_returns()
    active_returns = compute_active_returns(returns, market_col=MARKET_ETF)
    market_returns = returns[MARKET_ETF]

    print("Fitting SJMs with record_loss_curve=True...")
    strategy.fit_regime_models(active_returns, market_returns)

    # Loss curves
    print("\n--- SJM loss curves (final objective, iterations) ---")
    for factor, model in strategy.regime_models.items():
        curve = getattr(model, "loss_curve_", [])
        if curve:
            print(f"  {factor}: {len(curve)} iters, final={curve[-1]:.4f}, min={min(curve):.4f}")
        else:
            print(f"  {factor}: no curve recorded")

    if args.plot_dir:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            os.makedirs(args.plot_dir, exist_ok=True)
            for factor, model in strategy.regime_models.items():
                curve = getattr(model, "loss_curve_", [])
                if not curve:
                    continue
                plt.figure(figsize=(6, 3))
                plt.plot(curve, color="steelblue")
                plt.xlabel("Iteration")
                plt.ylabel("SJM objective")
                plt.title(f"SJM training loss: {factor}")
                plt.tight_layout()
                path = os.path.join(args.plot_dir, f"sjm_loss_{factor}.png")
                plt.savefig(path, dpi=120)
                plt.close()
                print(f"  Saved {path}")
        except ImportError:
            print("  matplotlib not available; skip plots")

    # Covariance check (triggers via optimize_portfolio with diagnose_cov=True)
    print("\n--- Covariance matrix diagnostics (one BL call) ---")
    recent = returns.iloc[-252:] if len(returns) >= 252 else returns
    current_regimes = {f: 0 for f in active_returns.columns}
    expected = strategy.generate_expected_returns(current_regimes)
    strategy.optimize_portfolio(recent, expected)
    print("  (See log above for BL covariance diagnostics)")


if __name__ == "__main__":
    main()

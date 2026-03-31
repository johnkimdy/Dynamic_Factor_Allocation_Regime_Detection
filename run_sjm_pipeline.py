#!/usr/bin/env python3
"""
JOH-9: Production pipeline for SJM hyperparameter tuning.

End-to-end: weekly tune → store params → train prod models → inference + validate → rebalance decision.

Usage:
  python run_sjm_pipeline.py                    # full pipeline
  python run_sjm_pipeline.py --step tune        # tuning only
  python run_sjm_pipeline.py --step store       # store latest CSV to param store
  python run_sjm_pipeline.py --step train       # train prod models with stored params
  python run_sjm_pipeline.py --step validate    # validate inference on latest data
  python run_sjm_pipeline.py --params-source local --params-version production  # use prod params
"""

import argparse
import logging
import os
import sys

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Silence noisy loggers
for _n in ("helix_factor_strategy", "data.market_data", "yfinance", "urllib3", "optuna"):
    logging.getLogger(_n).setLevel(logging.WARNING)


def step_tune(n_trials=50, use_wandb=None):
    """Run Optuna tuning per factor."""
    from tune_sjm_hyperparameters import tune_hyperparameters

    use_wb = use_wandb if use_wandb is not None else (os.environ.get("WANDB_MODE") != "disabled")
    results = tune_hyperparameters(n_trials_per_factor=n_trials, use_wandb=use_wb)
    return results


def step_store(results=None, use_wandb=True, use_local=True):
    """Store params to param store (local + optional W&B). Returns (results, local_version)."""
    from sjm_params_store import save

    if results is None:
        # Load from hyperparam/ (best or latest or legacy CSV)
        from hyperparam_io import load_results_for_store
        results = load_results_for_store()
        if not results:
            raise FileNotFoundError(
                "No results and no hyperparam runs found. Run --step tune first."
            )
    local_ver, wb_ver = save(
        results, metadata={"source": "run_sjm_pipeline"}, use_wandb=use_wandb, use_local=use_local
    )
    logger.info("Stored: local=%s, wandb=%s", local_ver, wb_ver)
    return results, local_ver


def step_train(params_source="local", params_version="latest"):
    """Train production SJM models with stored params."""
    from helix_factor_strategy import HelixFactorStrategy, compute_active_returns, MARKET_ETF
    from sjm_params_store import load

    sjm_config = load(source=params_source, version_or_alias=params_version)
    strategy = HelixFactorStrategy(lookback_days=252 * 10)
    # Use reasonable date range for prod
    strategy.fetch_data("2014-01-01", "2025-12-31")
    returns = strategy.calculate_returns()
    active_returns = compute_active_returns(returns, market_col=MARKET_ETF)
    market_returns = returns[MARKET_ETF]
    strategy.fit_regime_models(active_returns, market_returns, sjm_config=sjm_config)
    logger.info("Trained prod models with params from %s:%s", params_source, params_version)
    return strategy


def load_prod_params(source="local", version="production"):
    """Load production params for backtest/analysis. Returns sjm_config dict or None for paper defaults."""
    from sjm_params_store import load

    try:
        return load(source=source, version_or_alias=version)
    except Exception as e:
        logger.warning("Could not load params from %s:%s: %s", source, version, e)
        return None


def step_validate(params_source="local", params_version="latest"):
    """Validate regime inference on latest data. Returns (ok, message)."""
    from helix_factor_strategy import HelixFactorStrategy, compute_active_returns, MARKET_ETF
    from sjm_params_store import load

    sjm_config = load(source=params_source, version_or_alias=params_version)
    strategy = HelixFactorStrategy(lookback_days=252 * 10)
    strategy.fetch_data("2019-01-01", "2025-12-31")
    returns = strategy.calculate_returns()
    active_returns = compute_active_returns(returns, market_col=MARKET_ETF)
    market_returns = returns[MARKET_ETF]
    strategy.fit_regime_models(active_returns, market_returns, sjm_config=sjm_config)

    # Get latest regimes and expected returns
    current_regimes = {}
    for factor in strategy.regime_models:
        rs = strategy.regime_models[factor].regimes_
        current_regimes[factor] = int(rs.iloc[-1]) if rs is not None and len(rs) > 0 else 0
    expected = strategy.generate_expected_returns(current_regimes)
    ok, msg = strategy.validate_regime_inference(current_regimes, expected)
    logger.info("Validation: %s - %s", "PASS" if ok else "FAIL", msg)
    return ok, msg


def step_promote(version):
    """Promote a version to production (rollback target)."""
    from sjm_params_store import promote_local

    promote_local(version)
    logger.info("Promoted %s to production", version)


def main():
    parser = argparse.ArgumentParser(description="SJM production pipeline (JOH-9)")
    parser.add_argument(
        "--step",
        choices=["tune", "store", "train", "validate", "promote", "all"],
        default="all",
        help="Pipeline step to run",
    )
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna trials per factor (tune)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B")
    parser.add_argument(
        "--params-source",
        choices=["local", "wandb", "paper"],
        default="local",
        help="Param store source (train/validate)",
    )
    parser.add_argument(
        "--params-version",
        default="latest",
        help="Param version or alias: latest, production, or version string",
    )
    parser.add_argument("--promote-version", help="Version to promote to production")
    args = parser.parse_args()

    if args.step == "promote":
        if not args.promote_version:
            parser.error("--promote-version required for promote step")
        step_promote(args.promote_version)
        return 0

    if args.step == "tune":
        step_tune(n_trials=args.n_trials, use_wandb=not args.no_wandb)
        return 0

    if args.step == "store":
        step_store(use_wandb=not args.no_wandb, use_local=True)
        return 0

    if args.step == "train":
        step_train(params_source=args.params_source, params_version=args.params_version)
        return 0

    if args.step == "validate":
        ok, _ = step_validate(
            params_source=args.params_source, params_version=args.params_version
        )
        return 0 if ok else 1

    if args.step == "all":
        # tune → store → train → validate; promote on success, rollback on fail
        results = step_tune(n_trials=args.n_trials, use_wandb=not args.no_wandb)
        _, local_ver = step_store(results=results, use_wandb=not args.no_wandb)
        step_train(params_source="local", params_version="latest")
        ok, msg = step_validate(params_source="local", params_version="latest")
        if not ok:
            logger.warning(
                "Validation failed. Do NOT promote. Rollback: run_sjm_pipeline.py --step promote --promote-version <prev_version>"
            )
            return 1
        if local_ver:
            step_promote(local_ver)
        logger.info("Full pipeline completed successfully")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

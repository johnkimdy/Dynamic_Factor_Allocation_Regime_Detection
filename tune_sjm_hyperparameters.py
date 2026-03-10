#!/usr/bin/env python3
"""
Phase 6: Single-factor long-short strategy for SJM hyperparameter tuning.
Per paper: tune λ (jump_penalty) and κ² (sparsity_param) per factor to maximize
long-short Sharpe. Linear position: long when E[active]>5%, short when <-5%.

Paper alignment (Exhibit 1): Data horizon 1993-2024, test period 2007-2024.
"""

import numpy as np
import pandas as pd
import logging
import os
from itertools import product

from helix_factor_strategy import (
    HelixFactorStrategy,
    compute_active_returns,
    DEFAULT_JUMP_PENALTY,
    DEFAULT_SPARSITY_PARAM,
    EXPECTED_RETURN_CAP,
    MARKET_ETF,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

FACTORS = ['QUAL', 'MTUM', 'USMV', 'VLUE', 'SIZE', 'IWF']

# Paper horizons (Exhibit 1)
DATA_START = '1993-01-01'
DATA_END = '2024-12-31'
TEST_START = '2007-01-01'
TEST_END = '2024-12-31'

# Wider grid per user request
LAMBDA_GRID = (5, 10, 30, 50, 70, 100)
KAPPA_GRID = (2, 4, 6.5, 9.5, 12, 15)

WANDB_PROJECT = 'helix-sjm-tuning'


def long_short_position(expected_active_ann):
    """
    Linear position per paper: long when >5%, short when <-5%, linear between.
    Returns weight in [-1, 1] for long-short.
    """
    cap = EXPECTED_RETURN_CAP
    if expected_active_ann >= cap:
        return 1.0
    if expected_active_ann <= -cap:
        return -1.0
    return expected_active_ann / cap


def compute_factor_long_short_sharpe(active_returns, regime_means, regimes, factor, period=None):
    """Sharpe of hypothetical long-short strategy for one factor on given period."""
    common = active_returns.index.intersection(regimes.index)
    ar = active_returns.loc[common, factor]
    if period is not None:
        ar = ar.loc[period[0]:period[1]]
    reg = regimes.reindex(ar.index).ffill().fillna(0)
    pos = reg.apply(lambda r: long_short_position(regime_means.get(int(r), 0))).shift(2)
    strategy_returns = (pos * ar).dropna()
    if len(strategy_returns) < 20 or strategy_returns.std() < 1e-10:
        return 0.0
    return float(strategy_returns.mean() / strategy_returns.std() * np.sqrt(252))


def tune_hyperparameters(
    start_date=DATA_START,
    end_date=DATA_END,
    test_start=TEST_START,
    test_end=TEST_END,
    lambda_grid=LAMBDA_GRID,
    kappa_grid=KAPPA_GRID,
    use_wandb=True,
):
    """
    Grid search λ and κ² per factor. Validates on test period 2007-2024.
    Logs to Weights & Biases if use_wandb=True and WANDB_API_KEY is set.
    """
    run = None
    if use_wandb:
        try:
            import wandb
            wandb.login()
            run = wandb.init(
                project=WANDB_PROJECT,
                config={
                    'data_start': start_date,
                    'data_end': end_date,
                    'test_start': test_start,
                    'test_end': test_end,
                    'lambda_grid': list(lambda_grid),
                    'kappa_grid': list(kappa_grid),
                },
                tags=['sjm', 'hyperparameter-tuning'],
            )
        except Exception as e:
            logger.warning("W&B init failed (%s), continuing without logging", e)
            run = None

    strategy = HelixFactorStrategy(lookback_days=252 * 10)
    strategy.fetch_data(start_date, end_date)
    returns = strategy.calculate_returns()
    active_returns = compute_active_returns(returns, market_col=MARKET_ETF)
    market_returns = returns[MARKET_ETF]

    test_period = (test_start, test_end)
    results = {}
    total_combos = len(FACTORS) * len(lambda_grid) * len(kappa_grid)
    combo = 0

    for factor in FACTORS:
        best_sharpe = -np.inf
        best_params = (DEFAULT_JUMP_PENALTY, DEFAULT_SPARSITY_PARAM)
        for lam, k2 in product(lambda_grid, kappa_grid):
            combo += 1
            cfg = {factor: {'jump_penalty': lam, 'sparsity_param': k2}}
            strategy.fit_regime_models(active_returns, market_returns, sjm_config=cfg)
            model = strategy.regime_models.get(factor)
            if model is None or model.regimes_ is None:
                continue
            sharpe = compute_factor_long_short_sharpe(
                active_returns, strategy.regime_means[factor], model.regimes_, factor, period=test_period
            )
            if run:
                run.log({
                    'factor': factor,
                    'lambda': lam,
                    'kappa_sq': k2,
                    'long_short_sharpe': sharpe,
                    'progress': combo / total_combos,
                }, step=combo)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = (lam, k2)
        results[factor] = {'lambda': best_params[0], 'kappa_sq': best_params[1], 'sharpe': best_sharpe}
        print("{}: λ={}, κ²={}, Sharpe={:.3f}".format(factor, *best_params, best_sharpe))


    if run:
        mean_sharpe = np.mean([r['sharpe'] for r in results.values()])
        run.log({'mean_sharpe_across_factors': mean_sharpe})
        for f in FACTORS:
            r = results[f]
            run.config[f'best_{f}_lambda'] = r['lambda']
            run.config[f'best_{f}_kappa_sq'] = r['kappa_sq']
            run.config[f'best_{f}_sharpe'] = r['sharpe']
        df = pd.DataFrame(results).T
        out_path = 'sjm_hyperparameters.csv'
        df.to_csv(out_path)
        run.log_artifact(out_path, name='sjm_hyperparameters', type='hyperparameters')
        run.finish()

    return results


if __name__ == '__main__':
    use_wb = os.environ.get('WANDB_API_KEY') or os.environ.get('WANDB_MODE') != 'disabled'
    print("Tuning SJM hyperparameters (Phase 6)...")
    print("Data: {} to {}, Test period: {} to {}".format(DATA_START, DATA_END, TEST_START, TEST_END))
    print("Grid: λ={}, κ²={}".format(LAMBDA_GRID, KAPPA_GRID))
    print("W&B logging: {}".format(use_wb))
    results = tune_hyperparameters(use_wandb=use_wb)
    print("\nBest hyperparameters per factor:")
    for f, r in results.items():
        print("  {}: jump_penalty={}, sparsity_param={}, long_short_sharpe={:.3f}".format(
            f, r['lambda'], r['kappa_sq'], r['sharpe']))
    pd.DataFrame(results).T.to_csv('sjm_hyperparameters.csv')
    print("\nSaved to sjm_hyperparameters.csv")

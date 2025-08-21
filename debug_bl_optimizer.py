#!/usr/bin/env python3.11
"""
Debug Black-Litterman Optimizer to fix the 0% returns issue
"""

import numpy as np
import pandas as pd
import yfinance as yf
from helix_factor_strategy import BlackLittermanOptimizer, HelixFactorStrategy
import logging

logging.basicConfig(level=logging.INFO)

def test_bl_optimizer_directly():
    """Test the Black-Litterman optimizer in isolation"""
    
    print("=== DEBUGGING BLACK-LITTERMAN OPTIMIZER ===\n")
    
    # Create sample data
    etfs = ['SPY', 'QUAL', 'MTUM', 'USMV', 'VLUE', 'SIZE', 'IWF']
    
    # Get real data for covariance matrix
    print("Fetching sample data...")
    data = yf.download(etfs, start='2024-01-01', end='2024-06-30', auto_adjust=True)['Close']
    returns = data.pct_change().dropna()
    cov_matrix = returns.cov()
    
    print("Data shape: {}".format(returns.shape))
    print("Covariance matrix shape: {}".format(cov_matrix.shape))
    
    # Create Black-Litterman optimizer
    optimizer = BlackLittermanOptimizer(risk_aversion=3.0, tau=0.025)
    
    # Test Case 1: Simple expected returns (all factors positive)
    print("\n--- Test Case 1: All positive expected returns ---")
    expected_returns = {
        'QUAL': 0.02,   # 2% expected return
        'MTUM': 0.015,  # 1.5% expected return  
        'USMV': 0.01,   # 1% expected return
        'VLUE': 0.025,  # 2.5% expected return
        'SIZE': 0.008,  # 0.8% expected return
        'IWF': 0.018    # 1.8% expected return
    }
    
    try:
        weights = optimizer.optimize(expected_returns, cov_matrix)
        print("Optimization successful!")
        print("Weights sum: {:.6f}".format(weights.sum()))
        print("Weights:")
        for etf, weight in weights.items():
            print("  {}: {:.4f} ({:.2%})".format(etf, weight, weight))
        
        # Check if weights are reasonable
        if abs(weights.sum() - 1.0) > 0.01:
            print("WARNING: Weights don't sum to 1!")
        if (weights < 0).any():
            print("WARNING: Negative weights found!")
        if weights.max() > 0.8:
            print("WARNING: Highly concentrated allocation!")
            
    except Exception as e:
        print("Optimization FAILED: {}".format(e))
        import traceback
        traceback.print_exc()
    
    # Test Case 2: Zero expected returns
    print("\n--- Test Case 2: Zero expected returns ---")
    expected_returns_zero = {etf: 0.0 for etf in etfs[1:]}  # Exclude SPY
    
    try:
        weights = optimizer.optimize(expected_returns_zero, cov_matrix)
        print("Zero returns case - Weights sum: {:.6f}".format(weights.sum()))
        print("Should default to equal weights...")
        
    except Exception as e:
        print("Zero returns case FAILED: {}".format(e))
    
    # Test Case 3: Mixed expected returns (some negative)
    print("\n--- Test Case 3: Mixed expected returns ---")
    expected_returns_mixed = {
        'QUAL': 0.01,
        'MTUM': -0.005,  # Negative expected return
        'USMV': 0.015,
        'VLUE': -0.01,   # Negative expected return
        'SIZE': 0.005,
        'IWF': 0.02
    }
    
    try:
        weights = optimizer.optimize(expected_returns_mixed, cov_matrix)
        print("Mixed returns case - Weights sum: {:.6f}".format(weights.sum()))
        print("Weights with mixed expected returns:")
        for etf, weight in weights.items():
            print("  {}: {:.4f} ({:.2%})".format(etf, weight, weight))
            
    except Exception as e:
        print("Mixed returns case FAILED: {}".format(e))

def test_strategy_expected_returns():
    """Test the expected returns generation from the strategy"""
    
    print("\n=== DEBUGGING EXPECTED RETURNS GENERATION ===\n")
    
    strategy = HelixFactorStrategy()
    
    # Fetch some real data
    data = strategy.fetch_data('2024-01-01', '2024-03-31')
    returns = strategy.calculate_returns()
    
    print("Returns data shape: {}".format(returns.shape))
    print("Sample returns (last 5 days):")
    print(returns.tail())
    
    # Fit regime models
    strategy.fit_regime_models(returns)
    
    print("\nRegime models fitted:")
    for etf, model in strategy.regime_models.items():
        if hasattr(model, 'regimes_') and model.regimes_ is not None:
            unique_regimes = model.regimes_.nunique()
            latest_regime = model.regimes_.iloc[-1]
            print("  {}: {} regimes detected, current regime: {}".format(
                etf, unique_regimes, latest_regime))
    
    # Test expected returns generation
    recent_returns = returns.tail(60)  # Last 60 days
    
    # Get current regimes (last day)
    current_regimes = {}
    for etf in strategy.factor_etfs.keys():
        if etf in strategy.regime_models:
            regime_series = strategy.regime_models[etf].regimes_
            current_regimes[etf] = regime_series.iloc[-1]
        else:
            current_regimes[etf] = 0
    
    print("\nCurrent regimes:")
    for etf, regime in current_regimes.items():
        print("  {}: regime {}".format(etf, regime))
    
    # Generate expected returns
    expected_returns = strategy.generate_expected_returns(recent_returns, current_regimes)
    
    print("\nGenerated expected returns:")
    for etf, exp_ret in expected_returns.items():
        print("  {}: {:.4f} ({:.2%} annualized)".format(etf, exp_ret, exp_ret * 252))
    
    # Check if expected returns are reasonable
    if expected_returns.isna().any():
        print("WARNING: NaN expected returns detected!")
    if (expected_returns == 0).all():
        print("WARNING: All expected returns are zero!")
    if expected_returns.abs().max() > 0.01:  # More than 1% daily return
        print("WARNING: Very high expected returns detected!")
    
    return expected_returns, recent_returns

def test_full_optimization_chain():
    """Test the complete optimization chain"""
    
    print("\n=== TESTING FULL OPTIMIZATION CHAIN ===\n")
    
    expected_returns, recent_returns = test_strategy_expected_returns()
    
    # Calculate covariance matrix
    cov_matrix = recent_returns.cov()
    print("Covariance matrix condition number: {:.2e}".format(
        np.linalg.cond(cov_matrix.values)))
    
    # Test the full optimization
    strategy = HelixFactorStrategy()
    
    try:
        weights = strategy.optimize_portfolio(recent_returns, expected_returns)
        print("Full optimization successful!")
        print("Final weights:")
        for etf, weight in weights.items():
            print("  {}: {:.4f} ({:.2%})".format(etf, weight, weight))
        
        # Check if this would trigger rebalancing
        strategy.current_weights = None  # Start fresh
        should_rebalance = strategy.should_rebalance(weights)
        print("Should rebalance: {}".format(should_rebalance))
        
        # Test rebalance threshold
        if should_rebalance:
            strategy.current_weights = weights.copy()
            # Create slightly different weights
            new_weights = weights * 1.01  # 1% change
            new_weights = new_weights / new_weights.sum()  # Normalize
            
            should_rebalance_small = strategy.should_rebalance(new_weights)
            print("Should rebalance with 1% change: {}".format(should_rebalance_small))
            
    except Exception as e:
        print("Full optimization FAILED: {}".format(e))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_bl_optimizer_directly()
    test_full_optimization_chain()
#!/usr/bin/env python3.11
"""
Comprehensive analysis of Helix 1.1 Factor Strategy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helix_factor_strategy import HelixFactorStrategy
import logging
import yfinance as yf

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_analysis():
    """Run comprehensive analysis of the Helix 1.1 strategy"""
    
    strategy = HelixFactorStrategy()
    
    # Run multiple test periods including August 2025
    test_periods = [
        ('2022-01-01', '2023-12-31', '2022-2023'),
        ('2021-01-01', '2023-12-31', '2021-2023'),
        ('2023-01-01', '2024-12-31', '2023-2024'),
        ('2024-01-01', '2025-08-31', '2024-2025'),
        ('2022-01-01', '2025-08-31', '2022-2025')
    ]
    
    results_summary = []
    
    print("=" * 60)
    print("HELIX 1.1 FACTOR STRATEGY ANALYSIS")
    print("=" * 60)
    
    for start_date, end_date, period_name in test_periods:
        print("\n" + "-" * 40)
        print("Testing Period: {} ({} to {})".format(period_name, start_date, end_date))
        print("-" * 40)
        
        try:
            results = strategy.backtest(start_date, end_date)
            
            # Extract key metrics
            metrics = {
                'period': period_name,
                'start_date': start_date,
                'end_date': end_date,
                'total_return': results['total_return'],
                'sharpe_ratio': results['sharpe_ratio'],
                'volatility': results['annualized_volatility'],
                'max_drawdown': results['max_drawdown'],
                'n_rebalances': results['n_rebalances']
            }
            
            results_summary.append(metrics)
            
            # Print results
            print("Total Return: {:.2%}".format(results['total_return']))
            print("Sharpe Ratio: {:.2f}".format(results['sharpe_ratio']))
            print("Annualized Volatility: {:.2%}".format(results['annualized_volatility']))
            print("Max Drawdown: {:.2%}".format(results['max_drawdown']))
            print("Number of Rebalances: {}".format(results['n_rebalances']))
            
            # Calculate annualized return
            portfolio_values = results['portfolio_values']
            n_days = len(portfolio_values)
            years = n_days / 252.0
            annualized_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (1/years) - 1
            print("Annualized Return: {:.2%}".format(annualized_return))
            
            # Calculate active return metrics relative to SPY (simplified)
            try:
                import yfinance as yf
                spy_data = yf.download('SPY', start=start_date, end=end_date, auto_adjust=True)['Close']
                
                # Calculate total returns
                spy_total_return = (spy_data.iloc[-1] / spy_data.iloc[0]) - 1
                strategy_total_return = results['total_return']
                
                # Calculate annualized returns
                spy_annualized = (1 + spy_total_return) ** (1/years) - 1
                strategy_annualized = annualized_return
                
                # Active return = strategy return - benchmark return
                active_return = strategy_annualized - spy_annualized
                
                # Simplified information ratio estimate using return difference and volatility
                # This is approximate since we don't have daily active returns
                information_ratio = active_return / results['annualized_volatility'] if results['annualized_volatility'] > 0 else 0
                
                print("Active Return (vs SPY): {:.2%}".format(float(active_return)))
                print("Information Ratio (approx): {:.2f}".format(float(information_ratio)))
                
                # Store in metrics for summary table
                metrics['active_return'] = active_return
                metrics['information_ratio'] = information_ratio
                metrics['active_volatility'] = results['annualized_volatility']
                    
            except Exception as e:
                print("Could not calculate active return metrics: {}".format(e))
                metrics['active_return'] = 0
                metrics['information_ratio'] = 0
                metrics['active_volatility'] = 0
            
        except Exception as e:
            print("Error during backtest: {}".format(e))
            continue
    
    # Create summary table
    if results_summary:
        print("\n" + "=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)
        
        df_results = pd.DataFrame(results_summary)
        
        # Format for display
        print("Period       | Total Ret | Sharpe | Volatility | Max DD | Active Ret | IR | Rebalances")
        print("-" * 90)
        for _, row in df_results.iterrows():
            print("{:<12} | {:>8.2%} | {:>6.2f} | {:>9.2%} | {:>6.2%} | {:>9.2%} | {:>10.2f} | {:>10}".format(
                str(row['period']),
                float(row['total_return']),
                float(row['sharpe_ratio']),
                float(row['volatility']),
                float(row['max_drawdown']),
                float(row.get('active_return', 0)),
                float(row.get('information_ratio', 0)),
                int(row['n_rebalances'])
            ))
    
    # Benchmark comparison
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON (SPY)")
    print("=" * 80)
    
    try:
        import yfinance as yf
        
        # Get SPY benchmark for 2022-2023
        spy_data = yf.download('SPY', start='2024-01-01', end='2025-08-31', auto_adjust=True)['Close']
        spy_returns = spy_data.pct_change().dropna()
        
        spy_total_return = (spy_data.iloc[-1] / spy_data.iloc[0]) - 1
        spy_volatility = spy_returns.std() * np.sqrt(252)
        spy_sharpe = spy_returns.mean() / spy_returns.std() * np.sqrt(252)
        
        # Calculate SPY max drawdown
        spy_cumulative = spy_data / spy_data.iloc[0]
        spy_running_max = spy_cumulative.expanding().max()
        spy_drawdown = (spy_cumulative - spy_running_max) / spy_running_max
        spy_max_dd = spy_drawdown.min()
        
        print("SPY (2024-2025):")
        print("Total Return: {:.2%}".format(float(spy_total_return)))
        print("Sharpe Ratio: {:.2f}".format(float(spy_sharpe)))
        print("Volatility: {:.2%}".format(float(spy_volatility)))
        print("Max Drawdown: {:.2%}".format(float(spy_max_dd)))
        
        # Compare with our strategy (2022-2023 period)
        strategy_2022_2023 = [r for r in results_summary if r['period'] == '2024-2025']
        if strategy_2022_2023:
            strat = strategy_2022_2023[0]
            print("\nHelix 1.1 vs SPY (2024-2025):")
            print("Return Difference: {:.2%}".format(float(strat['total_return'] - spy_total_return)))
            print("Sharpe Difference: {:.2f}".format(float(strat['sharpe_ratio'] - spy_sharpe)))
            print("Volatility Difference: {:.2%}".format(float(strat['volatility'] - spy_volatility)))
        
    except Exception as e:
        print("Error getting SPY benchmark: {}".format(e))
    
    return results_summary

if __name__ == "__main__":
    results = run_analysis()
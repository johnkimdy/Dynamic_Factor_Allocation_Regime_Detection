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
    
    # Test periods: grouped by ending year, sorted by starting year desc within each group
    # Order: 2024-2025, 2023-2025, 2022-2025, ..., 2017-2025 | 2023-2024, 2022-2024, ... | 2022-2023, 2021-2023
    _raw_periods = [
        # Ending 2025
        ('2024-01-01', '2025-08-31', '2024-2025'),
        ('2023-01-01', '2025-08-31', '2023-2025'),
        ('2022-01-01', '2025-08-31', '2022-2025'),
        ('2021-01-01', '2025-08-31', '2021-2025'),
        ('2020-01-01', '2025-08-31', '2020-2025'),
        ('2019-01-01', '2025-08-31', '2019-2025'),
        ('2018-01-01', '2025-08-31', '2018-2025'),
        ('2017-01-01', '2025-08-31', '2017-2025'),
        # Ending 2024
        ('2023-01-01', '2024-12-31', '2023-2024'),
        ('2022-01-01', '2024-12-31', '2022-2024'),
        ('2021-01-01', '2024-12-31', '2021-2024'),
        # Ending 2023
        ('2022-01-01', '2023-12-31', '2022-2023'),
        ('2021-01-01', '2023-12-31', '2021-2023'),
    ]
    test_periods = sorted(_raw_periods, key=lambda p: (-int(p[1][:4]), -int(p[0][:4])))
    
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
                spy_raw = yf.download('SPY', start=start_date, end=end_date, auto_adjust=True, progress=False)
                spy_data = spy_raw['Close'].squeeze()
                
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
    
    # Benchmark comparison vs SPY for all timeframes
    print("\n" + "=" * 110)
    print("BENCHMARK COMPARISON vs SPY (ALL PERIODS)")
    print("=" * 110)
    
    try:
        comparison_rows = []
        for strat in results_summary:
            start_date = strat['start_date']
            end_date = strat['end_date']
            period_name = strat['period']
            try:
                spy_raw = yf.download('SPY', start=start_date, end=end_date, auto_adjust=True, progress=False)
                spy_data = spy_raw['Close'].squeeze()
                spy_returns = spy_data.pct_change().dropna()
                
                spy_total_return = float((spy_data.iloc[-1] / spy_data.iloc[0]) - 1)
                spy_volatility = float(spy_returns.std() * np.sqrt(252))
                spy_sharpe = float(spy_returns.mean() / spy_returns.std() * np.sqrt(252))
                
                spy_cumulative = spy_data / spy_data.iloc[0]
                spy_running_max = spy_cumulative.expanding().max()
                spy_drawdown = (spy_cumulative - spy_running_max) / spy_running_max
                spy_max_dd = float(spy_drawdown.min())
                
                comparison_rows.append({
                    'period': period_name,
                    'helix_return': strat['total_return'],
                    'spy_return': spy_total_return,
                    'return_diff': strat['total_return'] - spy_total_return,
                    'helix_sharpe': strat['sharpe_ratio'],
                    'spy_sharpe': spy_sharpe,
                    'sharpe_diff': strat['sharpe_ratio'] - spy_sharpe,
                    'helix_vol': strat['volatility'],
                    'spy_vol': spy_volatility,
                    'vol_diff': strat['volatility'] - spy_volatility,
                    'helix_dd': strat['max_drawdown'],
                    'spy_dd': spy_max_dd,
                })
            except Exception as e:
                print("Could not fetch SPY for {}: {}".format(period_name, e))
        
        if comparison_rows:
            print("\n{:<12} | {:>12} | {:>12} | {:>10} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8}".format(
                "Period", "Helix Ret", "SPY Ret", "Ret Diff", "Helix SR", "SPY SR", "SR Diff", "Helix Vol", "SPY Vol", "Vol Diff"
            ))
            print("-" * 110)
            for r in comparison_rows:
                print("{:<12} | {:>11.2%} | {:>11.2%} | {:>+9.2%} | {:>7.2f} | {:>7.2f} | {:>+7.2f} | {:>7.2%} | {:>7.2%} | {:>+7.2%}".format(
                    r['period'],
                    r['helix_return'],
                    r['spy_return'],
                    r['return_diff'],
                    r['helix_sharpe'],
                    r['spy_sharpe'],
                    r['sharpe_diff'],
                    r['helix_vol'],
                    r['spy_vol'],
                    r['vol_diff']
                ))
            print("\n{:<12} | {:>12} | {:>12} | {:>10}".format("Period", "Helix Max DD", "SPY Max DD", "DD Diff"))
            print("-" * 55)
            for r in comparison_rows:
                dd_diff = r['helix_dd'] - r['spy_dd']
                print("{:<12} | {:>11.2%} | {:>11.2%} | {:>+9.2%}".format(
                    r['period'], r['helix_dd'], r['spy_dd'], dd_diff
                ))
        
    except Exception as e:
        print("Error getting SPY benchmark: {}".format(e))
    
    return results_summary

if __name__ == "__main__":
    results = run_analysis()
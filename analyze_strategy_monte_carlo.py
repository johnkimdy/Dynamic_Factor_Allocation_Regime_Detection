#!/usr/bin/env python
"""
Monte Carlo Bootstrap Analysis for Helix 1.1 Factor Strategy

This script runs bootstrap simulations to generate confidence intervals
for strategy performance metrics.

Usage:
    # Run with all CPU cores (default)
    python analyze_strategy_monte_carlo.py

    # Run sequentially (1 core)
    python -c "from analyze_strategy_monte_carlo import run_analysis; run_analysis(n_jobs=1)"

    # Run with specific number of cores
    python -c "from analyze_strategy_monte_carlo import run_analysis; run_analysis(n_jobs=4)"

    # Run with custom number of simulations
    python -c "from analyze_strategy_monte_carlo import run_analysis; run_analysis(n_simulations=500)"

Expected speedup with parallel execution:
    - 8 cores: ~6-7x faster
    - 4 cores: ~3-4x faster
    - 2 cores: ~1.8-2x faster
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helix_factor_strategy_fixed import HelixFactorStrategyFixed
import logging
from tqdm import tqdm
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise during Monte Carlo
logger = logging.getLogger(__name__)


class MonteCarloBacktest:
    """
    Monte Carlo bootstrap analysis for portfolio strategy
    """

    def __init__(self, strategy, n_simulations=100, random_seed=42, n_jobs=-1):
        """
        Initialize Monte Carlo backtester

        Args:
            strategy: HelixFactorStrategy instance
            n_simulations: Number of bootstrap simulations to run
            random_seed: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 = use all CPUs, 1 = sequential)
        """
        self.strategy = strategy
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()
        np.random.seed(random_seed)

    def bootstrap_returns(self, returns_df):
        """
        Bootstrap resample daily returns with replacement

        Args:
            returns_df: DataFrame of daily returns (index=dates, columns=tickers)

        Returns:
            Bootstrapped returns DataFrame with same structure
        """
        n_days = len(returns_df)

        # Sample with replacement
        bootstrap_indices = np.random.choice(n_days, size=n_days, replace=True)

        # Create bootstrapped returns while preserving cross-sectional correlation
        # (resample entire rows to keep correlation structure)
        bootstrapped = returns_df.iloc[bootstrap_indices].copy()

        # Reset index to original dates to maintain time structure for strategy
        bootstrapped.index = returns_df.index

        return bootstrapped

    def run_bootstrap_backtest(self, original_data, start_idx, initial_training_days):
        """
        Run a single bootstrap backtest

        Args:
            original_data: Original price DataFrame
            start_idx: Starting index for backtest
            initial_training_days: Initial training period

        Returns:
            Dictionary of performance metrics
        """
        # Calculate returns from original data
        original_returns = original_data.pct_change().dropna()

        # Bootstrap the returns
        bootstrapped_returns = self.bootstrap_returns(original_returns)

        # Store original data
        original_strategy_data = self.strategy.data.copy() if self.strategy.data is not None else None

        try:
            # Reconstruct price data from bootstrapped returns
            # Start with arbitrary prices and compound returns
            initial_prices = 100.0
            bootstrapped_prices = pd.DataFrame(
                index=bootstrapped_returns.index,
                columns=bootstrapped_returns.columns
            )

            bootstrapped_prices.iloc[0] = initial_prices
            for i in range(1, len(bootstrapped_returns)):
                bootstrapped_prices.iloc[i] = bootstrapped_prices.iloc[i-1] * (1 + bootstrapped_returns.iloc[i])

            # Temporarily replace strategy data
            self.strategy.data = bootstrapped_prices

            # Run the backtest logic using walk-forward approach
            results = self._run_single_backtest(bootstrapped_prices, initial_training_days)

            return results

        except Exception as e:
            logger.warning(f"Bootstrap iteration failed: {e}")
            return None
        finally:
            # Restore original data
            self.strategy.data = original_strategy_data

    def _run_single_backtest(self, data, initial_training_days):
        """
        Simplified walk-forward backtest for bootstrap simulation
        Uses the same logic as HelixFactorStrategyFixed.walk_forward_backtest
        """
        returns = data.pct_change().dropna()

        # Initialize tracking
        portfolio_values = []
        rebalance_dates = []

        initial_value = 100000
        current_value = initial_value

        # Reset strategy state
        self.strategy.current_weights = None
        self.strategy.regime_models = {}

        # Start after initial training period
        start_idx = initial_training_days

        for i in range(start_idx, len(returns)):
            current_date = returns.index[i]

            # IMPORTANT: Only use data UP TO time i (no future data!)
            historical_returns = returns.iloc[:i]

            # Retrain/update regime models periodically
            if i == start_idx or i % self.strategy.regime_update_frequency == 0:
                self.strategy._update_regime_models_online(historical_returns)

            # Get current regime for each factor (using only past data)
            current_regimes = {}
            for etf in self.strategy.factor_etfs.keys():
                if etf in self.strategy.regime_models:
                    # Predict regime using only historical data
                    regime = self.strategy.regime_models[etf].regimes_.iloc[-1]
                    current_regimes[etf] = regime
                else:
                    current_regimes[etf] = 0

            # Calculate expected returns using only past data
            recent_returns = historical_returns.iloc[-self.strategy.lookback_days:]
            expected_returns = self.strategy._generate_expected_returns(recent_returns, current_regimes)

            # Optimize portfolio using only past data
            try:
                cov_matrix = recent_returns.cov()
                new_weights = self.strategy.optimizer.optimize(expected_returns, cov_matrix)
            except:
                new_weights = pd.Series(1.0/len(self.strategy.factor_etfs),
                                       index=self.strategy.factor_etfs.keys())

            # Check if rebalancing needed
            if self.strategy._should_rebalance(new_weights):
                self.strategy.current_weights = new_weights.copy()
                rebalance_dates.append(current_date)

            # Calculate portfolio return for TODAY (time i)
            if self.strategy.current_weights is not None:
                daily_returns = returns.iloc[i]
                portfolio_return = (self.strategy.current_weights * daily_returns).sum()
                current_value = current_value * (1 + portfolio_return)

            portfolio_values.append(current_value)

        # Calculate metrics
        results_dates = returns.index[start_idx:]
        portfolio_series = pd.Series(portfolio_values, index=results_dates)

        total_return = (current_value - initial_value) / initial_value
        portfolio_returns = portfolio_series.pct_change().dropna()

        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
                       if portfolio_returns.std() > 0 else 0)
        max_dd = self.strategy._calculate_max_drawdown(portfolio_series)

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility,
            'max_drawdown': max_dd,
            'n_rebalances': len(rebalance_dates),
            'final_value': current_value
        }

    def _parallel_bootstrap_worker(self, iteration, data, start_idx, initial_training_days):
        """
        Worker function for parallel bootstrap execution

        Args:
            iteration: Iteration number (used for seeding)
            data: Price data
            start_idx: Starting index
            initial_training_days: Initial training period

        Returns:
            Bootstrap results dictionary or None
        """
        # Set unique seed for this iteration
        np.random.seed(self.random_seed + iteration)

        # Create a fresh strategy instance for this worker
        # (avoids shared state issues in multiprocessing)
        worker_strategy = HelixFactorStrategyFixed(
            lookback_days=self.strategy.lookback_days,
            rebalance_threshold=self.strategy.rebalance_threshold,
            regime_update_frequency=self.strategy.regime_update_frequency
        )

        # Temporarily swap strategy
        original_strategy = self.strategy
        self.strategy = worker_strategy

        try:
            result = self.run_bootstrap_backtest(data, start_idx, initial_training_days)
            return result
        except Exception as e:
            logger.warning(f"Bootstrap iteration {iteration} failed: {e}")
            return None
        finally:
            self.strategy = original_strategy

    def run_monte_carlo(self, start_date, end_date, initial_training_days=252):
        """
        Run Monte Carlo bootstrap analysis

        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            initial_training_days: Initial training period for walk-forward validation

        Returns:
            Dictionary with results and confidence intervals
        """
        logger.info(f"Running Monte Carlo with {self.n_simulations} simulations")

        # First run the actual backtest using walk_forward_backtest
        print("Running actual backtest...")
        actual_results = self.strategy.walk_forward_backtest(
            start_date, end_date, initial_training_days=initial_training_days
        )

        # Get the price data
        data = self.strategy.data
        start_idx = initial_training_days

        # Run bootstrap simulations
        print(f"\nRunning {self.n_simulations} bootstrap simulations...")

        if self.n_jobs == 1:
            # Sequential execution
            print("Running in sequential mode...")
            bootstrap_results = []
            for i in tqdm(range(self.n_simulations), desc="Monte Carlo Progress"):
                result = self.run_bootstrap_backtest(data, start_idx, initial_training_days)
                if result is not None:
                    bootstrap_results.append(result)
        else:
            # Parallel execution
            print(f"Running in parallel mode with {self.n_jobs} workers...")

            # Create worker function with fixed parameters
            worker_func = partial(
                self._parallel_bootstrap_worker,
                data=data,
                start_idx=start_idx,
                initial_training_days=initial_training_days
            )

            # Run parallel bootstrap
            with Pool(processes=self.n_jobs) as pool:
                bootstrap_results = list(tqdm(
                    pool.imap(worker_func, range(self.n_simulations)),
                    total=self.n_simulations,
                    desc="Monte Carlo Progress"
                ))

            # Filter out None results
            bootstrap_results = [r for r in bootstrap_results if r is not None]

        # Calculate statistics
        metrics = ['total_return', 'sharpe_ratio', 'volatility', 'max_drawdown', 'n_rebalances']

        confidence_intervals = {}
        for metric in metrics:
            values = [r[metric] for r in bootstrap_results]
            confidence_intervals[metric] = {
                'actual': actual_results[metric] if metric != 'volatility' else actual_results['annualized_volatility'],
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'ci_5': np.percentile(values, 5),
                'ci_25': np.percentile(values, 25),
                'ci_75': np.percentile(values, 75),
                'ci_95': np.percentile(values, 95),
                'min': np.min(values),
                'max': np.max(values),
                'all_values': values
            }

        return {
            'actual_results': actual_results,
            'bootstrap_results': bootstrap_results,
            'confidence_intervals': confidence_intervals,
            'n_successful_simulations': len(bootstrap_results)
        }

    def print_results(self, mc_results):
        """
        Print Monte Carlo results with confidence intervals
        """
        ci = mc_results['confidence_intervals']

        print("\n" + "="*80)
        print("MONTE CARLO BOOTSTRAP ANALYSIS RESULTS")
        print("="*80)
        print(f"Number of simulations: {mc_results['n_successful_simulations']}")
        print("\n")

        # Format for different metrics
        formats = {
            'total_return': '{:.2%}',
            'sharpe_ratio': '{:.2f}',
            'volatility': '{:.2%}',
            'max_drawdown': '{:.2%}',
            'n_rebalances': '{:.0f}'
        }

        labels = {
            'total_return': 'Total Return',
            'sharpe_ratio': 'Sharpe Ratio',
            'volatility': 'Volatility (Ann.)',
            'max_drawdown': 'Max Drawdown',
            'n_rebalances': 'Rebalances'
        }

        for metric in ['total_return', 'sharpe_ratio', 'volatility', 'max_drawdown', 'n_rebalances']:
            fmt = formats[metric]
            data = ci[metric]

            print(f"{labels[metric]}:")
            print(f"  Actual:          {fmt.format(data['actual'])}")
            print(f"  Bootstrap Mean:  {fmt.format(data['mean'])} ± {fmt.format(data['std'])}")
            print(f"  Bootstrap Median: {fmt.format(data['median'])}")
            print(f"  90% CI:          [{fmt.format(data['ci_5'])}, {fmt.format(data['ci_95'])}]")
            print(f"  50% CI:          [{fmt.format(data['ci_25'])}, {fmt.format(data['ci_75'])}]")
            print(f"  Range:           [{fmt.format(data['min'])}, {fmt.format(data['max'])}]")
            print()

    def plot_distributions(self, mc_results, save_path='monte_carlo_distributions.png'):
        """
        Plot distributions of bootstrapped metrics
        """
        ci = mc_results['confidence_intervals']

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Monte Carlo Bootstrap Distributions', fontsize=16, fontweight='bold')

        metrics = [
            ('total_return', 'Total Return', '{:.1%}'),
            ('sharpe_ratio', 'Sharpe Ratio', '{:.2f}'),
            ('volatility', 'Annualized Volatility', '{:.1%}'),
            ('max_drawdown', 'Max Drawdown', '{:.1%}'),
            ('n_rebalances', 'Number of Rebalances', '{:.0f}')
        ]

        for idx, (metric, title, fmt) in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]

            data = ci[metric]
            values = data['all_values']

            # Histogram
            ax.hist(values, bins=30, alpha=0.7, color='steelblue', edgecolor='black')

            # Actual value line
            ax.axvline(data['actual'], color='red', linestyle='--', linewidth=2, label='Actual')

            # Mean line
            ax.axvline(data['mean'], color='green', linestyle='--', linewidth=2, label='Bootstrap Mean')

            # 90% CI
            ax.axvline(data['ci_5'], color='orange', linestyle=':', linewidth=1.5, label='90% CI')
            ax.axvline(data['ci_95'], color='orange', linestyle=':', linewidth=1.5)

            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        # Remove extra subplot
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nDistribution plot saved to: {save_path}")
        plt.close()


def run_analysis(n_jobs=-1, n_simulations=200):
    """
    Run comprehensive Monte Carlo analysis

    Args:
        n_jobs: Number of parallel jobs (-1 = all CPUs, 1 = sequential)
        n_simulations: Number of bootstrap simulations per period
    """
    strategy = HelixFactorStrategyFixed()

    # Display system info
    n_cpus = cpu_count()
    actual_jobs = n_jobs if n_jobs != -1 else n_cpus

    print("=" * 80)
    print("SYSTEM CONFIGURATION")
    print("=" * 80)
    print(f"Available CPUs: {n_cpus}")
    print(f"Parallel workers: {actual_jobs}")
    print(f"Mode: {'Parallel' if actual_jobs > 1 else 'Sequential'}")
    print(f"Simulations per period: {n_simulations}")
    print("=" * 80)

    # Test periods with initial training days
    test_periods = [
        ('2022-01-01', '2023-12-31', '2022-2023', n_simulations, 252),
        ('2024-01-01', '2025-08-31', '2024-2025', n_simulations, 252),
    ]

    all_results = {}

    for start_date, end_date, period_name, n_sims, training_days in test_periods:
        print("\n" + "="*80)
        print(f"PERIOD: {period_name} ({start_date} to {end_date})")
        print(f"Initial Training Days: {training_days}")
        print("="*80)

        # Create Monte Carlo backtester with parallel support
        mc_backtest = MonteCarloBacktest(strategy, n_simulations=n_sims, n_jobs=n_jobs)

        # Run Monte Carlo analysis
        mc_results = mc_backtest.run_monte_carlo(start_date, end_date,
                                                initial_training_days=training_days)

        # Print results
        mc_backtest.print_results(mc_results)

        # Plot distributions
        plot_path = f'monte_carlo_distributions_{period_name}.png'
        mc_backtest.plot_distributions(mc_results, save_path=plot_path)

        # Store results
        all_results[period_name] = mc_results

    return all_results


if __name__ == "__main__":
    results = run_analysis()

    print("\n" + "="*80)
    print("Monte Carlo analysis complete!")
    print("="*80)

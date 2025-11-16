#!/usr/bin/env python
"""
GPU-Accelerated Monte Carlo Bootstrap Analysis for Helix 1.1 Factor Strategy

This is an EXPERIMENTAL version that uses CuPy for GPU acceleration.
Requires NVIDIA GPU with CUDA support.

IMPORTANT: This version provides marginal benefits due to:
1. Sequential time-series logic (can't parallelize across time)
2. Heavy Python/pandas overhead
3. Optimization routines that don't benefit from GPU

For most users, the CPU parallel version (analyze_strategy_monte_carlo.py)
will be faster and easier to use.

Installation:
    pip install cupy-cuda11x  # Replace 11x with your CUDA version
    # or
    pip install cupy-cuda12x

Usage:
    python analyze_strategy_monte_carlo_gpu.py

Fallback:
    If CuPy is not available, falls back to NumPy automatically.
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

# Try to import CuPy, fallback to NumPy
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU detected! Using CuPy for acceleration.")
    print(f"GPU: {cp.cuda.Device().name}")
except ImportError:
    cp = np  # Fallback to NumPy
    GPU_AVAILABLE = False
    print("CuPy not available. Falling back to NumPy (CPU).")
    print("To enable GPU: pip install cupy-cuda11x (or cupy-cuda12x)")

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class GPUAcceleratedMonteCarloBacktest:
    """
    Monte Carlo bootstrap analysis with GPU acceleration for numerical operations

    Note: GPU benefits are limited due to sequential time-series logic
    """

    def __init__(self, strategy, n_simulations=100, random_seed=42, n_jobs=-1, use_gpu=True):
        """
        Initialize GPU-accelerated Monte Carlo backtester

        Args:
            strategy: HelixFactorStrategy instance
            n_simulations: Number of bootstrap simulations to run
            random_seed: Random seed for reproducibility
            n_jobs: Number of parallel CPU jobs (-1 = use all CPUs)
            use_gpu: Whether to use GPU for numerical operations (if available)
        """
        self.strategy = strategy
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()
        self.use_gpu = use_gpu and GPU_AVAILABLE

        np.random.seed(random_seed)
        if self.use_gpu:
            cp.random.seed(random_seed)

    def bootstrap_returns_gpu(self, returns_df):
        """
        GPU-accelerated bootstrap resampling

        This provides minimal speedup since bootstrapping is already fast
        """
        n_days = len(returns_df)

        if self.use_gpu:
            # Use GPU for random sampling
            bootstrap_indices = cp.random.choice(n_days, size=n_days, replace=True)
            bootstrap_indices = cp.asnumpy(bootstrap_indices)  # Transfer back to CPU
        else:
            bootstrap_indices = np.random.choice(n_days, size=n_days, replace=True)

        # Resample (must stay on CPU due to pandas)
        bootstrapped = returns_df.iloc[bootstrap_indices].copy()
        bootstrapped.index = returns_df.index

        return bootstrapped

    def compute_covariance_gpu(self, returns_df):
        """
        GPU-accelerated covariance matrix calculation

        This can provide 2-3x speedup for large matrices
        """
        if self.use_gpu and len(returns_df) > 100:
            # Transfer to GPU
            returns_array = cp.array(returns_df.values)

            # Compute on GPU
            cov_gpu = cp.cov(returns_array, rowvar=False)

            # Transfer back to CPU
            cov_matrix = cp.asnumpy(cov_gpu)

            # Convert to DataFrame
            return pd.DataFrame(cov_matrix,
                              index=returns_df.columns,
                              columns=returns_df.columns)
        else:
            # Use standard pandas for small matrices or no GPU
            return returns_df.cov()

    def batch_portfolio_simulation_gpu(self, returns_array, weights_array):
        """
        GPU-accelerated batch portfolio return calculation

        Computes portfolio returns for multiple weight vectors simultaneously

        Args:
            returns_array: (n_days, n_assets) array of returns
            weights_array: (n_simulations, n_assets) array of portfolio weights

        Returns:
            (n_simulations, n_days) array of portfolio returns
        """
        if self.use_gpu:
            returns_gpu = cp.array(returns_array)
            weights_gpu = cp.array(weights_array)

            # Matrix multiplication: (n_sim, n_assets) @ (n_assets, n_days)
            portfolio_returns_gpu = cp.dot(weights_gpu, returns_gpu.T)

            return cp.asnumpy(portfolio_returns_gpu)
        else:
            return np.dot(weights_array, returns_array.T)

    def _run_single_backtest(self, data, initial_training_days):
        """
        Walk-forward backtest with GPU-accelerated numerical operations

        Note: Most of the logic must stay sequential due to dependencies
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

        start_idx = initial_training_days

        for i in range(start_idx, len(returns)):
            current_date = returns.index[i]

            # Historical data (must be sequential)
            historical_returns = returns.iloc[:i]

            # Update regime models
            if i == start_idx or i % self.strategy.regime_update_frequency == 0:
                self.strategy._update_regime_models_online(historical_returns)

            # Get regimes
            current_regimes = {}
            for etf in self.strategy.factor_etfs.keys():
                if etf in self.strategy.regime_models:
                    regime = self.strategy.regime_models[etf].regimes_.iloc[-1]
                    current_regimes[etf] = regime
                else:
                    current_regimes[etf] = 0

            # Expected returns
            recent_returns = historical_returns.iloc[-self.strategy.lookback_days:]
            expected_returns = self.strategy._generate_expected_returns(recent_returns, current_regimes)

            # GPU-accelerated covariance calculation
            try:
                cov_matrix = self.compute_covariance_gpu(recent_returns)
                new_weights = self.strategy.optimizer.optimize(expected_returns, cov_matrix)
            except:
                new_weights = pd.Series(1.0/len(self.strategy.factor_etfs),
                                       index=self.strategy.factor_etfs.keys())

            # Rebalancing
            if self.strategy._should_rebalance(new_weights):
                self.strategy.current_weights = new_weights.copy()
                rebalance_dates.append(current_date)

            # Portfolio return calculation (could batch across simulations, but not across time)
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

    def run_bootstrap_backtest(self, original_data, start_idx, initial_training_days):
        """
        Run a single bootstrap backtest with GPU acceleration
        """
        original_returns = original_data.pct_change().dropna()

        # GPU-accelerated bootstrap
        bootstrapped_returns = self.bootstrap_returns_gpu(original_returns)

        original_strategy_data = self.strategy.data.copy() if self.strategy.data is not None else None

        try:
            # Reconstruct prices
            initial_prices = 100.0
            bootstrapped_prices = pd.DataFrame(
                index=bootstrapped_returns.index,
                columns=bootstrapped_returns.columns
            )

            bootstrapped_prices.iloc[0] = initial_prices
            for i in range(1, len(bootstrapped_returns)):
                bootstrapped_prices.iloc[i] = bootstrapped_prices.iloc[i-1] * (1 + bootstrapped_returns.iloc[i])

            self.strategy.data = bootstrapped_prices

            results = self._run_single_backtest(bootstrapped_prices, initial_training_days)

            return results

        except Exception as e:
            logger.warning(f"Bootstrap iteration failed: {e}")
            return None
        finally:
            self.strategy.data = original_strategy_data

    def _parallel_bootstrap_worker(self, iteration, data, start_idx, initial_training_days):
        """Worker function for parallel execution"""
        np.random.seed(self.random_seed + iteration)
        if self.use_gpu:
            cp.random.seed(self.random_seed + iteration)

        worker_strategy = HelixFactorStrategyFixed(
            lookback_days=self.strategy.lookback_days,
            rebalance_threshold=self.strategy.rebalance_threshold,
            regime_update_frequency=self.strategy.regime_update_frequency
        )

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
        Run Monte Carlo with GPU-accelerated numerical operations
        """
        logger.info(f"Running Monte Carlo with {self.n_simulations} simulations")

        print("Running actual backtest...")
        actual_results = self.strategy.walk_forward_backtest(
            start_date, end_date, initial_training_days=initial_training_days
        )

        data = self.strategy.data
        start_idx = initial_training_days

        print(f"\nRunning {self.n_simulations} bootstrap simulations...")

        if self.n_jobs == 1:
            print(f"Running in sequential mode (GPU: {self.use_gpu})...")
            bootstrap_results = []
            for _ in tqdm(range(self.n_simulations), desc="Monte Carlo Progress"):
                result = self.run_bootstrap_backtest(data, start_idx, initial_training_days)
                if result is not None:
                    bootstrap_results.append(result)
        else:
            print(f"Running in parallel mode with {self.n_jobs} workers (GPU: {self.use_gpu})...")

            worker_func = partial(
                self._parallel_bootstrap_worker,
                data=data,
                start_idx=start_idx,
                initial_training_days=initial_training_days
            )

            with Pool(processes=self.n_jobs) as pool:
                bootstrap_results = list(tqdm(
                    pool.imap(worker_func, range(self.n_simulations)),
                    total=self.n_simulations,
                    desc="Monte Carlo Progress"
                ))

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
        """Print results (same as CPU version)"""
        ci = mc_results['confidence_intervals']

        print("\n" + "="*80)
        print("MONTE CARLO BOOTSTRAP ANALYSIS RESULTS (GPU-ACCELERATED)")
        print("="*80)
        print(f"Number of simulations: {mc_results['n_successful_simulations']}")
        print(f"GPU enabled: {self.use_gpu}")
        print("\n")

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


def run_analysis(n_jobs=-1, n_simulations=200, use_gpu=True):
    """
    Run GPU-accelerated Monte Carlo analysis

    Args:
        n_jobs: Number of parallel CPU jobs
        n_simulations: Number of bootstrap simulations
        use_gpu: Whether to use GPU acceleration (if available)
    """
    strategy = HelixFactorStrategyFixed()

    n_cpus = cpu_count()
    actual_jobs = n_jobs if n_jobs != -1 else n_cpus

    print("=" * 80)
    print("SYSTEM CONFIGURATION (GPU-ACCELERATED VERSION)")
    print("=" * 80)
    print(f"Available CPUs: {n_cpus}")
    print(f"Parallel workers: {actual_jobs}")
    print(f"GPU available: {GPU_AVAILABLE}")
    print(f"GPU enabled: {use_gpu and GPU_AVAILABLE}")
    if GPU_AVAILABLE:
        print(f"GPU device: {cp.cuda.Device().name}")
    print(f"Simulations per period: {n_simulations}")
    print("=" * 80)

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

        mc_backtest = GPUAcceleratedMonteCarloBacktest(
            strategy, n_simulations=n_sims, n_jobs=n_jobs, use_gpu=use_gpu
        )

        mc_results = mc_backtest.run_monte_carlo(start_date, end_date,
                                                initial_training_days=training_days)

        mc_backtest.print_results(mc_results)

        all_results[period_name] = mc_results

    return all_results


if __name__ == "__main__":
    results = run_analysis()

    print("\n" + "="*80)
    print("GPU Monte Carlo analysis complete!")
    if not GPU_AVAILABLE:
        print("Note: CuPy not available. Install with: pip install cupy-cuda11x")
        print("For better performance, use the CPU parallel version: analyze_strategy_monte_carlo.py")
    print("="*80)

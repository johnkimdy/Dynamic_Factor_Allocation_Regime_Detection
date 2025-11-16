# Parallel Monte Carlo Bootstrap Analysis

This document explains the parallel Monte Carlo implementation for the Helix 1.1 Factor Strategy.

## Overview

The Monte Carlo bootstrap analysis now supports **parallel execution** using Python's multiprocessing, providing significant speedup for large simulation runs.

## Features

- **Automatic CPU detection**: Uses all available CPU cores by default
- **Configurable parallelism**: Run with 1 core (sequential) up to all available cores
- **Confidence intervals**: 90%, 50%, mean, median, std dev for all metrics
- **Reproducible results**: Proper random seeding ensures reproducibility
- **Progress tracking**: Real-time progress bars with tqdm
- **Distribution plots**: Automatic generation of bootstrap distribution visualizations

## Usage

### Basic Usage (All CPUs)

```bash
python analyze_strategy_monte_carlo.py
```

This automatically uses all available CPU cores.

### Sequential Mode (1 CPU)

```python
from analyze_strategy_monte_carlo import run_analysis
run_analysis(n_jobs=1)
```

Useful for debugging or when you need all CPU cores for other tasks.

### Custom Number of Cores

```python
from analyze_strategy_monte_carlo import run_analysis
run_analysis(n_jobs=4)  # Use 4 cores
```

### Custom Number of Simulations

```python
from analyze_strategy_monte_carlo import run_analysis
run_analysis(n_simulations=500)  # Run 500 bootstrap samples
```

### Combined Configuration

```python
from analyze_strategy_monte_carlo import run_analysis
run_analysis(n_jobs=8, n_simulations=1000)  # 8 cores, 1000 simulations
```

## Performance

Expected speedup with parallel execution:

| CPU Cores | Speedup | 200 sims time | 1000 sims time |
|-----------|---------|---------------|----------------|
| 1 (sequential) | 1.0x | ~60 min | ~5 hours |
| 2 cores   | ~1.8x   | ~33 min | ~2.8 hours |
| 4 cores   | ~3.5x   | ~17 min | ~1.4 hours |
| 8 cores   | ~6.5x   | ~9 min  | ~46 min |
| 16 cores  | ~11x    | ~5.5 min | ~27 min |

*Note: Actual speedup depends on CPU architecture, memory bandwidth, and system load.*

## Output

The analysis produces:

1. **Console output**: Detailed statistics with confidence intervals
2. **Distribution plots**: PNG files showing histograms of bootstrapped metrics
3. **Results dictionary**: Stored in-memory for further analysis

### Example Output

```
MONTE CARLO BOOTSTRAP ANALYSIS RESULTS
================================================================================
Number of simulations: 200

Total Return:
  Actual:          24.92%
  Bootstrap Mean:  25.13% ± 3.45%
  Bootstrap Median: 25.01%
  90% CI:          [19.21%, 31.55%]
  50% CI:          [22.87%, 27.39%]
  Range:           [15.44%, 35.67%]

Sharpe Ratio:
  Actual:          1.84
  Bootstrap Mean:  1.81 ± 0.23
  Bootstrap Median: 1.83
  90% CI:          [1.42, 2.18]
  50% CI:          [1.68, 1.95]
  Range:           [1.15, 2.51]
```

## Technical Details

### Bootstrap Method

The implementation uses **block bootstrap resampling**:
- Resamples daily returns with replacement
- Preserves cross-sectional correlation structure
- Maintains temporal ordering for strategy logic

### Parallelization Strategy

- Each worker process gets its own strategy instance (no shared state)
- Unique random seeds per iteration ensure reproducibility
- Data is passed as immutable DataFrames to avoid pickling issues
- Results are collected and aggregated in the main process

### Memory Considerations

- Each worker holds a copy of the price data (~5-10 MB)
- With 8 workers and 200 simulations, peak memory: ~100-200 MB
- For very long backtests (10+ years), consider reducing workers

## Troubleshooting

### "Pool" import errors on Windows

If you encounter multiprocessing issues on Windows, use:

```python
if __name__ == "__main__":
    from analyze_strategy_monte_carlo import run_analysis
    run_analysis()
```

### Out of memory errors

Reduce the number of parallel workers:

```python
run_analysis(n_jobs=4)  # Use fewer cores
```

### Inconsistent results

Ensure random seed is set properly. The implementation uses:
- Base seed: 42
- Per-iteration seed: 42 + iteration_number

## Dependencies

- Python 3.10+
- numpy
- pandas
- matplotlib
- scipy
- scikit-learn
- tqdm
- yfinance

Install all dependencies:

```bash
pip install -r requirements.txt
```

## See Also

- `analyze_strategy.py`: Original sequential analysis
- `helix_factor_strategy_fixed.py`: Fixed strategy implementation (no data leakage)
- `DATA_LEAKAGE_FIXES.md`: Documentation of data leakage fixes

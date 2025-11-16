# Helix 1.1: Factor-Based Portfolio Optimization Strategy

A lightweight daily portfolio rebalancing system using Sparse Jump Models for regime identification across factor ETFs, inspired by the dual-system architecture concept but adapted for practical portfolio management.

## Overview

Helix 1.1 represents a significant evolution from the original high-frequency approach, focusing on:

- **Daily portfolio optimization** using factor ETFs rather than individual stocks
- **Lightweight models** avoiding transformers for computational efficiency  
- **Long-only constraints** for practical implementation
- **Regime-aware allocation** using Sparse Jump Models for factor timing

## Architecture

### System Design
The strategy maintains the dual-system concept but adapts it for portfolio management:

- **System 2 (Strategic)**: Sparse Jump Models running daily for regime identification across factors
- **System 1 (Tactical)**: Black-Litterman optimizer for portfolio construction and rebalancing decisions

### Factor Universe
The strategy focuses on six factor ETFs with low expense ratios (0.15% each):

| Factor | ETF | Description |
|--------|-----|-------------|
| Market | SPY | SPDR S&P 500 ETF Trust |
| Quality | QUAL | iShares MSCI USA Quality Factor ETF |
| Momentum | MTUM | iShares MSCI USA Momentum Factor ETF |
| Low Volatility | USMV | iShares MSCI USA Min Vol Factor ETF |
| Value | VLUE | iShares MSCI USA Value Factor ETF |
| Size | SIZE | iShares MSCI USA Size Factor ETF |

## Methodology

### Regime Identification
- **Sparse Jump Model (SJM)** for each factor ETF
- Technical features: EWMA returns, RSI-like momentum, price momentum, volatility
- Jump penalty parameter to reduce regime switching frequency
- Binary regime classification (positive/negative market conditions)

### Portfolio Construction
- **Black-Litterman optimization** with equal-weight benchmark
- Dynamic allocation based on factor-specific regime inferences
- **Long-only constraint** with full investment requirement
- Rebalancing threshold of 2% to minimize transaction costs

## Performance Results

### Latest Performance (2024-2025 Period) - Updated 2025-01-16
```
Total Return:           10.66%
Sharpe Ratio:           0.73
Annualized Volatility:  22.10%
Maximum Drawdown:       -18.12%
Number of Rebalances:   33
```

### Monte Carlo Bootstrap Analysis (200 Simulations)
**Total Return:**
- Bootstrap Mean: 12.81% ± 14.31%
- 90% Confidence Interval: [-6.90%, 37.90%]
- 50% Confidence Interval: [3.42%, 21.09%]

**Sharpe Ratio:**
- Bootstrap Mean: 1.14 ± 1.20
- 90% Confidence Interval: [-0.71, 3.21]
- 50% Confidence Interval: [0.35, 1.83]

**Volatility (Annualized):**
- Bootstrap Mean: 16.59% ± 2.52%
- 90% Confidence Interval: [13.04%, 20.67%]

**Max Drawdown:**
- Bootstrap Mean: -10.54% ± 3.65%
- 90% Confidence Interval: [-17.26%, -4.94%]

### Key Performance Insights
- **Moderate risk-adjusted returns**: Sharpe ratio of 0.73 indicates positive but moderate risk-adjusted performance
- **Active rebalancing**: 33 rebalances demonstrates adaptive regime detection
- **Bootstrap validation**: Wide confidence intervals reflect uncertainty inherent in factor timing
- **Realistic expectations**: Performance metrics align with fixed data leakage implementation

### Multi-Period Analysis with Monte Carlo Confidence Intervals

| Period | Total Return | Sharpe Ratio | Volatility | Max Drawdown | Rebalances |
|--------|--------------|--------------|------------|--------------|------------|
| **2024-2025** | | | | | |
| Actual | 10.66% | 0.73 | 22.10% | -18.12% | 33 |
| Bootstrap Mean | 12.81% ± 14.31% | 1.14 ± 1.20 | 16.59% ± 2.52% | -10.54% ± 3.65% | 48 ± 11 |
| 90% CI | [-6.90%, 37.90%] | [-0.71, 3.21] | [13.04%, 20.67%] | [-17.26%, -4.94%] | [33, 71] |
| 50% CI | [3.42%, 21.09%] | [0.35, 1.83] | [14.58%, 18.45%] | [-12.78%, -7.80%] | [40, 53] |
| **2022-2023** | | | | | |
| Actual | 29.57% | 2.13 | 13.43% | -9.46% | 80 |
| Bootstrap Mean | (running) | (running) | (running) | (running) | (running) |
| 90% CI | (running) | (running) | (running) | (running) | (running) |
| 50% CI | (running) | (running) | (running) | (running) | (running) |

*Note: Performance varies significantly across market regimes. Monte Carlo bootstrap analysis (200 simulations) shows wide confidence intervals, particularly for 2024-2025 period, highlighting the inherent uncertainty in factor timing strategies. The 2022-2023 period demonstrates stronger and more consistent performance.*

## Technical Implementation

### Core Components

1. **SparseJumpModel**: Lightweight regime identification
2. **BlackLittermanOptimizer**: Portfolio optimization engine
3. **HelixFactorStrategy**: Main strategy orchestrator

### Key Features
- **Regime persistence**: Jump penalty reduces unnecessary switching
- **Feature engineering**: Multiple technical indicators for robust regime detection
- **Risk management**: Long-only constraints and drawdown control
- **Computational efficiency**: Avoids complex transformers or deep learning

### Data Leakage Prevention (Fixed - 2025-01-16)

**Important Update:** The original implementation contained several critical data leakage issues that have been fixed:

#### Fixed Implementation Files
- `helix_factor_strategy_fixed.py` - Production-ready version with no data leakage
- `sjm_regime_visualization_FIXED.ipynb` - Properly validated visualization notebook
- `DATA_LEAKAGE_FIXES.md` - Complete documentation of all fixes

#### Issues Addressed
1. **Online StandardScaler**: Incremental normalization preventing future data contamination
2. **Forward-Only Regime Detection**: Removed backward Viterbi pass that used future data
3. **Walk-Forward Validation**: Proper temporal train/test splits
4. **Expanding Window**: Models only see historical data at each decision point
5. **Out-of-Sample Testing**: Realistic performance validation

#### Performance Impact
- Original (with leakage): Sharpe Ratio 4.39 (unrealistic)
- Fixed (no leakage): Sharpe Ratio 0.73-1.14 (realistic for factor timing)

**For production use, always use `helix_factor_strategy_fixed.py` instead of the original implementation.**

See [DATA_LEAKAGE_FIXES.md](DATA_LEAKAGE_FIXES.md) for complete technical details.

## Usage

### Setup Environment
```bash
# Create conda environment
conda env create -f environment.yml
conda activate factor_regime_detection
```

### Basic Implementation (Fixed Version - Recommended)
```python
from helix_factor_strategy_fixed import HelixFactorStrategyFixed

# Initialize strategy with walk-forward validation
strategy = HelixFactorStrategyFixed(
    lookback_days=252,              # 1-year lookback for optimization
    rebalance_threshold=0.02,       # 2% threshold for rebalancing
    regime_update_frequency=20      # Update regime models every 20 days
)

# Run walk-forward backtest (no data leakage)
results = strategy.walk_forward_backtest(
    start_date='2023-01-01',
    end_date='2025-08-31',
    initial_training_days=252
)

# View performance
print("Total Return: {:.2%}".format(results['total_return']))
print("Sharpe Ratio: {:.2f}".format(results['sharpe_ratio']))
print("Max Drawdown: {:.2%}".format(results['max_drawdown']))
print("Rebalances: {}".format(results['n_rebalances']))
```

### Original Implementation (Research Only - Has Data Leakage)
```python
from helix_factor_strategy import HelixFactorStrategy

# WARNING: This version has data leakage and inflated performance metrics
# Only use for research/visualization, NOT for backtesting or live trading
strategy = HelixFactorStrategy()
results = strategy.backtest('2022-01-01', '2023-12-31')
```

### Comprehensive Analysis
```bash
# Run full analysis with Monte Carlo bootstrap
python analyze_strategy_monte_carlo.py

# Or with GPU acceleration (if CUDA available)
python analyze_strategy_monte_carlo_gpu.py
```

### Visualization Notebooks
```bash
# Fixed version (recommended)
jupyter notebook sjm_regime_visualization_FIXED.ipynb

# Original version (research only - has data leakage)
jupyter notebook sjm_regime_visualization.ipynb
```

## Theoretical Foundation

### Academic Inspiration
This implementation draws from ["Portfolio Allocation Using Sparse Jump Model"](https://arxiv.org/html/2410.14841v1) (arXiv:2410.14841v1), which demonstrates:

- Regime identification across factor indices using jump models
- Daily rebalancing with Black-Litterman portfolio construction
- Improved information ratios through factor timing

### Model Advantages
1. **Interpretability**: Clear regime signals and factor allocations
2. **Efficiency**: Lightweight computation suitable for daily execution
3. **Robustness**: Long-only constraints reduce extreme positions
4. **Practicality**: Uses liquid ETFs with reasonable expense ratios

## Limitations and Considerations

### Known Constraints
- **Regime dependency**: Performance varies significantly across market conditions
- **Limited universe**: Restricted to US equity factors only
- **Lookback sensitivity**: Requires sufficient historical data for regime fitting
- **Transaction costs**: Not explicitly modeled in backtests

### Risk Factors
- **Factor timing risk**: Regime models may fail during market transitions
- **Concentration risk**: Limited diversification across asset classes
- **Model risk**: Simple regime identification may miss complex market dynamics

## Installation & Requirements

### Recommended Setup (Conda)
```bash
# Clone or download the repository
cd Dynamic_Factor_Allocation_Regime_Detection

# Create conda environment from provided file
conda env create -f environment.yml

# Activate environment
conda activate factor_regime_detection
```

### Alternative Setup (pip)
```bash
pip install numpy>=1.24.0 pandas>=2.0.0 matplotlib>=3.7.0 seaborn>=0.12.0
pip install scipy>=1.10.0 scikit-learn>=1.3.0 yfinance>=0.2.28
pip install jupyter ipykernel  # For notebooks
```

### Python Version
- Requires Python 3.10+ (tested on Python 3.10 and 3.11)
- Compatible with standard scientific computing stack

## Future Enhancements

### Potential Improvements
- [ ] **Multi-asset expansion**: Include international and fixed-income factors
- [ ] **Transaction cost modeling**: Explicit cost consideration in optimization
- [ ] **Risk budgeting**: Volatility targeting and risk parity approaches
- [ ] **Alternative models**: Comparison with Hidden Markov Models or regime-switching GARCH
- [ ] **Live trading integration**: Real-time data feeds and execution systems

### Research Directions
- **Factor timing evaluation**: Comparative analysis vs. static allocation
- **Regime model validation**: Out-of-sample regime identification accuracy
- **Benchmark comparison**: Performance vs. traditional factor strategies

## Repository Structure

### Core Implementation Files
- `helix_factor_strategy_fixed.py` - **Production-ready** implementation (no data leakage)
- `helix_factor_strategy.py` - Original implementation (research only, has data leakage)
- `environment.yml` - Conda environment specification

### Analysis Scripts
- `analyze_strategy_monte_carlo.py` - Monte Carlo bootstrap validation
- `analyze_strategy_monte_carlo_gpu.py` - GPU-accelerated Monte Carlo analysis
- `analyze_strategy.py` - Basic strategy analysis

### Notebooks
- `sjm_regime_visualization_FIXED.ipynb` - **Recommended** visualization (no data leakage)
- `sjm_regime_visualization.ipynb` - Original notebook (has data leakage)

### Documentation
- `DATA_LEAKAGE_FIXES.md` - Complete technical documentation of all data leakage fixes
- `README.md` - This file
- `GPU_ACCELERATION_README.md` - GPU acceleration documentation
- `MONTE_CARLO_PARALLEL.md` - Parallel Monte Carlo documentation

### Results
- `monte_carlo_distributions_2024-2025.png` - Latest Monte Carlo analysis results

## Disclaimer

This is an experimental portfolio optimization strategy developed for research and educational purposes. Past performance does not guarantee future results. The strategy involves significant risks including potential loss of capital. Users should conduct their own due diligence and consider consulting with financial professionals before implementing any trading strategy.

**Important:** Always use the fixed implementation (`helix_factor_strategy_fixed.py`) for backtesting and production. The original implementation contains data leakage issues that inflate performance metrics.

## References

1. Shu, Y. O. (2024). ["Portfolio Allocation Using Sparse Jump Model"](https://arxiv.org/html/2410.14841v1). arXiv preprint arXiv:2410.14841.
2. [Jump Models GitHub Repository](https://github.com/Yizhan-Oliver-Shu/jump-models)
3. Black, F., & Litterman, R. (1992). Global portfolio optimization. Financial Analysts Journal, 48(5), 28-43.

---

*Helix 1.1 - A pragmatic evolution towards practical factor-based portfolio optimization.*

**Last Updated:** 2025-01-16 - Added data leakage fixes and Monte Carlo validation results
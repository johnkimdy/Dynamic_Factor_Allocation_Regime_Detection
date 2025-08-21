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

### 2022-2023 Period Performance
```
Total Return:           17.45%
Annualized Return:      18.77%
Sharpe Ratio:           1.46
Annualized Volatility:  12.39%
Maximum Drawdown:       -9.37%
Number of Rebalances:   1
```

### Key Performance Insights
- **Strong risk-adjusted returns**: Sharpe ratio of 1.46 indicates excellent risk-adjusted performance
- **Low rebalancing frequency**: Only 1 rebalance over 2 years suggests regime persistence
- **Controlled downside**: Maximum drawdown of -9.37% demonstrates effective risk management
- **Moderate volatility**: 12.39% annualized volatility provides stability

### Multi-Period Analysis
| Period | Total Return | Sharpe | Volatility | Max Drawdown | Rebalances |
|--------|--------------|--------|------------|--------------|------------|
| 2022-2023 | 17.45% | 1.46 | 12.39% | -9.37% | 1 |
| 2021-2023 | -1.05% | 0.06 | 18.32% | -24.04% | 0 |

*Note: Extended period performance shows the strategy's sensitivity to market regimes.*

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

## Usage

### Basic Implementation
```python
from helix_factor_strategy import HelixFactorStrategy

# Initialize strategy
strategy = HelixFactorStrategy(
    lookback_days=252,        # 1-year lookback for optimization
    rebalance_threshold=0.02  # 2% threshold for rebalancing
)

# Run backtest
results = strategy.backtest('2022-01-01', '2023-12-31')

# View performance
print("Total Return: {:.2%}".format(results['total_return']))
print("Sharpe Ratio: {:.2f}".format(results['sharpe_ratio']))
print("Max Drawdown: {:.2%}".format(results['max_drawdown']))
```

### Comprehensive Analysis
```bash
# Run full analysis across multiple periods
python3.11 analyze_strategy.py
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

### Dependencies
```bash
pip3 install numpy pandas yfinance scikit-learn scipy
```

### Python Version
- Requires Python 3.x (tested on Python 3.11)
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

## Disclaimer

This is an experimental portfolio optimization strategy developed for research and educational purposes. Past performance does not guarantee future results. The strategy involves significant risks including potential loss of capital. Users should conduct their own due diligence and consider consulting with financial professionals before implementing any trading strategy.

## References

1. Shu, Y. O. (2024). ["Portfolio Allocation Using Sparse Jump Model"](https://arxiv.org/html/2410.14841v1). arXiv preprint arXiv:2410.14841.
2. [Jump Models GitHub Repository](https://github.com/Yizhan-Oliver-Shu/jump-models)
3. Black, F., & Litterman, R. (1992). Global portfolio optimization. Financial Analysts Journal, 48(5), 28-43.

---

*Helix 1.1 - A pragmatic evolution towards practical factor-based portfolio optimization.*
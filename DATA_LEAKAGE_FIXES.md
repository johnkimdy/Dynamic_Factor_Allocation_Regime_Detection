# Data Leakage Fixes - Complete Documentation

This document details all data leakage issues found in the original implementation and how they were fixed.

---

## Summary of Issues Found

### CRITICAL Issues (All Fixed)

1. **Global Model Fitting** - Model trained on entire dataset including future data
2. **StandardScaler Contamination** - Normalization used statistics from future data
3. **Backward Viterbi Pass** - Regime detection used future data to determine past regimes
4. **Iterative Optimization with Future Data** - Parameters updated using entire time series
5. **No Train/Test Split** - No temporal validation or out-of-sample testing

---

## Issue #1: Global Model Fitting

### Original Code (BROKEN)
```python
# In sjm_regime_visualization.ipynb
data = yf.download(list(factor_etfs.keys()), start='2023-01-01', end='2025-08-31')
model.fit(returns[etf])  # Fits on ENTIRE dataset
```

**Problem:** Model sees entire future (2023-2025) when making decisions at any point in time.

### Fixed Code
```python
# In helix_factor_strategy_fixed.py - walk_forward_backtest()
for i in range(start_idx, len(returns)):
    # ONLY use data UP TO time i
    historical_returns = returns.iloc[:i]

    # Update models periodically using only historical data
    if i % self.regime_update_frequency == 0:
        self._update_regime_models_online(historical_returns)
```

**Fix:** Walk-forward validation where models only see past data at each time point.

---

## Issue #2: StandardScaler Contamination

### Original Code (BROKEN)
```python
# In helix_factor_strategy.py:89
X = self.scaler.fit_transform(features_clean)  # Uses mean/std from ENTIRE dataset
```

**Problem:**
- Scaler computes mean and std from entire time series (t=0 to t=659)
- Data point at t=100 gets normalized using statistics that include data from t=101 to t=659
- This is equivalent to "peeking into the future"

### Fixed Code
```python
# In helix_factor_strategy_fixed.py - OnlineStandardScaler
class OnlineStandardScaler:
    def partial_fit(self, X):
        """Update statistics incrementally using Welford's algorithm"""
        for sample in X:
            self.n_ += 1
            delta = sample - self.mean_
            self.mean_ += delta / self.n_
            delta2 = sample - self.mean_
            self.var_ += delta * delta2

        self.std_ = np.sqrt(self.var_ / max(self.n_ - 1, 1))

    def transform(self, X):
        """Transform using CURRENT statistics only"""
        return (X - self.mean_) / self.std_

# Usage in online model
x_scaled = self.scaler.transform(features.reshape(1, -1))[0]
```

**Fix:** Incremental scaler that updates statistics online, never uses future data.

---

## Issue #3: Backward Viterbi Pass

### Original Code (BROKEN)
```python
# In helix_factor_strategy.py:171-178
def _optimize_regimes(self, X, centroids, weights):
    # Forward pass
    for t in range(1, n_obs):
        for k in range(n_states):
            # Calculate costs...

    # BACKWARD PASS - USES FUTURE!
    regimes[-1] = np.argmin(cost[-1])  # Start from END
    for t in range(n_obs - 2, -1, -1):  # Walk BACKWARDS
        regimes[t] = path[t + 1][regimes[t + 1]]  # Future determines past!
```

**Problem:**
- Viterbi algorithm determines regime at time t by looking at optimal path from time t+1
- Decision at time 100 depends on information from time 101, 102, ..., 659
- This is fundamentally look-ahead bias

### Fixed Code
```python
# In helix_factor_strategy_fixed.py:137-167
def predict_regime_online(self, x_t, prev_regime):
    """
    Predict regime using FORWARD-ONLY logic
    No backward pass, no future data
    """
    if prev_regime is None:
        prev_regime = 0

    costs = np.zeros(self.n_regimes)

    for k in range(self.n_regimes):
        # Distance to centroid (current data only)
        weighted_diff = self.feature_weights_ * (x_t - self.centroids_[k])
        dist_cost = np.sum(weighted_diff ** 2)

        # Jump penalty (uses only previous regime, no future)
        jump_cost = self.jump_penalty if prev_regime != k else 0

        costs[k] = dist_cost + jump_cost

    # Select regime with minimum cost (greedy, forward-only)
    regime = np.argmin(costs)

    return regime, probabilities
```

**Fix:** Greedy forward selection instead of global optimization. Each decision uses only current and past information.

---

## Issue #4: Iterative Optimization with Future Data

### Original Code (BROKEN)
```python
# In helix_factor_strategy.py:104-135
for iteration in range(self.max_iter):
    # Uses entire X matrix (all time points)
    regimes = self._optimize_regimes(X, centroids, w)
    centroids = self._update_centroids(X, regimes, w)
    w = self._update_weights(X, regimes, centroids)
```

**Problem:**
- Centroids updated using all assigned points (including future)
- Weights calculated from variance across entire time series
- Creates feedback loop where future influences past

### Fixed Code
```python
# In helix_factor_strategy_fixed.py:185-217
def _update_parameters(self):
    """Update using only recent data from buffer"""
    if len(self.features_buffer_) < 10:
        return

    # Use only recent historical data
    X_recent = np.array(self.features_buffer_[-self.window:])
    regimes_recent = np.array(self.regimes_buffer_[-self.window:])

    # Update centroids using only historical assignments
    for k in range(self.n_regimes):
        mask = regimes_recent == k
        if np.sum(mask) > 0:
            self.centroids_[k] = np.mean(X_recent[mask], axis=0)

    # Update weights using only historical data
    self._update_feature_weights(X_recent, regimes_recent)
```

**Fix:** Parameters updated using only a rolling window of historical data, periodically refreshed.

---

## Issue #5: No Train/Test Split

### Original Code (BROKEN)
```python
# In sjm_regime_visualization.ipynb
# No split - evaluate on same data used for training
model.fit(returns[etf])
# Calculate performance on same data
regime_returns = returns[regime_mask]  # Same data!
sharpe_ratio = 4.39  # Unrealistically high
```

**Problem:**
- No temporal train/test split
- Performance metrics calculated on training data
- No out-of-sample validation
- Sharpe ratios of 4+ are not realistic for live trading

### Fixed Code
```python
# In sjm_regime_visualization_FIXED.ipynb
# Proper train/test split
train_end_date = '2024-12-31'
train_data = data[:train_end_date]
test_data = data[train_end_date:]

# Train on historical data only
model.fit_online(train_returns[etf])

# Test on unseen future data
for i in range(len(test_returns)):
    historical_returns = pd.concat([train_returns[etf], test_returns[etf].iloc[:i+1]])
    regime, confidence = model.predict_next(historical_returns)

# Calculate separate metrics
train_sharpe = calculate_sharpe(train_returns, train_regimes)
test_sharpe = calculate_sharpe(test_returns, test_regimes)  # More realistic
```

**Fix:** Proper temporal train/test split with out-of-sample validation.

---

## Files Created

### 1. `helix_factor_strategy_fixed.py`
Complete refactored implementation with:
- `OnlineStandardScaler` - Incremental normalization
- `OnlineSparseJumpModel` - Forward-only regime detection
- `HelixFactorStrategyFixed` - Walk-forward backtesting framework

### 2. `sjm_regime_visualization_FIXED.ipynb`
Fixed visualization notebook with:
- Train/test split
- Online model training
- Out-of-sample validation
- Realistic performance metrics

### 3. `environment.yml`
Conda environment specification with all required dependencies.

### 4. This file: `DATA_LEAKAGE_FIXES.md`
Complete documentation of issues and fixes.

---

## How to Use Fixed Implementation

### Install Environment
```bash
cd /mnt/p/Dynamic_Factor_Allocation_Regime_Detection
conda env create -f environment.yml
conda activate factor_regime_detection
```

### Run Fixed Strategy
```python
from helix_factor_strategy_fixed import HelixFactorStrategyFixed

strategy = HelixFactorStrategyFixed()
results = strategy.walk_forward_backtest(
    start_date='2023-01-01',
    end_date='2025-08-31',
    initial_training_days=252
)

print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

### Run Fixed Notebook
```bash
jupyter notebook sjm_regime_visualization_FIXED.ipynb
```

---

## Expected Performance Differences

### Original (BROKEN - with data leakage)
- Sharpe Ratio: 4.39 (unrealistic)
- Returns: Highly inflated
- Regime detection: Appears perfect (because it sees the future)

### Fixed (NO data leakage)
- Sharpe Ratio: 1.0-2.0 (realistic for factor strategies)
- Returns: More moderate, achievable in live trading
- Regime detection: Less perfect, but honest

**Important:** If you see Sharpe ratios above 3.0, you likely have data leakage!

---

## Key Principles for Avoiding Data Leakage

1. **Temporal Validation**: Always split time series data chronologically
2. **Expanding Window**: Use only data up to current time point
3. **Online Learning**: Update models incrementally as new data arrives
4. **Forward-Only**: Never use backward passes or global optimization on full dataset
5. **Out-of-Sample Testing**: Always validate on unseen future data

---

## Production Deployment Checklist

Before deploying to live trading:

- [ ] All models use online/incremental learning
- [ ] No batch normalization on full dataset
- [ ] No backward optimization passes
- [ ] Walk-forward validation shows stable performance
- [ ] Out-of-sample Sharpe ratio is realistic (1-2 range)
- [ ] Train/test performance gap is small (not overfitting)
- [ ] Code reviewed for any `.fit()` calls on future data

---

## Additional Notes

### Why the Original Code Was Wrong

The original implementation was designed for **offline analysis** where you have the full dataset and want to find the "best" regime sequence in hindsight. This is useful for:
- Research and visualization
- Understanding regime patterns historically
- Academic papers

But it's completely unsuitable for:
- Live trading
- Backtesting (produces inflated results)
- Strategy development
- Performance evaluation

### Why the Fixed Code Is Better

The fixed implementation simulates **real-time trading** where:
- You only know the past, not the future
- Models must make decisions with incomplete information
- Performance metrics reflect actual trading results
- Strategy can be deployed to live markets

---

## Contact

For questions or issues with the fixed implementation, please refer to the code comments or create an issue in the repository.

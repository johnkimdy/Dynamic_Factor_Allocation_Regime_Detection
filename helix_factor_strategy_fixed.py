"""
Helix 1.1: Factor-Based Portfolio Optimization Strategy (FIXED VERSION)

This version fixes all data leakage issues:
- Online/incremental StandardScaler
- Forward-only regime detection (no backward pass)
- Walk-forward validation framework
- Proper temporal train/test splits

A lightweight daily portfolio rebalancing strategy using Sparse Jump Models
for regime identification across factor ETFs, inspired by the paper:
"Portfolio Allocation Using Sparse Jump Model" (arXiv:2410.14841v1)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from scipy.optimize import minimize
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OnlineStandardScaler:
    """
    Incremental/Online StandardScaler that prevents data leakage
    Updates mean and std incrementally as new data arrives
    """

    def __init__(self):
        self.mean_ = None
        self.var_ = None
        self.n_ = 0
        self.std_ = None

    def partial_fit(self, X):
        """
        Update statistics with new data point(s)
        Uses Welford's online algorithm for numerical stability
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        n_samples, n_features = X.shape

        if self.mean_ is None:
            # Initialize
            self.mean_ = np.zeros(n_features)
            self.var_ = np.zeros(n_features)

        for sample in X:
            self.n_ += 1
            delta = sample - self.mean_
            self.mean_ += delta / self.n_
            delta2 = sample - self.mean_
            self.var_ += delta * delta2

        # Calculate std with minimum threshold to avoid division by zero
        self.std_ = np.sqrt(self.var_ / max(self.n_ - 1, 1))
        self.std_ = np.maximum(self.std_, 1e-8)  # Prevent division by zero

        return self

    def transform(self, X):
        """Transform data using current statistics"""
        if self.mean_ is None:
            raise ValueError("Scaler not fitted. Call partial_fit() first.")

        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        return (X - self.mean_) / self.std_

    def partial_fit_transform(self, X):
        """Fit and transform in one step"""
        self.partial_fit(X)
        return self.transform(X)


class OnlineSparseJumpModel:
    """
    Online Sparse Jump Model - prevents data leakage by:
    1. Using incremental scaler
    2. Forward-only regime detection (no backward Viterbi pass)
    3. Online parameter updates
    """

    def __init__(self, n_regimes=2, jump_penalty=0.1, window=100, update_frequency=20):
        self.n_regimes = n_regimes
        self.jump_penalty = jump_penalty
        self.window = window  # Window for parameter estimation
        self.update_frequency = update_frequency  # How often to update centroids/weights

        self.scaler = OnlineStandardScaler()
        self.feature_weights_ = None
        self.centroids_ = None
        self.regimes_ = []
        self.regime_probabilities_ = []  # Track probability distribution
        self.features_buffer_ = []  # Buffer for recent features
        self.regimes_buffer_ = []  # Buffer for recent regimes
        self.t_ = 0  # Time counter

    def _calculate_features_single(self, returns_history):
        """
        Calculate features for a single time point using only historical data

        Args:
            returns_history: pandas Series of historical returns up to current time
        """
        if len(returns_history) < 20:
            return None

        features = []

        # EWMA - uses only past data
        ewma = returns_history.ewm(span=20).mean().iloc[-1]
        features.append(ewma)

        # RSI - uses only past data
        delta = returns_history.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().iloc[-1]
        if loss == 0:
            rsi = 100
        else:
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
        features.append(rsi)

        # Momentum - uses only past data
        momentum = returns_history.tail(20).sum()
        features.append(momentum)

        # Volatility - uses only past data
        volatility = returns_history.tail(20).std()
        features.append(volatility)

        return np.array(features)

    def _initialize_model(self, X_initial):
        """Initialize model parameters with first batch of data"""
        n_features = X_initial.shape[1]

        # Initialize feature weights uniformly
        self.feature_weights_ = np.ones(n_features) / n_features

        # Initialize centroids with k-means on initial data
        if len(X_initial) >= self.n_regimes:
            # Simple k-means initialization
            indices = np.random.choice(len(X_initial), self.n_regimes, replace=False)
            self.centroids_ = X_initial[indices].copy()
        else:
            # Random initialization if not enough data
            self.centroids_ = np.random.randn(self.n_regimes, n_features) * 0.1

        logger.info(f"Initialized model with {n_features} features and {self.n_regimes} regimes")

    def predict_regime_online(self, x_t, prev_regime):
        """
        Predict regime for time t using only current and past information (FORWARD ONLY)

        Args:
            x_t: Current feature vector (already scaled)
            prev_regime: Previous regime (or None if first prediction)

        Returns:
            regime: Predicted regime for time t
            probabilities: Probability distribution over regimes
        """
        if prev_regime is None:
            prev_regime = 0

        # Calculate cost for each regime
        costs = np.zeros(self.n_regimes)

        for k in range(self.n_regimes):
            # Distance to centroid
            weighted_diff = self.feature_weights_ * (x_t - self.centroids_[k])
            dist_cost = np.sum(weighted_diff ** 2)

            # Jump penalty
            jump_cost = self.jump_penalty if prev_regime != k else 0

            costs[k] = dist_cost + jump_cost

        # Select regime with minimum cost
        regime = np.argmin(costs)

        # Convert costs to probabilities (softmax)
        exp_costs = np.exp(-costs / 2)
        probabilities = exp_costs / np.sum(exp_costs)

        return regime, probabilities

    def _update_parameters(self):
        """Update centroids and weights using recent data (from buffer)"""
        if len(self.features_buffer_) < 10:
            return

        X_recent = np.array(self.features_buffer_[-self.window:])
        regimes_recent = np.array(self.regimes_buffer_[-self.window:])

        # Update centroids
        for k in range(self.n_regimes):
            mask = regimes_recent == k
            if np.sum(mask) > 0:
                self.centroids_[k] = np.mean(X_recent[mask], axis=0)

        # Update feature weights based on variance reduction
        self._update_feature_weights(X_recent, regimes_recent)

    def _update_feature_weights(self, X, regimes):
        """Update feature weights based on how well they separate regimes"""
        n_features = X.shape[1]

        # Calculate total variance
        total_var = np.var(X, axis=0)

        # Calculate within-regime variance
        within_var = np.zeros(n_features)
        total_points = 0

        for k in range(self.n_regimes):
            mask = regimes == k
            n_k = np.sum(mask)
            if n_k > 1:
                cluster_var = np.var(X[mask], axis=0)
                within_var += n_k * cluster_var
                total_points += n_k

        if total_points > 0:
            within_var /= total_points
            variance_reduction = total_var - within_var
            variance_reduction = np.maximum(variance_reduction, 0)

            # Normalize to create weights
            if np.sum(variance_reduction) > 0:
                self.feature_weights_ = variance_reduction / np.sum(variance_reduction)
            else:
                self.feature_weights_ = np.ones(n_features) / n_features

    def fit_online(self, returns):
        """
        Fit the model in an online/streaming fashion
        Processes data sequentially without look-ahead

        Args:
            returns: pandas Series of returns (for walk-forward validation)

        Returns:
            self
        """
        logger.info(f"Starting online fit for {len(returns)} time steps")

        self.regimes_ = []
        self.regime_probabilities_ = []
        self.features_buffer_ = []
        self.regimes_buffer_ = []

        # Warmup period to initialize scaler and model
        warmup_period = 30

        for t in range(len(returns)):
            # Get historical returns up to time t
            returns_history = returns.iloc[:t+1]

            # Calculate features using only past data
            features = self._calculate_features_single(returns_history)

            if features is None or np.any(np.isnan(features)):
                # Not enough data yet
                self.regimes_.append(0)
                self.regime_probabilities_.append(np.array([1.0, 0.0]))
                continue

            # Update scaler incrementally and transform
            if t < warmup_period:
                # During warmup, just accumulate data
                self.scaler.partial_fit(features.reshape(1, -1))
                x_scaled = self.scaler.transform(features.reshape(1, -1))[0]
                self.features_buffer_.append(x_scaled)
                self.regimes_.append(0)
                self.regime_probabilities_.append(np.array([1.0, 0.0]))
                self.regimes_buffer_.append(0)

                # Initialize model at end of warmup
                if t == warmup_period - 1:
                    self._initialize_model(np.array(self.features_buffer_))

                continue

            # Scale features using current statistics (no future data)
            x_scaled = self.scaler.transform(features.reshape(1, -1))[0]

            # Predict regime using forward-only logic
            prev_regime = self.regimes_[-1] if len(self.regimes_) > 0 else None
            regime, probs = self.predict_regime_online(x_scaled, prev_regime)

            # Store results
            self.regimes_.append(regime)
            self.regime_probabilities_.append(probs)
            self.features_buffer_.append(x_scaled)
            self.regimes_buffer_.append(regime)

            # Periodically update model parameters
            if t % self.update_frequency == 0:
                self._update_parameters()

            self.t_ = t

        # Convert to pandas Series
        self.regimes_ = pd.Series(self.regimes_, index=returns.index)

        logger.info(f"Online fit complete. Detected {self.regimes_.nunique()} unique regimes")
        logger.info(f"Regime distribution: {self.regimes_.value_counts().to_dict()}")

        return self

    def predict_next(self, returns_history):
        """
        Predict regime for next time step given historical returns
        Used for live/production inference

        Args:
            returns_history: pandas Series of historical returns

        Returns:
            regime: Predicted regime
            probability: Confidence in prediction
        """
        features = self._calculate_features_single(returns_history)

        if features is None or np.any(np.isnan(features)):
            return self.regimes_.iloc[-1] if len(self.regimes_) > 0 else 0, 0.5

        # Scale features
        x_scaled = self.scaler.transform(features.reshape(1, -1))[0]

        # Predict regime
        prev_regime = self.regimes_.iloc[-1] if len(self.regimes_) > 0 else 0
        regime, probs = self.predict_regime_online(x_scaled, prev_regime)

        return regime, probs[regime]


class BlackLittermanOptimizer:
    """
    Black-Litterman implementation (unchanged from original - no data leakage here)
    """

    def __init__(self, risk_aversion=3.0, tau=0.025):
        self.risk_aversion = risk_aversion
        self.tau = tau

    def optimize(self, expected_factor_returns, cov_matrix, market_weights=None,
                 confidence_multiplier=1.0):
        """
        Optimize portfolio weights using Black-Litterman model
        """
        assets = list(cov_matrix.index)
        N = len(assets)

        # Set up benchmark weights
        if market_weights is None:
            market_weights = pd.Series(1.0/N, index=assets)

        w_bmk = market_weights.reindex(assets).values
        Sigma = cov_matrix.reindex(assets, columns=assets).values

        # Calculate prior returns
        pi = self.risk_aversion * np.dot(Sigma, w_bmk)

        # Construct view matrix
        valid_factors = [f for f in expected_factor_returns.keys() if f in assets]
        K = len(valid_factors)

        if K == 0:
            logger.warning("No valid factor views, returning benchmark weights")
            return pd.Series(w_bmk, index=assets)

        market_asset = 'SPY' if 'SPY' in assets else assets[0]
        market_idx = assets.index(market_asset)

        P = np.zeros((K, N))
        v = np.zeros(K)

        for i, factor in enumerate(valid_factors):
            factor_idx = assets.index(factor)
            P[i, factor_idx] = 1.0
            P[i, market_idx] = -1.0
            v[i] = expected_factor_returns[factor]

        # Calculate view uncertainty
        PΣP_T = np.dot(P, np.dot(Sigma, P.T))
        Omega = confidence_multiplier * np.diag(np.diag(PΣP_T))

        # Calculate λ
        try:
            matrix_to_invert = PΣP_T + Omega / self.tau
            inv_matrix = np.linalg.inv(matrix_to_invert)
            P_pi = np.dot(P, pi)
            view_error = v - P_pi
            lambda_vec = (1.0 / self.risk_aversion) * np.dot(inv_matrix, view_error)
        except np.linalg.LinAlgError as e:
            logger.error(f"Matrix inversion failed: {e}")
            return pd.Series(w_bmk, index=assets)

        # Calculate optimal weights
        P_T = P.T
        active_weights = np.dot(P_T, lambda_vec)
        w_optimal = w_bmk + active_weights

        # Apply constraints
        w_optimal = np.maximum(w_optimal, 0.001)
        w_optimal = w_optimal / np.sum(w_optimal)

        return pd.Series(w_optimal, index=assets)


class HelixFactorStrategyFixed:
    """
    Helix 1.1 Strategy with proper walk-forward validation (FIXED VERSION)
    """

    def __init__(self, lookback_days=252, rebalance_threshold=0.02,
                 regime_update_frequency=20):
        self.factor_etfs = {
            'SPY': 'SPDR S&P 500 ETF Trust',
            'QUAL': 'iShares MSCI USA Quality Factor ETF',
            'MTUM': 'iShares MSCI USA Momentum Factor ETF',
            'USMV': 'iShares MSCI USA Min Vol Factor ETF',
            'VLUE': 'iShares MSCI USA Value Factor ETF',
            'SIZE': 'iShares MSCI USA Size Factor ETF',
            'IWF': 'Russell 1000 Growth Index'
        }

        self.lookback_days = lookback_days
        self.rebalance_threshold = rebalance_threshold
        self.regime_update_frequency = regime_update_frequency
        self.regime_models = {}
        self.optimizer = BlackLittermanOptimizer()
        self.current_weights = None
        self.data = None

    def fetch_data(self, start_date, end_date):
        """Fetch ETF price data"""
        logger.info(f"Fetching data for {list(self.factor_etfs.keys())}")

        try:
            raw_data = yf.download(list(self.factor_etfs.keys()),
                                 start=start_date, end=end_date, auto_adjust=True)

            if raw_data.empty:
                raise ValueError("No data retrieved")

            if hasattr(raw_data.columns, 'nlevels') and raw_data.columns.nlevels > 1:
                data = raw_data['Close']
            else:
                data = raw_data

            if isinstance(data, pd.Series):
                data = data.to_frame(list(self.factor_etfs.keys())[0])

            data = data.fillna(method='ffill').dropna()
            logger.info(f"Retrieved {len(data)} days of data")

            self.data = data
            return data

        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise

    def walk_forward_backtest(self, start_date, end_date, initial_training_days=252):
        """
        Walk-forward backtest with proper temporal validation

        This ensures NO data leakage:
        - Models are retrained periodically using only past data
        - Regime detection is done online/incrementally
        - No future data is used for any decision
        """
        logger.info(f"Starting walk-forward backtest from {start_date} to {end_date}")

        # Fetch data
        data = self.fetch_data(start_date, end_date)
        returns = data.pct_change().dropna()

        # Initialize tracking
        portfolio_values = []
        weights_history = []
        rebalance_dates = []
        regime_history = {etf: [] for etf in self.factor_etfs.keys()}

        initial_value = 100000
        current_value = initial_value

        # Start after initial training period
        start_idx = initial_training_days

        logger.info(f"Initial training period: {initial_training_days} days")
        logger.info(f"Testing period: {len(returns) - start_idx} days")

        for i in range(start_idx, len(returns)):
            current_date = returns.index[i]

            # IMPORTANT: Only use data UP TO time i (no future data!)
            historical_returns = returns.iloc[:i]

            # Retrain/update regime models periodically
            if i == start_idx or i % self.regime_update_frequency == 0:
                logger.info(f"Updating regime models at {current_date}")
                self._update_regime_models_online(historical_returns)

            # Get current regime for each factor (using only past data)
            current_regimes = {}
            for etf in self.factor_etfs.keys():
                if etf in self.regime_models:
                    # Predict regime using only historical data
                    regime = self.regime_models[etf].regimes_.iloc[-1]
                    current_regimes[etf] = regime
                    regime_history[etf].append((current_date, regime))
                else:
                    current_regimes[etf] = 0
                    regime_history[etf].append((current_date, 0))

            # Calculate expected returns using only past data
            recent_returns = historical_returns.iloc[-self.lookback_days:]
            expected_returns = self._generate_expected_returns(recent_returns, current_regimes)

            # Optimize portfolio using only past data
            try:
                cov_matrix = recent_returns.cov()
                new_weights = self.optimizer.optimize(expected_returns, cov_matrix)
            except Exception as e:
                logger.warning(f"Optimization failed on {current_date}: {e}")
                new_weights = pd.Series(1.0/len(self.factor_etfs),
                                       index=self.factor_etfs.keys())

            # Check if rebalancing needed
            if self._should_rebalance(new_weights):
                self.current_weights = new_weights.copy()
                rebalance_dates.append(current_date)
                weights_history.append((current_date, new_weights.copy()))

            # Calculate portfolio return for TODAY (time i)
            # This is the key: we make decision at time i-1, observe return at time i
            if self.current_weights is not None:
                daily_returns = returns.iloc[i]
                portfolio_return = (self.current_weights * daily_returns).sum()
                current_value = current_value * (1 + portfolio_return)

            portfolio_values.append(current_value)

        # Create results
        results_dates = returns.index[start_idx:]
        portfolio_series = pd.Series(portfolio_values, index=results_dates)

        # Calculate metrics
        total_return = (current_value - initial_value) / initial_value
        portfolio_returns = portfolio_series.pct_change().dropna()

        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
                       if portfolio_returns.std() > 0 else 0)
        max_dd = self._calculate_max_drawdown(portfolio_series)

        results = {
            'portfolio_values': portfolio_series,
            'weights_history': weights_history,
            'rebalance_dates': rebalance_dates,
            'regime_history': regime_history,
            'total_return': total_return,
            'annualized_volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'n_rebalances': len(rebalance_dates)
        }

        logger.info("=" * 60)
        logger.info("WALK-FORWARD BACKTEST RESULTS (NO DATA LEAKAGE)")
        logger.info("=" * 60)
        logger.info(f"Total Return: {total_return:.2%}")
        logger.info(f"Annualized Volatility: {volatility:.2%}")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: {max_dd:.2%}")
        logger.info(f"Number of Rebalances: {len(rebalance_dates)}")
        logger.info("=" * 60)

        return results

    def _update_regime_models_online(self, returns):
        """Update regime models using online learning (no data leakage)"""
        for etf in returns.columns:
            if etf not in self.regime_models:
                # Create new online model
                self.regime_models[etf] = OnlineSparseJumpModel(
                    update_frequency=self.regime_update_frequency
                )

            # Fit model in online fashion using only provided (historical) returns
            try:
                self.regime_models[etf].fit_online(returns[etf])
            except Exception as e:
                logger.warning(f"Failed to fit online model for {etf}: {e}")
                # Create dummy model
                dummy_model = OnlineSparseJumpModel()
                dummy_model.regimes_ = pd.Series(0, index=returns.index)
                self.regime_models[etf] = dummy_model

    def _generate_expected_returns(self, returns, current_regimes):
        """Generate expected returns based on current regimes"""
        expected_returns = pd.Series(index=returns.columns, dtype=float)

        for etf in returns.columns:
            regime = current_regimes.get(etf, 0)
            etf_returns = returns[etf]

            if len(etf_returns) == 0:
                expected_returns[etf] = 0.0
                continue

            if regime == 1:  # Positive regime
                positive_returns = etf_returns[etf_returns > 0]
                if len(positive_returns) > 0:
                    expected_returns[etf] = positive_returns.tail(20).mean()
                else:
                    expected_returns[etf] = etf_returns.tail(20).mean()
            else:  # Negative/neutral regime
                expected_returns[etf] = etf_returns.tail(60).mean() * 0.5

        return expected_returns

    def _should_rebalance(self, new_weights):
        """Determine if portfolio should be rebalanced"""
        if self.current_weights is None:
            return True

        weight_changes = abs(new_weights - self.current_weights)
        max_change = weight_changes.max()

        return max_change > self.rebalance_threshold

    def _calculate_max_drawdown(self, portfolio_series):
        """Calculate maximum drawdown"""
        if len(portfolio_series) == 0:
            return 0.0

        cumulative = portfolio_series / portfolio_series.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()


if __name__ == "__main__":
    # Example usage with proper walk-forward validation
    strategy = HelixFactorStrategyFixed()

    # Run walk-forward backtest
    results = strategy.walk_forward_backtest('2023-01-01', '2025-08-31',
                                             initial_training_days=252)

    print("\n" + "=" * 60)
    print("HELIX 1.1 STRATEGY - FIXED VERSION (NO DATA LEAKAGE)")
    print("=" * 60)
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Rebalances: {results['n_rebalances']}")
    print("=" * 60)

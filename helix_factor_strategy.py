"""
Helix 1.1: Factor-Based Portfolio Optimization Strategy

Dynamic factor allocation using Sparse Jump Models for regime identification,
aligned with Princeton paper: "Dynamic Factor Allocation Leveraging Regime-Switching
Signals" (arXiv:2410.14841v1).
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import logging

try:
    from data.market_data import fetch_market_data, FACTOR_ETFS, MARKET_ETF
except ImportError:
    import sys
    import os
    _root = os.path.dirname(os.path.abspath(__file__))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from data.market_data import fetch_market_data, FACTOR_ETFS, MARKET_ETF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Paper defaults (Phase 3)
DEFAULT_JUMP_PENALTY = 50.0   # λ
DEFAULT_SPARSITY_PARAM = 9.5  # κ²
EXPECTED_RETURN_CAP = 0.05    # ±5% p.a. (Phase 4)
COV_HALFLIFE = 126            # EWMA halflife days (Phase 5)
RISK_AVERSION = 2.5           # δ per paper (Phase 5)
TXN_COST_BPS = 5.0            # 5 bps per side (Phase 5)


def compute_active_returns(returns, market_col='SPY'):
    """
    Compute factor active returns = factor return - market return.
    Returns DataFrame with 6 factor columns (excludes market).
    """
    if market_col not in returns.columns:
        raise ValueError("Market column '%s' not in returns" % market_col)
    market_ret = returns[market_col]
    active = pd.DataFrame(index=returns.index)
    for col in returns.columns:
        if col != market_col:
            active[col] = returns[col] - market_ret
    return active


def _rsi(series, window):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / (loss + 1e-12)
    return (100 - (100 / (1 + rs))).fillna(50).values


def _stoch_k(series, window):
    low = series.rolling(window).min()
    high = series.rolling(window).max()
    return (100 * (series - low) / (high - low + 1e-10)).values


def _macd(series, fast, slow):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    return (ema_fast - ema_slow).values


class SparseJumpModel:
    """
    Sparse Jump Model per Princeton paper. Uses ~20 features (Exhibit 3).
    """
    
    def __init__(self, n_regimes=2, jump_penalty=None, sparsity_param=None, max_iter=100, tol=1e-6):
        self.n_regimes = n_regimes
        self.jump_penalty = jump_penalty if jump_penalty is not None else DEFAULT_JUMP_PENALTY
        self.sparsity_param = sparsity_param if sparsity_param is not None else DEFAULT_SPARSITY_PARAM
        self.max_iter = max_iter
        self.tol = tol
        
        self.scaler = StandardScaler()
        self.feature_weights_ = None
        self.centroids_ = None
        self.regimes_ = None
        self.converged_ = False
        
    def _calculate_features(self, active_return, market_data, factor_name, market_returns):
        """
        ~20 features per paper Exhibit 3.
        Factor-specific: EWMA(8,21,63), RSI(8,21,63), %K(8,21,63), MACD(8,21),(21,63), DD(21), active beta(21).
        Market-environment: market EWMA(21), VIX(log,diff,EWMA21), 2Y diff EWMA21, 10Y-2Y slope diff EWMA21.
        """
        idx = active_return.index
        ar = active_return.reindex(market_data.index).ffill().bfill()
        mr = market_returns.reindex(market_data.index).ffill().bfill()
        md = market_data.reindex(ar.index).ffill().bfill()
        common = ar.index.intersection(mr.index).intersection(md.index)
        ar = ar.loc[common].dropna()
        mr = mr.loc[common].dropna()
        md = md.loc[common].dropna()
        common = ar.index.intersection(mr.index).intersection(md.index)
        ar = ar.loc[common]
        mr = mr.loc[common]
        md = md.loc[common]
        feat_list = []
        for w in [8, 21, 63]:
            feat_list.append(ar.ewm(span=w).mean().values)
        for w in [8, 21, 63]:
            feat_list.append(_rsi(ar, w))
        for w in [8, 21, 63]:
            feat_list.append(_stoch_k(ar, w))
        feat_list.append(_macd(ar, 8, 21))
        feat_list.append(_macd(ar, 21, 63))
        log_ret = np.log(1 + ar)
        downside = np.where(ar < 0, log_ret ** 2, 0)
        dd_21 = pd.Series(downside, index=ar.index).rolling(21).mean()
        feat_list.append(np.sqrt(dd_21.fillna(0)).values)
        cov_21 = (ar * mr).rolling(21).mean() - ar.rolling(21).mean() * mr.rolling(21).mean()
        var_m_21 = mr.rolling(21).var().replace(0, np.nan)
        beta = (cov_21 / var_m_21).fillna(0)
        feat_list.append(beta.values)
        feat_list.append(mr.ewm(span=21).mean().values)
        vix_log = np.log(md['vix'].replace(0, np.nan).ffill() + 1)
        feat_list.append(vix_log.values)
        feat_list.append(md['vix'].diff().values)
        feat_list.append(md['vix'].ewm(span=21).mean().values)
        y2d = md['yield_2y'].diff().ewm(span=21).mean()
        feat_list.append(y2d.values)
        slope = md['yield_10y'] - md['yield_2y']
        slope_diff = slope.diff().ewm(span=21).mean()
        feat_list.append(slope_diff.values)
        n = len(ar)
        features = np.column_stack([f[:n] if len(f) >= n else np.resize(f, n) for f in feat_list])
        features = pd.DataFrame(features, index=ar.index)
        return features.reindex(idx).values
    
    def _calculate_features_legacy(self, returns):
        """Legacy 4-feature set when market data unavailable"""
        w = 20
        return np.column_stack([
            returns.ewm(span=w).mean().values,
            _rsi(returns if isinstance(returns, pd.Series) else pd.Series(returns), 14),
            returns.rolling(w).sum().values,
            returns.rolling(w).std().values
        ])
    
    def fit(self, active_return, market_data=None, market_returns=None, factor_name=None):
        """Fit the sparse jump model using alternating optimization"""
        if market_data is None or market_returns is None:
            features = self._calculate_features_legacy(active_return)
        else:
            features = self._calculate_features(
                active_return, market_data, factor_name or 'factor', market_returns
            )
        valid_idx = ~np.isnan(features).any(axis=1)
        features_clean = features[valid_idx]
        
        if len(features_clean) == 0:
            raise ValueError("No valid data after feature calculation")
        
        # Standardize features
        X = self.scaler.fit_transform(features_clean)
        n_obs, n_features = X.shape
        
        # Initialize feature weights (uniform)
        w = np.ones(n_features) / n_features
        
        # Initialize centroids randomly
        np.random.seed(42)
        centroids = np.random.randn(self.n_regimes, n_features)
        
        # Initialize regime sequence
        regimes = np.random.randint(0, self.n_regimes, n_obs)
        
        prev_objective = np.inf
        
        for iteration in range(self.max_iter):
            # Step 1: Optimize regime sequence given current weights and centroids
            regimes = self._optimize_regimes(X, centroids, w)
            
            # Step 2: Update centroids given current regimes and weights
            centroids = self._update_centroids(X, regimes, w)
            
            # Step 3: Update feature weights given current regimes and centroids
            w = self._update_weights(X, regimes, centroids)
            
            # Check convergence
            current_objective = self._calculate_objective(X, regimes, centroids, w)
            
            if abs(prev_objective - current_objective) < self.tol:
                self.converged_ = True
                break
                
            prev_objective = current_objective
            
            if iteration % 10 == 0:
                logger.info(f"SJM Iteration {iteration}, Objective: {current_objective:.6f}")
        
        # Store results
        self.feature_weights_ = w
        self.centroids_ = centroids
        
        # Map back to original time series
        self.regimes_ = pd.Series(index=active_return.index, dtype=float)
        self.regimes_.iloc[valid_idx] = regimes
        self.regimes_ = self.regimes_.ffill().fillna(0)
        
        return self
    
    def _optimize_regimes(self, X, centroids, weights):
        """
        Optimize regime sequence using dynamic programming approach
        This solves: min Σ w^T ||x_t - c_{s_t}||² + γ Σ I(s_t ≠ s_{t-1})
        """
        n_obs, n_features = X.shape
        n_states = self.n_regimes
        
        # Dynamic programming matrices
        # cost[t][k] = minimum cost to reach state k at time t
        cost = np.full((n_obs, n_states), np.inf)
        path = np.zeros((n_obs, n_states), dtype=int)
        
        # Initialize first time step
        for k in range(n_states):
            weighted_diff = weights * (X[0] - centroids[k])
            cost[0][k] = np.sum(weighted_diff ** 2)
        
        # Forward pass
        for t in range(1, n_obs):
            for k in range(n_states):
                # Calculate weighted distance to centroid k
                weighted_diff = weights * (X[t] - centroids[k])
                dist_cost = np.sum(weighted_diff ** 2)
                
                # Find best previous state
                for prev_k in range(n_states):
                    jump_cost = self.jump_penalty if prev_k != k else 0
                    total_cost = cost[t-1][prev_k] + dist_cost + jump_cost
                    
                    if total_cost < cost[t][k]:
                        cost[t][k] = total_cost
                        path[t][k] = prev_k
        
        # Backward pass to find optimal path
        regimes = np.zeros(n_obs, dtype=int)
        regimes[-1] = np.argmin(cost[-1])
        
        for t in range(n_obs - 2, -1, -1):
            regimes[t] = path[t + 1][regimes[t + 1]]
        
        return regimes
    
    def _update_centroids(self, X, regimes, weights):
        """Update centroids as weighted means of assigned points"""
        n_features = X.shape[1]
        centroids = np.zeros((self.n_regimes, n_features))
        
        for k in range(self.n_regimes):
            mask = regimes == k
            if np.sum(mask) > 0:
                # Simple centroid calculation (unweighted mean)
                # Note: Feature weighting is handled in the distance calculations, not here
                X_k = X[mask]
                centroids[k] = np.mean(X_k, axis=0)
            else:
                # If no points assigned, keep previous centroid or random
                centroids[k] = np.random.randn(n_features)
        
        return centroids
    
    def _update_weights(self, X, regimes, centroids):
        """
        Update feature weights based on clustering effect
        Features that reduce variance more get higher weights
        """
        n_features = X.shape[1]
        variance_reduction = np.zeros(n_features)
        
        # Calculate total variance for each feature
        total_var = np.var(X, axis=0)
        
        # Calculate within-cluster variance for each feature
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
        
        # Normalize to create weights (this is a simplified version)
        # In practice, you'd solve: max w^T variance_reduction s.t. ||w||_1 <= sparsity_param
        if np.sum(variance_reduction) > 0:
            weights = variance_reduction / np.sum(variance_reduction)
            
            # Apply L1 constraint (simplified)
            if np.sum(weights) > self.sparsity_param:
                # Sort and keep top features
                sorted_idx = np.argsort(weights)[::-1]
                cumsum = np.cumsum(weights[sorted_idx])
                n_keep = np.sum(cumsum <= self.sparsity_param)
                n_keep = max(n_keep, 1)  # Keep at least one feature
                
                new_weights = np.zeros(n_features)
                new_weights[sorted_idx[:n_keep]] = weights[sorted_idx[:n_keep]]
                weights = new_weights / np.sum(new_weights)
        else:
            weights = np.ones(n_features) / n_features
        
        return weights
    
    def _calculate_objective(self, X, regimes, centroids, weights):
        """Calculate the total SJM objective function value"""
        n_obs = len(regimes)
        
        # Clustering cost
        clustering_cost = 0
        for t in range(n_obs):
            k = regimes[t]
            weighted_diff = weights * (X[t] - centroids[k])
            clustering_cost += np.sum(weighted_diff ** 2)
        
        # Jump cost
        jump_cost = 0
        for t in range(1, n_obs):
            if regimes[t] != regimes[t-1]:
                jump_cost += self.jump_penalty
        
        return clustering_cost + jump_cost
    
    def predict_regime(self, returns):
        """Predict regime for new data using online inference"""
        if self.regimes_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        # This would implement the online inference algorithm from the paper
        # For now, return the last regime (simplified)
        last_regime = self.regimes_.iloc[-1] if len(self.regimes_) > 0 else 0
        return pd.Series([last_regime] * len(returns), index=returns.index)


class BlackLittermanOptimizer:
    """
    Black-Litterman implementation matching the exact LaTeX formulation from the paper
    """
    
    def __init__(self, risk_aversion=3.0, tau=0.025):
        self.risk_aversion = risk_aversion  # δ (delta) - scalar
        self.tau = tau  # τ (tau) - scalar
    
    def optimize(self, expected_factor_returns, cov_matrix, market_weights=None, 
                 confidence_multiplier=1.0):
        """
        Parameters:
        - expected_factor_returns: dict {factor_name: expected_active_return}
        - cov_matrix: pandas DataFrame, N×N covariance matrix of all assets
        - market_weights: pandas Series, benchmark weights (default: equal weights)
        - confidence_multiplier: scalar, higher = less confident in views
        
        Returns:
        - pandas Series with optimal weights
        """
        
        assets = list(cov_matrix.index)
        N = len(assets)  # Number of assets (7)
        
        # Step 1: Set up benchmark weights w^bmk
        if market_weights is None:
            market_weights = pd.Series(1.0/N, index=assets)
        
        w_bmk = market_weights.reindex(assets).values  # Ensure correct order
        
        # Step 2: Convert covariance matrix Σ (N×N)
        Sigma = cov_matrix.reindex(assets, columns=assets).values
        
        # Step 3: Calculate prior returns π = δ Σ w^bmk
        pi = self.risk_aversion * np.dot(Sigma, w_bmk)  # N×1
        
        # Step 4: Construct view portfolio matrix P (K×N)
        valid_factors = [f for f in expected_factor_returns.keys() if f in assets]
        K = len(valid_factors)  # Number of views (6)
        
        if K == 0:
            logger.warning("No valid factor views found, returning benchmark weights")
            return pd.Series(w_bmk, index=assets)
        
        # Identify market asset (assume SPY or first asset if SPY not found)
        market_asset = 'SPY' if 'SPY' in assets else assets[0]
        market_idx = assets.index(market_asset)
        
        # Build P matrix: K×N where each row represents one view portfolio
        P = np.zeros((K, N))  # K×N (6×7)
        v = np.zeros(K)       # K×1 (6×1) - expected view returns
        
        for i, factor in enumerate(valid_factors):
            factor_idx = assets.index(factor)
            P[i, factor_idx] = 1.0   # Long the factor
            P[i, market_idx] = -1.0  # Short the market
            v[i] = expected_factor_returns[factor]  # Expected active return
        
        logger.info(f"Constructed P matrix: {K}×{N}, v vector: {K}×1")
        
        # Step 5: Calculate view uncertainty matrix Ω (K×K diagonal)
        # Following common practice: Ω_ii = c × (PΣP^T)_ii
        PΣP_T = np.dot(P, np.dot(Sigma, P.T))  # K×K (6×6)
        Omega = confidence_multiplier * np.diag(np.diag(PΣP_T))  # K×K diagonal
        
        # Step 6: Calculate λ using the EXACT LaTeX formula
        # λ = δ^(-1) (PΣP^T + Ω/τ)^(-1) (v - Pπ)
        try:
            # Matrix to invert: (PΣP^T + Ω/τ) - this is K×K
            matrix_to_invert = PΣP_T + Omega / self.tau  # 6×6 + 6×6 = 6×6
            
            # Invert the K×K matrix
            inv_matrix = np.linalg.inv(matrix_to_invert)  # 6×6
            
            # Calculate (v - Pπ)
            P_pi = np.dot(P, pi)  # K×1 (6×1)
            view_error = v - P_pi  # K×1 (6×1)
            
            # Calculate λ
            lambda_vec = (1.0 / self.risk_aversion) * np.dot(inv_matrix, view_error)  # K×1
            
            # logger.info(f"Lambda vector calculated successfully: shape {lambda_vec.shape}")
            
        except np.linalg.LinAlgError as e:
            logger.error(f"Matrix inversion failed: {e}")
            logger.warning("Falling back to benchmark weights")
            return pd.Series(w_bmk, index=assets)
        
        # Step 7: Calculate optimal weights w^BL = w^bmk + P^T λ
        P_T = P.T  # N×K (7×6)
        active_weights = np.dot(P_T, lambda_vec)  # N×1 (7×1)
        w_optimal = w_bmk + active_weights  # N×1 + N×1 = N×1
        
        # Step 8: Apply constraints (long-only, fully invested)
        w_optimal = np.maximum(w_optimal, 0.001)  # Long-only constraint
        w_optimal = w_optimal / np.sum(w_optimal)  # Fully invested constraint
        
        # Log optimization results
        active_weight_norm = np.linalg.norm(active_weights)
        weight_change_norm = np.linalg.norm(w_optimal - w_bmk)
        
        logger.info(f"Black-Litterman optimization successful:")
        # logger.info(f"  Active weights magnitude: {active_weight_norm:.4f}")
        # logger.info(f"  Total weight change: {weight_change_norm:.4f}")
        # logger.info(f"  Final weights sum: {np.sum(w_optimal):.6f}")
        
        return pd.Series(w_optimal, index=assets)
    
    def calculate_single_factor_lambda(self, factor_expected_return, factor_asset, 
                                       market_asset, cov_matrix, confidence_multiplier=1.0):
        """
        Calculate λ_j for a single factor using Equation (3) from the paper
        This is used for the hypothetical long-short strategy
        
        λ_j = (1/δ) × (v_j - P_j^T μ^BL_{-j}) / η_j
        """
        assets = list(cov_matrix.index)
        N = len(assets)
        
        # Construct single view portfolio P_j
        P_j = np.zeros(N)
        factor_idx = assets.index(factor_asset)
        market_idx = assets.index(market_asset)
        P_j[factor_idx] = 1.0
        P_j[market_idx] = -1.0
        
        Sigma = cov_matrix.reindex(assets, columns=assets).values
        
        # Calculate η_j using the full formula from the LaTeX
        # η_j = P_j^T Σ P_j + ω_j/τ - P_j^T Σ P^T (PΣP^T + Ω/τ)^(-1) P Σ P_j
        
        # For single view case, this simplifies significantly
        risk_term = np.dot(P_j, np.dot(Sigma, P_j))  # P_j^T Σ P_j
        omega_j = confidence_multiplier * risk_term
        eta_j = risk_term + omega_j / self.tau
        
        # For single view, μ^BL_{-j} ≈ π (prior returns)
        w_equal = np.ones(N) / N
        pi = self.risk_aversion * np.dot(Sigma, w_equal)
        prior_view_return = np.dot(P_j, pi)
        
        # Calculate λ_j
        lambda_j = (1.0 / self.risk_aversion) * (factor_expected_return - prior_view_return) / eta_j
        
        return lambda_j, P_j, eta_j


class HelixFactorStrategy:
    """
    Helix 1.1: Factor-based portfolio optimization with daily rebalancing
    """
    
    def __init__(self, lookback_days=252, rebalance_threshold=0.02):
        self.factor_etfs = {
            'SPY': 'SPDR S&P 500 ETF Trust',  # Market benchmark
            'QUAL': 'iShares MSCI USA Quality Factor ETF',
            'MTUM': 'iShares MSCI USA Momentum Factor ETF', 
            'USMV': 'iShares MSCI USA Min Vol Factor ETF',
            'VLUE': 'iShares MSCI USA Value Factor ETF',
            'SIZE': 'iShares MSCI USA Size Factor ETF',
            'IWF': 'Russell 1000 Growth Index'
        }
        
        self.lookback_days = lookback_days
        self.rebalance_threshold = rebalance_threshold
        self.regime_models = {}
        self.regime_means = {}   # factor -> {regime: mean_active_return_annualized}
        self.optimizer = BlackLittermanOptimizer(risk_aversion=RISK_AVERSION)
        self.current_weights = None
        self.data = None
        self.market_data = None
        
    def fetch_data(self, start_date, end_date):
        """Fetch ETF price data"""
        logger.info("Fetching data for %s" % list(self.factor_etfs.keys()))
        
        try:
            raw_data = yf.download(list(self.factor_etfs.keys()), 
                                 start=start_date, end=end_date, auto_adjust=True)
            
            if raw_data.empty:
                raise ValueError("No data retrieved")
            
            # Extract Close prices (which are auto-adjusted)
            if hasattr(raw_data.columns, 'nlevels') and raw_data.columns.nlevels > 1:
                # Multi-level columns - get 'Close' prices
                data = raw_data['Close']
            else:
                # Single ETF case
                data = raw_data
            
            # Handle single ETF case
            if isinstance(data, pd.Series):
                data = data.to_frame(list(self.factor_etfs.keys())[0])
            
            # Forward fill missing data
            data = data.ffill().dropna()
            logger.info("Retrieved {} days of data".format(len(data)))
            
            self.data = data
            try:
                self.market_data = fetch_market_data(start_date, end_date)
                self.market_data = self.market_data.reindex(data.index).ffill().bfill()
            except Exception as e:
                logger.warning("Market data fetch failed, using legacy features: {}".format(e))
                self.market_data = None
            return data
            
        except Exception as e:
            logger.error("Error fetching data: {}".format(e))
            raise
    
    def calculate_returns(self):
        """Calculate daily returns"""
        if self.data is None:
            raise ValueError("No data available. Run fetch_data() first.")
        
        returns = self.data.pct_change().dropna()
        return returns
    
    def fit_regime_models(self, active_returns, market_returns, sjm_config=None):
        """
        Fit 6 SJMs (one per factor) on active returns. No model for SPY.
        sjm_config: optional dict of factor -> {jump_penalty, sparsity_param} for per-factor overrides.
        """
        logger.info("Fitting regime models on active returns (6 factors)...")
        self.regime_means = {}
        market_ret = market_returns.reindex(active_returns.index).ffill().bfill()
        
        for factor in active_returns.columns:
            logger.info("Fitting regime model for {}".format(factor))
            cfg = (sjm_config or {}).get(factor, {})
            model = SparseJumpModel(
                jump_penalty=cfg.get('jump_penalty', DEFAULT_JUMP_PENALTY),
                sparsity_param=cfg.get('sparsity_param', DEFAULT_SPARSITY_PARAM)
            )
            try:
                if self.market_data is not None and not self.market_data.empty:
                    model.fit(
                        active_returns[factor],
                        market_data=self.market_data,
                        market_returns=market_ret,
                        factor_name=factor
                    )
                else:
                    model.fit(active_returns[factor])
                self.regime_models[factor] = model
                regimes = model.regimes_
                means = {}
                for k in range(model.n_regimes):
                    mask = (regimes == k) & regimes.index.isin(active_returns.index)
                    if mask.sum() > 0:
                        mu = active_returns.loc[mask, factor].mean() * 252
                        means[k] = np.clip(mu, -EXPECTED_RETURN_CAP, EXPECTED_RETURN_CAP)
                    else:
                        means[k] = 0.0
                self.regime_means[factor] = means
            except Exception as e:
                logger.warning("Failed to fit model for {}: {}".format(factor, e))
                dummy = SparseJumpModel()
                dummy.regimes_ = pd.Series(0, index=active_returns.index)
                self.regime_models[factor] = dummy
                self.regime_means[factor] = {0: 0.0}
    
    def generate_expected_returns(self, current_regimes):
        """
        Expected active returns from regime historical averages (Phase 4).
        Returns dict {factor: expected_active_return} for the 6 factors.
        Stored as annualized; convert to daily for BL (covariance is daily).
        """
        expected = {}
        for factor in self.regime_models:
            regime = int(current_regimes.get(factor, 0))
            means = self.regime_means.get(factor, {0: 0.0})
            ann = means.get(regime, 0.0)
            expected[factor] = ann / 252.0  # daily for BL
        return expected
    
    def _ewma_covariance(self, returns):
        """EWMA covariance with halflife=126 days."""
        span = 2 * COV_HALFLIFE - 1
        alpha = 2.0 / (span + 1)
        r = returns.fillna(0).values
        cov = np.outer(r[0], r[0])
        for i in range(1, len(r)):
            cov = alpha * np.outer(r[i], r[i]) + (1 - alpha) * cov
        return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
    
    def optimize_portfolio(self, returns, expected_active_returns):
        """Optimize portfolio weights using EWMA covariance (halflife=126)"""
        cov_matrix = self._ewma_covariance(returns)
        weights = self.optimizer.optimize(expected_active_returns, cov_matrix)
        return weights
    
    def should_rebalance(self, new_weights):
        """Determine if portfolio should be rebalanced"""
        if self.current_weights is None:
            return True
        
        # Calculate weight changes
        weight_changes = abs(new_weights - self.current_weights)
        max_change = weight_changes.max()
        
        return max_change > self.rebalance_threshold
    
    def backtest(self, start_date, end_date):
        """Run backtest with paper alignment: active returns, 1-day delay, 5 bps tx costs"""
        logger.info("Starting backtest from {} to {}".format(start_date, end_date))
        
        data = self.fetch_data(start_date, end_date)
        returns = self.calculate_returns()
        active_returns = compute_active_returns(returns, market_col=MARKET_ETF)
        market_returns = returns[MARKET_ETF]
        
        self.fit_regime_models(active_returns, market_returns)
        
        portfolio_values = []
        weights_history = []
        rebalance_dates = []
        initial_value = 100000
        current_value = initial_value
        txn_cost_bps = TXN_COST_BPS / 10000
        
        start_idx = max(self.lookback_days, 252)
        
        for i in range(start_idx, len(returns)):
            current_date = returns.index[i]
            recent_returns = returns.iloc[max(0, i - self.lookback_days):i]
            
            if len(recent_returns) < 252:
                portfolio_values.append(current_value)
                continue
            
            regime_idx = max(0, i - 2)  # One-day delay: regime at T-2 applied at T
            current_regimes = {}
            for factor in self.regime_models:
                rs = self.regime_models[factor].regimes_
                if regime_idx < len(rs):
                    current_regimes[factor] = int(rs.iloc[regime_idx])
                else:
                    current_regimes[factor] = 0
            
            expected_active = self.generate_expected_returns(current_regimes)
            
            try:
                new_weights = self.optimize_portfolio(recent_returns, expected_active)
            except Exception as e:
                logger.warning("Optimization failed on {}: {}".format(current_date, e))
                new_weights = pd.Series(1.0 / len(self.factor_etfs), index=self.factor_etfs.keys())
            
            if self.should_rebalance(new_weights):
                if self.current_weights is not None:
                    weight_chg = abs(new_weights - self.current_weights).sum()
                    current_value *= (1 - txn_cost_bps * weight_chg)
                self.current_weights = new_weights.copy()
                rebalance_dates.append(current_date)
                weights_history.append((current_date, new_weights.copy()))
            
            if self.current_weights is not None and i > 0:
                daily_returns = returns.iloc[i]
                portfolio_return = (self.current_weights * daily_returns).sum()
                current_value = current_value * (1 + portfolio_return)
            
            portfolio_values.append(current_value)
        
        # Create results DataFrame
        results_dates = returns.index[start_idx:]
        portfolio_series = pd.Series(portfolio_values, index=results_dates)
        
        # Calculate performance metrics
        total_return = (current_value - initial_value) / initial_value
        portfolio_returns = portfolio_series.pct_change().dropna()
        
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252) if portfolio_returns.std() > 0 else 0
        
        max_dd = self._calculate_max_drawdown(portfolio_series)
        
        results = {
            'portfolio_values': portfolio_series,
            'weights_history': weights_history,
            'rebalance_dates': rebalance_dates,
            'total_return': total_return,
            'annualized_volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'n_rebalances': len(rebalance_dates)
        }
        
        logger.info("Backtest completed:")
        logger.info("Total Return: {:.2%}".format(total_return))
        logger.info("Sharpe Ratio: {:.2f}".format(sharpe_ratio))
        logger.info("Max Drawdown: {:.2%}".format(max_dd))
        logger.info("Number of Rebalances: {}".format(len(rebalance_dates)))
        
        return results
    
    def _calculate_max_drawdown(self, portfolio_series):
        """Calculate maximum drawdown"""
        if len(portfolio_series) == 0:
            return 0.0
        
        cumulative = portfolio_series / portfolio_series.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()


if __name__ == "__main__":
    # Example usage
    strategy = HelixFactorStrategy()
    
    # Run backtest for 2023
    results = strategy.backtest('2024-01-01', '2025-08-31')
    
    print("\n=== Helix 1.1 Strategy Results ===")
    print("Total Return: {:.2%}".format(results['total_return']))
    print("Sharpe Ratio: {:.2f}".format(results['sharpe_ratio']))
    print("Max Drawdown: {:.2%}".format(results['max_drawdown']))
    print("Rebalances: {}".format(results['n_rebalances']))
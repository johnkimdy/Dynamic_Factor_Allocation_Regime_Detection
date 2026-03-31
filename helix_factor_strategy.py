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


def _compute_bcss(X, proba, centers=None, tol=1e-6):
    """
    Between Cluster Sum of Squares per feature (author's reference).
    BCSS_j = sum_k N_k * (center_kj - X_mean_j)^2
    """
    X_arr = np.asarray(X)
    proba_arr = np.asarray(proba)
    if centers is None:
        # Weighted mean per cluster: center_k = sum_t proba[t,k]*X[t] / sum_t proba[t,k]
        Ns = proba_arr.sum(axis=0)
        Ns = np.where(Ns < tol, 1, Ns)
        centers = (proba_arr.T @ X_arr) / Ns[:, np.newaxis]
    centers = np.nan_to_num(centers, nan=0.0)
    Ns = proba_arr.sum(axis=0)
    X_mean = X_arr.mean(axis=0)
    bcss = Ns @ ((centers - X_mean) ** 2)
    bcss = np.maximum(bcss, 0)
    return bcss


def _soft_thres_l2_normalized(x, thres):
    """Soft threshold then L2 normalize. Author's sparse_jump."""
    y = np.maximum(0.0, np.asarray(x, dtype=float) - thres)
    n = np.linalg.norm(y)
    if n < 1e-12:
        return y
    return y / n


def _solve_lasso(a, norm_ub, tol=1e-8):
    """
    Lasso: max L1-norm of weights subject to L2-normalized soft-thresholding.
    Binary search for threshold so L1(soft_thres(a, t)) = norm_ub.
    Author's sparse_jump; norm_ub = kappa = sqrt(max_feats).
    """
    a_arr = np.asarray(a, dtype=float).ravel()
    if norm_ub < 1.0:
        norm_ub = 1.0
    uniq = np.unique(a_arr)
    if len(uniq) < 2:
        return np.ones_like(a_arr) / np.sqrt(len(a_arr))
    right = uniq[-2] if len(uniq) > 1 else uniq[-1]

    def l1_at_thres(thres):
        w = _soft_thres_l2_normalized(a_arr, thres)
        return np.sum(w)

    if l1_at_thres(0) <= norm_ub:
        return _soft_thres_l2_normalized(a_arr, 0)
    if l1_at_thres(right) >= norm_ub:
        return _soft_thres_l2_normalized(a_arr, right)

    lo, hi = 0.0, float(right)
    for _ in range(100):
        mid = (lo + hi) / 2
        val = l1_at_thres(mid)
        if abs(val - norm_ub) < tol:
            return _soft_thres_l2_normalized(a_arr, mid)
        if val < norm_ub:
            hi = mid
        else:
            lo = mid
    return _soft_thres_l2_normalized(a_arr, (lo + hi) / 2)


def _symmetric_jump_penalty_matrix(n_regimes, lam):
    """Λ[i,j] = λ for i≠j, 0 on diagonal (scalar jump_penalty case)."""
    L = np.full((n_regimes, n_regimes), float(lam), dtype=float)
    np.fill_diagonal(L, 0.0)
    return L


class SparseJumpModel:
    """
    Sparse Jump Model per Princeton paper. Uses ~20 features (Exhibit 3).
    Defaults follow author's reference: https://github.com/Yizhan-Oliver-Shu/jump-models
    (max_iter=30, tol=1e-8; we keep jump_penalty/sparsity_param from paper Phase 3).

    Jump costs: transition i→j costs Λ[i,j]. Pass scalar ``jump_penalty`` for symmetric
    penalties (all off-diagonals equal); or ``jump_penalty_matrix`` (n_regimes×n_regimes)
    for asymmetric transitions (JOH-12).
    """
    
    def __init__(self, n_regimes=2, jump_penalty=None, jump_penalty_matrix=None, sparsity_param=None, max_iter=30, tol=1e-8,
                 min_iter=5, record_loss_curve=False):
        self.n_regimes = n_regimes
        self.sparsity_param = sparsity_param if sparsity_param is not None else DEFAULT_SPARSITY_PARAM
        if jump_penalty_matrix is not None:
            L = np.asarray(jump_penalty_matrix, dtype=float)
            if L.shape != (n_regimes, n_regimes):
                raise ValueError(
                    "jump_penalty_matrix has shape %s; expected (%d, %d) for n_regimes=%d"
                    % (L.shape, n_regimes, n_regimes, n_regimes)
                )
            self.jump_penalty_matrix = L.copy()
            off = ~np.eye(n_regimes, dtype=bool)
            self.jump_penalty = (
                float(jump_penalty)
                if jump_penalty is not None
                else float(np.mean(self.jump_penalty_matrix[off]))
            )
        else:
            self.jump_penalty = float(jump_penalty if jump_penalty is not None else DEFAULT_JUMP_PENALTY)
            self.jump_penalty_matrix = _symmetric_jump_penalty_matrix(n_regimes, self.jump_penalty)
        self.max_iter = max_iter   # author default 30 (coordinate descent outer loop)
        self.tol = tol             # author tol_jm=1e-8 for inner JM convergence
        self.min_iter = min_iter   # don't stop before this (avoids "instant" convergence in UI)
        self.record_loss_curve = record_loss_curve
        
        self.scaler = StandardScaler()
        self.feature_weights_ = None
        self.centroids_ = None
        self.regimes_ = None
        self.converged_ = False
        self.loss_curve_ = []  # filled when record_loss_curve=True
        
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
        # VIX per Exhibit 3: "log, diff, EWMA" = EWMA of diff(log(VIX)) with window 21
        vix_safe = md['vix'].replace(0, np.nan).ffill().fillna(20)
        vix_log_diff_ewm = np.log(vix_safe + 1).diff().ewm(span=21).mean()
        feat_list.append(vix_log_diff_ewm.values)
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
    
    def fit(self, active_return, market_data=None, market_returns=None, factor_name=None,
            on_iteration_callback=None):
        """Fit the sparse jump model using alternating optimization.
        on_iteration_callback: optional callable(iteration, objective) called each iteration for live streaming."""
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
        
        # Jump penalty matrix scaled by 1/sqrt(n_features) for scale-invariance (author's reference)
        scaled_jump_matrix = self.jump_penalty_matrix / np.sqrt(n_features)
        norm_ub = np.sqrt(self.sparsity_param)  # kappa; sparsity_param = kappa^2

        prev_objective = np.inf
        self.loss_curve_ = []

        for iteration in range(self.max_iter):
            # Step 1: Optimize regime sequence given current weights and centroids
            regimes = self._optimize_regimes(X, centroids, w, scaled_jump_matrix)
            
            # Step 2: Update centroids given current regimes and weights
            centroids = self._update_centroids(X, regimes, w)

            # Step 3: Update feature weights via BCSS + Lasso (author's reference)
            w = self._update_weights_bcss_lasso(X, regimes, centroids, norm_ub)

            # Check convergence
            current_objective = self._calculate_objective(X, regimes, centroids, w, scaled_jump_matrix)
            if self.record_loss_curve:
                self.loss_curve_.append(float(current_objective))
            if on_iteration_callback is not None:
                try:
                    on_iteration_callback(iteration, float(current_objective))
                except Exception:
                    pass

            if iteration >= self.min_iter and abs(prev_objective - current_objective) < self.tol:
                self.converged_ = True
                break
                
            prev_objective = current_objective
            
            # Log every iteration for first 20, then every 10 (so dashboard/streaming shows progress)
            if iteration < 20 or iteration % 10 == 0:
                logger.info(f"SJM Iteration {iteration}, Objective: {current_objective:.6f}")
        
        # Store results
        self.feature_weights_ = w
        self.centroids_ = centroids
        
        # Map back to original time series
        self.regimes_ = pd.Series(index=active_return.index, dtype=float)
        self.regimes_.iloc[valid_idx] = regimes
        self.regimes_ = self.regimes_.ffill().fillna(0)
        
        return self
    
    def _optimize_regimes(self, X, centroids, weights, jump_penalty_matrix=None):
        """
        Optimize regime sequence using dynamic programming.
        Solves: min Σ ½‖√w ⊙ (x_t − θ_{s_t})‖² + Σ_t Λ[s_{t-1}, s_t]
        Distance uses feat_weights = sqrt(w): sum_j w_j * (x_j - c_j)^2
        """
        if jump_penalty_matrix is None:
            jump_penalty_matrix = self.jump_penalty_matrix / np.sqrt(X.shape[1])
        jump_penalty_matrix = np.asarray(jump_penalty_matrix, dtype=float)
        n_obs, n_features = X.shape
        n_states = self.n_regimes

        cost = np.full((n_obs, n_states), np.inf)
        path = np.zeros((n_obs, n_states), dtype=int)

        for k in range(n_states):
            weighted_diff = weights * (X[0] - centroids[k])
            cost[0][k] = 0.5 * np.sum(weighted_diff ** 2)

        for t in range(1, n_obs):
            for k in range(n_states):
                weighted_diff = weights * (X[t] - centroids[k])
                dist_cost = 0.5 * np.sum(weighted_diff ** 2)
                for prev_k in range(n_states):
                    jc = jump_penalty_matrix[prev_k, k]
                    total = cost[t - 1][prev_k] + dist_cost + jc
                    if total < cost[t][k]:
                        cost[t][k] = total
                        path[t][k] = prev_k

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
    
    def _update_weights_bcss_lasso(self, X, regimes, centroids, norm_ub):
        """
        BCSS + Lasso feature weights (author's reference).
        proba from hard labels; BCSS per feature; solve_lasso for sparsity.
        """
        n_features = X.shape[1]
        n_obs = len(regimes)
        proba = np.zeros((n_obs, self.n_regimes))
        proba[np.arange(n_obs), regimes] = 1.0
        centers = np.zeros((self.n_regimes, n_features))
        for k in range(self.n_regimes):
            mask = regimes == k
            if mask.sum() > 0:
                centers[k] = X[mask].mean(axis=0)
            else:
                centers[k] = X.mean(axis=0)
        bcss = _compute_bcss(X, proba, centers)
        if np.all(bcss <= 0):
            return np.ones(n_features) / np.sqrt(n_features)
        a = bcss / (bcss.max() + 1e-12)
        w = _solve_lasso(a, norm_ub)
        return w
    
    def _calculate_objective(self, X, regimes, centroids, weights, jump_penalty_matrix=None):
        """Total SJM objective: ½Σ‖√w⊙(x-θ)‖² + Σ_t Λ[s_{t-1},s_t]"""
        if jump_penalty_matrix is None:
            jump_penalty_matrix = self.jump_penalty_matrix / np.sqrt(X.shape[1])
        jump_penalty_matrix = np.asarray(jump_penalty_matrix, dtype=float)
        n_obs = len(regimes)
        cluster_cost = 0.0
        for t in range(n_obs):
            k = regimes[t]
            d = weights * (X[t] - centroids[k])
            cluster_cost += 0.5 * np.sum(d ** 2)
        jump_cost = 0.0
        for t in range(1, n_obs):
            jump_cost += jump_penalty_matrix[regimes[t - 1], regimes[t]]
        return cluster_cost + jump_cost
    
    def infer_regime_online(self, X_scaled):
        """
        Nystrup et al. online inference: solve JM over lookback window with fixed
        centroids and weights; return last optimal state. X_scaled = scaled features.
        """
        if self.centroids_ is None or self.feature_weights_ is None:
            raise ValueError("Model must be fitted before online inference")
        X = np.asarray(X_scaled)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_features = X.shape[1]
        if n_features != self.centroids_.shape[1]:
            raise ValueError("Feature dim mismatch")
        scaled_matrix = self.jump_penalty_matrix / np.sqrt(n_features)
        regimes = self._optimize_regimes(
            X, self.centroids_, self.feature_weights_, scaled_matrix
        )
        return int(regimes[-1])

    def predict_regime(self, returns):
        """Predict regime for new data (legacy; use infer_regime_online for paper alignment)"""
        if self.regimes_ is None:
            raise ValueError("Model must be fitted before prediction")
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

    def _tracking_error_annual(self, w_opt, w_bmk, Sigma, sqrt_252=np.sqrt(252)):
        """Ex-ante annualized tracking error: sqrt((w-w_bmk)^T Sigma (w-w_bmk)) * sqrt(252)."""
        diff = w_opt - w_bmk
        return float(np.sqrt(np.dot(diff, np.dot(Sigma, diff))) * sqrt_252)

    def optimize_with_te_target(self, expected_factor_returns, cov_matrix, market_weights=None,
                                target_te_annual=0.02, te_tolerance=0.005, max_iter=20):
        """
        Optimize with tracking-error targeting (paper: 1-4%). Binary search on confidence_multiplier.
        target_te_annual: e.g. 0.02 for 2%; te_tolerance: stop when within this (e.g. 0.005).
        """
        assets = list(cov_matrix.index)
        N = len(assets)
        w_bmk = market_weights.reindex(assets).values if market_weights is not None else np.ones(N) / N
        Sigma = cov_matrix.reindex(assets, columns=assets).values

        lo, hi = 0.1, 50.0
        for _ in range(max_iter):
            mid = np.sqrt(lo * hi)
            w = self.optimize(expected_factor_returns, cov_matrix, market_weights=market_weights,
                              confidence_multiplier=mid)
            w_arr = w.reindex(assets).values
            te = self._tracking_error_annual(w_arr, w_bmk, Sigma)
            if abs(te - target_te_annual) <= te_tolerance:
                return w
            if te > target_te_annual:
                hi = mid  # need more confidence (larger Omega) -> less tilt -> lower TE
            else:
                lo = mid
        w = self.optimize(expected_factor_returns, cov_matrix, market_weights=market_weights,
                          confidence_multiplier=np.sqrt(lo * hi))
        return w
    
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
    
    def __init__(self, lookback_days=252, rebalance_threshold=0.0, target_tracking_error=None,
                 online_lookback_days=63, apply_daily_weights=True, diagnose_cov=False, record_sjm_loss_curve=False):
        """target_tracking_error: if set (e.g. 0.02 for 2%), tune BL view confidence to hit TE (paper: 1-4%).
        online_lookback_days: window for Nystrup online inference (default 63).
        apply_daily_weights: if True (paper default), apply BL weights every day at T+2; else use rebalance_threshold.
        diagnose_cov: if True, log covariance matrix diagnostics in optimize_portfolio.
        record_sjm_loss_curve: if True, each SJM stores loss_curve_ (list of objective per iteration) for plotting."""
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
        self.target_tracking_error = target_tracking_error  # e.g. 0.02 for 2% TE
        self.online_lookback_days = online_lookback_days
        self.apply_daily_weights = apply_daily_weights
        self.diagnose_cov = diagnose_cov
        self.record_sjm_loss_curve = record_sjm_loss_curve
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
        sjm_config: optional dict of factor -> {jump_penalty, sparsity_param, jump_penalty_matrix}
        for per-factor overrides. If jump_penalty_matrix is set (n_regimes×n_regimes), it defines
        asymmetric transition costs Λ[i,j] (JOH-12); optional jump_penalty then overrides the
        stored scalar attribute only.
        """
        logger.info("Fitting regime models on active returns (6 factors)...")
        self.regime_means = {}
        market_ret = market_returns.reindex(active_returns.index).ffill().bfill()
        
        for factor in active_returns.columns:
            logger.info("Fitting regime model for {}".format(factor))
            cfg = (sjm_config or {}).get(factor, {})
            jmp_m = cfg.get('jump_penalty_matrix')
            model_kw = {
                'sparsity_param': cfg.get('sparsity_param', DEFAULT_SPARSITY_PARAM),
                'record_loss_curve': getattr(self, 'record_sjm_loss_curve', False),
            }
            if jmp_m is not None:
                model_kw['jump_penalty_matrix'] = np.asarray(jmp_m, dtype=float)
                if cfg.get('jump_penalty') is not None:
                    model_kw['jump_penalty'] = float(cfg['jump_penalty'])
            else:
                model_kw['jump_penalty'] = cfg.get('jump_penalty', DEFAULT_JUMP_PENALTY)
            model = SparseJumpModel(**model_kw)
            try:
                on_cb = (lambda f: lambda i, o: self.on_sjm_iteration(f, i, o))(factor) \
                    if getattr(self, 'on_sjm_iteration', None) else None
                if self.market_data is not None and not self.market_data.empty:
                    model.fit(
                        active_returns[factor],
                        market_data=self.market_data,
                        market_returns=market_ret,
                        factor_name=factor,
                        on_iteration_callback=on_cb
                    )
                else:
                    model.fit(active_returns[factor], on_iteration_callback=on_cb)
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
    
    def generate_expected_returns(self, current_regimes, regime_means=None):
        """
        Expected active returns from regime historical averages (Phase 4).
        regime_means: optional override (e.g. from walk-forward) {factor: {regime: mean_ann}}.
        Returns dict {factor: expected_active_return} for the 6 factors.
        """
        means_src = regime_means if regime_means is not None else self.regime_means
        expected = {}
        for factor in self.regime_models:
            regime = int(current_regimes.get(factor, 0))
            means = means_src.get(factor, {0: 0.0})
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
    
    @staticmethod
    def check_covariance_matrix(cov_matrix, label="cov"):
        """
        Run diagnostics on a covariance matrix. Returns a dict and logs warnings.
        Use for training/backtest sanity checks.
        """
        if isinstance(cov_matrix, pd.DataFrame):
            Sigma = cov_matrix.values
            names = list(cov_matrix.index)
        else:
            Sigma = np.asarray(cov_matrix)
            names = list(range(Sigma.shape[0]))
        n = Sigma.shape[0]
        diag = np.diag(Sigma)
        diag_nonpos = (diag <= 0).sum()
        try:
            evals = np.linalg.eigvalsh(Sigma)
            evals_min, evals_max = float(evals.min()), float(evals.max())
            cond = evals_max / (np.abs(evals_min) + 1e-20)
            is_psd = evals_min >= -1e-10
        except Exception as e:
            evals_min = evals_max = cond = None
            is_psd = False
            logger.warning("Covariance eigenvalue check failed: %s", e)
        out = {
            "shape": (n, n),
            "diag_min": float(np.min(diag)),
            "diag_max": float(np.max(diag)),
            "diag_nonpositive_count": int(diag_nonpos),
            "eigenvalue_min": evals_min,
            "eigenvalue_max": evals_max,
            "condition_number": cond,
            "is_psd": is_psd,
        }
        if diag_nonpos > 0:
            logger.warning("[%s] Covariance has %d non-positive diagonal entries", label, diag_nonpos)
        if not is_psd and evals_min is not None:
            logger.warning("[%s] Covariance is not PSD: min eigenvalue = %.2e", label, evals_min)
        if cond is not None and cond > 1e10:
            logger.warning("[%s] Covariance condition number very large: %.2e", label, cond)
        return out
    
    def optimize_portfolio(self, returns, expected_active_returns):
        """Optimize portfolio weights using EWMA covariance (halflife=126).
        If target_tracking_error is set, tunes view confidence to hit that TE (paper: 1-4%)."""
        cov_matrix = self._ewma_covariance(returns)
        if self.diagnose_cov:
            diag = self.check_covariance_matrix(cov_matrix, label="BL")
            logger.info("Covariance diagnostics: %s", diag)
        if self.target_tracking_error is not None and self.target_tracking_error > 0:
            weights = self.optimizer.optimize_with_te_target(
                expected_active_returns, cov_matrix,
                target_te_annual=self.target_tracking_error,
                te_tolerance=0.005,
            )
        else:
            weights = self.optimizer.optimize(expected_active_returns, cov_matrix)
        return weights
    
    def should_rebalance(self, new_weights):
        """Determine if portfolio should be rebalanced. Paper: apply daily at T+2 (apply_daily_weights=True)."""
        if self.current_weights is None:
            return True
        if self.apply_daily_weights:
            return True
        weight_changes = abs(new_weights - self.current_weights)
        return weight_changes.max() > self.rebalance_threshold
    
    def _build_regime_getter(self, active_returns, market_returns, returns, start_date, end_date,
                             sjm_config, use_walk_forward, use_online_inference=True):
        """
        Build (i, current_date) -> (current_regimes, regime_means).
        Walk-forward: refit monthly. If use_online_inference, run Nystrup lookback over window
        with fixed centroids; else use last in-sample regime from fit.
        """
        if not use_walk_forward:
            self.fit_regime_models(active_returns, market_returns, sjm_config=sjm_config)

            def getter(i, _):
                regime_idx = max(0, i - 2)
                out = {}
                for factor in self.regime_models:
                    rs = self.regime_models[factor].regimes_
                    out[factor] = int(rs.iloc[regime_idx]) if regime_idx < len(rs) else 0
                return out, self.regime_means
            return getter

        first_me = pd.Timestamp(start_date) - pd.offsets.MonthEnd(1)
        month_ends = pd.date_range(start=first_me, end=end_date, freq='ME')
        fits = {}
        prev_me = pd.Timestamp(start_date) - pd.offsets.MonthEnd(1)
        returns_index = returns.index
        lb = self.online_lookback_days
        _feat_model = SparseJumpModel()

        for me in month_ends:
            train_end = prev_me
            train_ar = active_returns.loc[:train_end]
            train_mr = market_returns.loc[:train_end]
            if len(train_ar) < 252:
                prev_me = me
                continue
            self.fit_regime_models(train_ar, train_mr, sjm_config=sjm_config)
            fits[prev_me] = {
                "models": dict(self.regime_models),
                "regime_means": {f: dict(m) for f, m in self.regime_means.items()},
            }
            prev_me = me

        me_list = sorted(fits.keys())
        market_data = self.market_data

        def getter(i, current_date):
            regime_date = returns_index[max(0, i - 2)]
            use_me = None
            for m in reversed(me_list):
                if m <= regime_date:
                    use_me = m
                    break
            if use_me is None or not me_list:
                factors = list(self.regime_models.keys()) if self.regime_models else []
                return {f: 0 for f in factors}, self.regime_means

            cached = fits[use_me]
            models = cached["models"]
            regime_means = cached["regime_means"]

            if not use_online_inference:
                regimes_at = {}
                for factor in models:
                    rs = models[factor].regimes_
                    regimes_at[factor] = int(rs.iloc[-1]) if rs is not None and len(rs) > 0 else 0
                return regimes_at, regime_means

            out = {}
            for factor in models:
                model = models[factor]
                if model.centroids_ is None or model.feature_weights_ is None:
                    out[factor] = 0
                    continue
                ar_slice = active_returns[factor].loc[:regime_date]
                if len(ar_slice) < max(21, lb):
                    out[factor] = int(model.regimes_.iloc[-1]) if model.regimes_ is not None and len(model.regimes_) > 0 else 0
                    continue
                try:
                    if market_data is not None and not market_data.empty:
                        feats = _feat_model._calculate_features(
                            ar_slice, market_data, factor,
                            market_returns.reindex(ar_slice.index).ffill().bfill()
                        )
                    else:
                        feats = _feat_model._calculate_features_legacy(ar_slice)
                    feats = np.asarray(feats)
                    valid = ~np.isnan(feats).any(axis=1)
                    feats_clean = feats[valid]
                    if len(feats_clean) < 21:
                        out[factor] = 0
                        continue
                    window = feats_clean[-lb:]
                    X_scaled = model.scaler.transform(window)
                    out[factor] = model.infer_regime_online(X_scaled)
                except Exception as e:
                    logger.debug("Online inference failed for %s: %s", factor, e)
                    out[factor] = int(model.regimes_.iloc[-1]) if model.regimes_ is not None and len(model.regimes_) > 0 else 0
            return out, regime_means

        return getter

    def validate_regime_inference(self, current_regimes, expected_active=None):
        """
        Validate regime inference before rebalance (JOH-9). Returns (ok, message).
        Checks: regimes in valid range, expected returns within ±5% p.a., no NaN.
        """
        if current_regimes is None or not current_regimes:
            return False, "No regimes"
        n_regimes = 2  # SJM default
        for factor, r in current_regimes.items():
            r_int = int(r)
            if r_int < 0 or r_int >= n_regimes:
                return False, "{} regime {} out of range [0,{})".format(factor, r_int, n_regimes)
        if expected_active is not None:
            for factor, er in expected_active.items():
                if np.isnan(er) or np.isinf(er):
                    return False, "{} expected return is NaN/Inf".format(factor)
                ann = er * 252.0
                if abs(ann) > EXPECTED_RETURN_CAP + 1e-6:
                    return False, "{} expected return {:.2%} exceeds ±5% cap".format(factor, ann)
        return True, "OK"

    def backtest(self, start_date, end_date, sjm_config=None, use_walk_forward=True):
        """Run backtest with paper alignment: active returns, 1-day delay, 5 bps tx costs.
        sjm_config: optional dict of factor -> {jump_penalty, sparsity_param} for tuned params.
        use_walk_forward: if True (default), refit SJMs monthly per paper; else fit once on full data."""
        logger.info("Starting backtest from {} to {} (walk_forward={})".format(start_date, end_date, use_walk_forward))

        fetch_start = start_date
        if use_walk_forward:
            try:
                from datetime import datetime, timedelta
                d = pd.Timestamp(start_date) - pd.Timedelta(days=504)
                fetch_start = d.strftime("%Y-%m-%d")
            except Exception:
                pass
        data = self.fetch_data(fetch_start, end_date)
        returns = self.calculate_returns()
        active_returns = compute_active_returns(returns, market_col=MARKET_ETF)
        market_returns = returns[MARKET_ETF]

        regime_getter = self._build_regime_getter(
            active_returns, market_returns, returns, start_date, end_date,
            sjm_config, use_walk_forward
        )

        portfolio_values = []
        weights_history = []
        rebalance_dates = []
        initial_value = 100000
        current_value = initial_value
        txn_cost_bps = TXN_COST_BPS / 10000

        start_idx = max(self.lookback_days, 252)
        backtest_start_ts = pd.Timestamp(start_date)
        returns_index = returns.index

        for i in range(start_idx, len(returns)):
            current_date = returns_index[i]
            recent_returns = returns.iloc[max(0, i - self.lookback_days):i]

            if len(recent_returns) < 252:
                portfolio_values.append(current_value)
                continue

            if current_date < backtest_start_ts:
                portfolio_values.append(current_value)
                continue

            regime_out = regime_getter(i, current_date)
            current_regimes, regime_means = regime_out if isinstance(regime_out, tuple) else (regime_out, None)
            if current_regimes is None:
                portfolio_values.append(current_value)
                continue

            expected_active = self.generate_expected_returns(current_regimes, regime_means=regime_means)
            ok, msg = self.validate_regime_inference(current_regimes, expected_active)
            if not ok:
                logger.warning("Regime validation failed on {}: {}; skipping rebalance".format(current_date, msg))
                new_weights = self.current_weights if self.current_weights is not None else pd.Series(1.0 / len(self.factor_etfs), index=self.factor_etfs.keys())
            else:
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
    sjm_config = None
    try:
        from sjm_params_store import load
        sjm_config = load(source="local", version_or_alias="production")
        if not sjm_config:
            sjm_config = load(source="local", version_or_alias="latest")
    except Exception:
        pass

    # Run backtest through latest available data
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    results = strategy.backtest('2024-01-01', end_date, sjm_config=sjm_config)
    
    print("\n=== Helix 1.1 Strategy Results ===")
    print("Total Return: {:.2%}".format(results['total_return']))
    print("Sharpe Ratio: {:.2f}".format(results['sharpe_ratio']))
    print("Max Drawdown: {:.2%}".format(results['max_drawdown']))
    print("Rebalances: {}".format(results['n_rebalances']))
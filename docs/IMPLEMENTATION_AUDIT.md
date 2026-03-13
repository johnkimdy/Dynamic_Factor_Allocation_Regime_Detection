# Implementation Audit: Helix vs Princeton Paper & Author's Reference Code

**Paper:** arXiv:2410.14841 — *Dynamic Factor Allocation Leveraging Regime-Switching Signals*  
**Author's SJM:** [jump-models/sparse_jump.py](https://github.com/Yizhan-Oliver-Shu/jump-models/blob/master/jumpmodels/sparse_jump.py)  
**Observed:** Paper reports IR 0.4–0.5 vs EW, Sharpe ~0.65, strong outperformance; our replication tracks SPY closely and underperforms.

---

## Executive Summary

Several **critical** implementation differences likely explain the performance gap. The most severe are:

1. **SJM algorithm** — Our custom implementation differs from the author's reference (BCSS+Lasso vs variance-reduction; no jump-penalty scaling).
2. **Online regime inference** — We use "last regime from last monthly refit"; the paper uses Nystrup et al. 2020a lookback algorithm.
3. **Data** — Paper uses Bloomberg + MSCI USA (PBUS); we use yfinance + SPY.
4. **Rebalance logic** — Paper applies BL weights at T+2 daily; we use a 2% threshold.

---

## 1. Sparse Jump Model (SJM) — Critical

### 1.1 Author's Implementation ([sparse_jump.py](https://github.com/Yizhan-Oliver-Shu/jump-models/blob/master/jumpmodels/sparse_jump.py))

| Component | Author's Code | Our Helix |
|-----------|---------------|-----------|
| Core JM | Full JumpModel (DP, E-step, coordinate descent) | Custom alternating optimization |
| Feature weights | BCSS (Between Cluster Sum of Squares) + `solve_lasso` | Variance reduction (total_var − within_var) |
| Sparsity | L1-norm constraint: `norm_ub = sqrt(max_feats)` = κ | Top-κ features: `n_keep = ceil(sqrt(κ²))` |
| Jump penalty | **Scaled by 1/√n_features** | Not scaled |
| Weight in distance | `feat_weights = sqrt(w)`; distance = ‖√w ⊙ (x − θ)‖² | `weights * (X - centroids)²` |
| Input | Multi-feature matrix X | Same |
| Output | `labels_`, `proba_`, `ret_`, `vol_` per state | `regimes_`, `centroids_` |

**Implications:**
- BCSS measures clustering value per feature; Lasso selects features. Our variance-reduction is a different criterion.
- Jump penalty scaling makes the penalty scale-invariant with feature count.
- Our sparsity keeps top N features; the author uses Lasso soft-thresholding (continuous weights, some → 0).

### 1.2 Paper Objective (Eq. 1)

```
min  Σ ½‖x_t − θ_{s_t}‖² + λ Σ I(s_{t-1}≠s_t)
```

SJM adds: L1 constraint on weight vector; distances use weighted features. Author uses `feat_weights = sqrt(w)` so the effective distance is weighted.

---

## 2. Online Regime Inference — Critical

### 2.1 Paper (Section 3.1.3)

> "We employ the **online inference algorithm outlined in Nystrup et al. 2020a**, which efficiently incorporates historical data. The core idea is to **solve the optimization problem (1) over the state sequence within a lookback window**, while keeping the centroids fixed at their previously estimated values, and then **extract the last optimal state** as the online inference."

> "Regime s_T inferred at end of day T is applied on day T+2 (one-day delay)."

### 2.2 Our Implementation

```python
# _build_regime_getter (walk-forward): we fit monthly, then for each day we use
# regimes_at_prev = last regime from the fit that ended at prev month-end
# So we use the SAME regime for the entire month until next refit.
```

We use the **last in-sample regime** from the most recent monthly refit, not a proper online inference over a lookback window. Regimes are effectively constant for a full month, whereas the paper updates the regime daily using the lookback algorithm.

---

## 3. Black–Litterman — Verified

### 3.1 Paper Appendix B

- **Formula:** w^BL = w^bmk + P^T λ  
- **λ:** λ = δ⁻¹ (PΣP^T + Ω/τ)⁻¹ (v − Pπ)

Our implementation matches this. Checks:
- π = δ Σ w_bmk
- P: long factor, short market
- Ω = c × diag(PΣP^T)
- τ = 0.025

### 3.2 Benchmark Weights

- Paper: EW among 7 indices, **rebalanced quarterly**
- We use fixed w_bmk = 1/7. Between quarter-ends the true EW drifts; impact on prior is likely small.

### 3.3 Covariance & Risk Aversion

- Paper: EWMA halflife 126 days, δ = 2.5
- We use `_ewma_covariance` and `RISK_AVERSION = 2.5` — aligned.

---

## 4. Rebalancing

### 4.1 Paper

- Apply regime/BL weights at T+2
- Implied: new weights every day when regime/views change
- Exhibit 4: TE=2% → ~395% annual turnover

### 4.2 Our Implementation

- `should_rebalance`: rebalance only if max weight change > 2%
- Result: 280 rebalances over ~1.5 years (2024–2025) — roughly every 2 days
- We may rebalance slightly less often than the paper, but both are high frequency.

---

## 5. Data

| Aspect | Paper | Helix |
|--------|-------|-------|
| Source | Bloomberg | yfinance |
| Market | MSCI USA (PBUS) | SPY (S&P 500) |
| Period | 1993 – mid-2024 | Depends on fetch |
| Factor indices | May use slightly different index vintages (see Appendix A) | VLUE, SIZE, MTUM, QUAL, USMV, IWF |

SPY vs MSCI USA and different vendors can introduce basis and timing differences.

---

## 6. Features

### 6.1 VIX (Exhibit 3)

- Paper: — transformations in sequence
- Interpretation: EWMA of diff(log(VIX)) with window 21

Our code has:
- `vix_log`
- `vix_diff`
- `vix_ewm`

These may not match the exact sequence (log → diff → EWMA). Need to confirm.

### 6.2 Other Features

- Factor-specific: EWMA, RSI, %K, MACD, DD, β — structure matches
- Market-environment: market EWMA, VIX, 2Y, 10Y−2Y — present

---

## 7. Hyperparameter Tuning

### 7.1 Paper (Section 3.2)

- Expanding window 8–12 years
- Refit monthly
- Validation: 6-year rolling, step 6 months
- Single-factor long-short Sharpe as objective
- Per-factor tuning of λ and κ

### 7.2 Our `tune_sjm_hyperparameters.py`

- Uses Optuna with temporal splits
- Validation period and rolling logic may differ
- Same idea, but exact procedure may not match

---

## 8. Prioritized Fixes

### P0 (Likely largest impact) — ✅ IMPLEMENTED

1. **Use the author's jump-models SJM** — DONE
   - Integrate or port [jump-models](https://github.com/Yizhan-Oliver-Shu/jump-models)
   - Ensures BCSS, Lasso, jump-penalty scaling, and JM E-step match the paper

2. **Implement Nystrup online inference** — DONE
   - Lookback window over recent history
   - Solve JM with fixed centroids; take last state as regime
   - Apply regime at T+2 as in the paper

### P1 — ✅ IMPLEMENTED

3. **Rebalance behavior** — DONE
   - Option A: apply BL weights daily at T+2 (no 2% filter)
   - Option B: keep 2% filter but verify it does not materially reduce turnover vs paper

4. **VIX feature** — DONE
   - EWMA of diff(log(VIX)) with window 21 (single feature per Exhibit 3)

### P2

5. **Data**
   - Consider MSCI USA / PBUS or align data definitions if possible
   - Document index/fund mappings and date ranges

6. **Hyperparameter tuning**
   - Mirror paper: 8–12y expanding window, monthly refit, 6y rolling validation, per-factor λ and κ

---

## 9. Reference: Author's SJM Fit Loop

From [sparse_jump.py](https://github.com/Yizhan-Oliver-Shu/jump-models/blob/master/jumpmodels/sparse_jump.py):

```python
norm_ub = np.sqrt(self.max_feats)  # κ
jump_penalty = self.jump_penalty / np.sqrt(self.n_features_all)  # scaled!
feat_weights = np.sqrt(w)
jm.fit(X, ret_ser=ret_ser, feat_weights=feat_weights, sort_by=sort_by)
BCSS = compute_BCSS(X_arr, jm.proba_, centers_unweighted)
w = solve_lasso(BCSS/BCSS.max(), norm_ub)
```

---

## 10. Conclusion

The performance gap is most plausibly driven by:

1. **Different SJM** — Our custom SJM diverges from the paper’s (BCSS+Lasso, jump scaling, weighting).
2. **No proper online inference** — We use a monthly regime snapshot instead of the Nystrup lookback algorithm.
3. **Data and auxiliary choices** — SPY vs MSCI USA, feature construction, tuning setup.

Highest leverage is to integrate the author’s jump-models and implement the Nystrup online inference, then re-run backtests.

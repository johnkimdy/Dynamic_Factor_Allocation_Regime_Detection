# Dynamic Factor Allocation: Implementation Audit vs. Princeton Paper

**Paper:** *Dynamic Factor Allocation Leveraging Regime-Switching Signals*  
**Authors:** Yizhan Shu, John M. Mulvey (Princeton ORFE)  
**Reference:** arXiv:2410.14841v1, October 2024

**Implementation:** Helix 1.1 (`helix_factor_strategy.py`)

**Linear Issue:** [JOH-6](https://linear.app/johnkimdy/issue/JOH-6/dynamic-factor-allocation-audit-implementation-against-princeton-paper)

---

## Executive Summary

The Helix 1.1 implementation is inspired by the Princeton paper but deviates significantly in several critical areas. The paper's core thesis—**regime analysis on factor active returns** (factor vs. market)—is not implemented. The current code uses absolute returns for regime identification and has a simplified feature set, no hyperparameter tuning, and different expected-return logic. Addressing the critical deviations below would bring the implementation in line with the paper's methodology.

---

## 1. Asset Universe

| Aspect | Princeton Paper | Helix Implementation | Status |
|--------|-----------------|----------------------|--------|
| Market index | MSCI USA / PBUS | SPY | ✅ Acceptable (SPY tracks S&P 500, similar) |
| Factor ETFs | VLUE, SIZE, MTUM, QUAL, USMV, IWF | Same 6 factors | ✅ Match |
| Total assets | 7 (market + 6 factors) | 7 | ✅ Match |

---

## 2. Regime Identification Input (CRITICAL)

| Aspect | Princeton Paper | Helix Implementation | Status |
|--------|-----------------|----------------------|--------|
| **Regime input** | **Factor active returns** (factor return − market return) | **Absolute returns** per ETF | ❌ **CRITICAL** |
| Scope | 6 SJMs, one per **factor** (active return vs. market) | 7 SJMs, one per **ETF** (absolute return) | ❌ **CRITICAL** |

**Paper (Section 2):**  
> "Despite the long-only index construction, we assess factor performance on a **relative basis** by computing **factor active returns**, defined as the factor index return minus the market index return."

**Why it matters:** Active returns remove the common market component and reveal factor-specific cycles. Using absolute returns mixes market and factor dynamics and makes regimes less interpretable.

**Required change:**  
- Compute active returns: `active_return[factor] = return[factor] - return[SPY]` for each of the 6 factors.  
- Fit **6** SJMs (one per factor) on each factor’s active return series, **not** 7 models on absolute returns.

---

## 3. SJM Feature Set (CRITICAL)

| Aspect | Princeton Paper | Helix Implementation | Status |
|--------|-----------------|----------------------|--------|
| Feature count | ~20 features (Exhibit 3) | 5 features | ❌ **CRITICAL** |
| Factor-specific | EWMA (8,21,63), RSI (8,21,63), %K (8,21,63), MACD (8,21) & (21,63), DD (log,21), β (21) | EWMA, RSI, momentum, volatility | ❌ Incomplete |
| Market-environment | Market return EWMA, VIX (log,diff,EWMA), 2Y yield diff, 10Y-2Y yield diff | None | ❌ **Missing** |
| Window lengths | 8, 21, 63 trading days | Single 20-day window | ❌ |
| Data source | FRED for VIX, Treasury yields | yfinance only | ❌ |

**Paper (Section 3.1.1, Exhibit 3):**  
Factor-specific features: EWMA active return, RSI, %K, MACD at multiple windows; downside deviation; active market beta.  
Market-environment features: market return EWMA, VIX (log, diff, EWMA), 2Y yield (diff, EWMA), 10Y−2Y slope (diff, EWMA).

**Required change:**  
- Add all factor-specific features with windows 8, 21, 63.  
- Fetch VIX and Treasury data (e.g., from FRED or equivalent) and compute market-environment features.  
- Use this expanded feature set for SJM inputs.

---

## 4. SJM Hyperparameters

| Aspect | Princeton Paper | Helix Implementation | Status |
|--------|-----------------|----------------------|--------|
| Jump penalty λ | Tuned via CV; example λ=50 | Default 0.1 | ❌ |
| Sparsity κ | κ²≈9.5 (L1 constraint) | 1.0 | ❌ |
| Tuning method | Single-factor long-short strategy, time-series CV | None | ❌ |
| Training window | Expanding 8–12 years | 252 days (1 year) | ❌ |
| Refit frequency | Monthly | Once at backtest start | ❌ |

**Paper (Section 3.2):**  
> "We tune the hyperparameters for each factor individually... Our expanding training window spans a minimum of 8 years and a maximum of 12 years. For each set of hyperparameters, we refit the SJM monthly."

**Required change:**  
- Implement single-factor long-short strategy per factor.  
- Use time-series cross-validation with expanding window (8–12 years).  
- Tune λ and κ per factor; refit SJMs monthly.

---

## 5. Online Regime Inference

| Aspect | Princeton Paper | Helix Implementation | Status |
|--------|-----------------|----------------------|--------|
| Method | Nystrup et al. (2020a) lookback algorithm | Last in-sample regime | ❌ |
| One-day delay | Regime at T applied at T+2 | No explicit delay | ❌ |
| Persistence | Uses historical context in window | No | ❌ |

**Paper (Section 3.1.3):**  
> "We assume that this inferred regime s_T can only be applied on day T+2 with a one-day delay – meaning that we estimate s_T at the end of day T and rebalance the portfolio accordingly at the end of the next day, T+1."

**Required change:**  
- Implement online inference with lookback window (or equivalent).  
- Apply regime inferred at end of day T only from day T+2 onward (one-day delay).

---

## 6. Expected Returns from Regimes

| Aspect | Princeton Paper | Helix Implementation | Status |
|--------|-----------------|----------------------|--------|
| Calculation | Historical average **active return** across training periods in same regime | Ad hoc: positive returns if regime 1, else scaled mean | ❌ |
| Input to BL | Expected **active returns** for view portfolios | Effectively absolute expected returns | ❌ |
| Cap | ±5% per annum for long-short strategy | No explicit cap | ⚠️ |

**Paper (Section 3.2):**  
> "We first associate the inferred regimes for a specific factor with the **historical average active return across all training periods under the same regime**. This historical average serves as the expected active return for the factor."

**Required change:**  
- Compute per-factor, per-regime historical average active return on training data.  
- Use inferred regime to select expected active return.  
- Pass these expected **active** returns to the Black–Litterman view portfolios (long factor, short market).

---

## 7. Black–Litterman Model

| Aspect | Princeton Paper | Helix Implementation | Status |
|--------|-----------------|----------------------|--------|
| Benchmark | EW among 7 indices, **quarterly rebalancing** | EW, but used implicitly in optimization | ⚠️ |
| View portfolios | 6 relative (long factor, short market) | Same structure | ✅ |
| Prior (δ) | 2.5 | 3.0 | ⚠️ Minor |
| Covariance | EWMA, halflife 126 days | Sample covariance | ❌ |
| View confidence | Tuned for target TE (1–4%) | Fixed `confidence_multiplier` | ❌ |
| τ | Used for prior/views uncertainty | 0.025 | ✅ Reasonable |

**Paper (Section 4):**  
> "We use an equally weighted (EW) allocation among the seven indices, **rebalanced quarterly**, as the benchmark... we estimate the covariance matrix using an **exponentially weighted moving approach with a halflife of 126 days**."

**Required changes:**  
- Use EWMA covariance with halflife 126 days.  
- Add tracking-error targeting (1–4%) and tune view confidence accordingly.

---

## 8. Rebalancing and Transaction Costs

| Aspect | Princeton Paper | Helix Implementation | Status |
|--------|-----------------|----------------------|--------|
| Rebalance trigger | Portfolio weights from BL; apply at T+2 | 2% threshold on weight change | ⚠️ Different |
| Transaction costs | 5 bps both sides | Not modeled | ❌ |
| Benchmark rebalancing | Quarterly | N/A | — |

**Required change:**  
- Model 5 bps round-trip transaction costs in performance evaluation.

---

## 9. Single-Factor Long-Short Strategy (Validation)

| Aspect | Princeton Paper | Helix Implementation | Status |
|--------|-----------------|----------------------|--------|
| Purpose | Validate regimes, tune hyperparameters | Not implemented | ❌ |
| Position sizing | Linear in expected return, cap ±5% p.a. | N/A | — |
| Success criteria | Positive Sharpe per factor, low cross-factor correlation | N/A | — |

**Required change:**  
- Implement per-factor long-short strategy as in paper.  
- Use it for SJM hyperparameter tuning and regime validation.

---

## 10. Implementation Checklist

### Critical (must fix for alignment)

- [x] **Active returns:** Use factor active returns (factor − market) for regime analysis, not absolute returns.
- [x] **6 factor SJMs:** Fit SJMs only for the 6 factors on their active returns; drop SJM for the market.
- [x] **Feature set:** Implement full ~20-feature set (factor-specific + market-environment) with correct windows.
- [x] **Market data:** Add VIX and Treasury yields (e.g., FRED) for market-environment features.
- [x] **Expected returns:** Use historical average active return per regime; pass expected **active** returns to BL.

### Important (should fix)

- [x] **SJM hyperparameters:** Tune λ and κ per factor via single-factor long-short CV (`tune_sjm_hyperparameters.py`).
- [x] **Online inference:** One-day delay (regime at T applied from T+2); lookback algorithm not yet implemented.
- [x] **Covariance:** EWMA with halflife 126 days instead of sample covariance.
- [x] **Tracking error:** Target 1–4% TE and set view confidence accordingly (`--target-te`, `target_tracking_error`).
- [x] **Transaction costs:** Apply 5 bps in performance evaluation.

### Nice to have

- [x] Benchmark: explicit EW portfolio rebalanced quarterly for comparison (`compute_ew7_benchmark`).
- [x] Training window: expanding 8–12 years, monthly refit (min_train_days in `paper_aligned.json`).

---

## References

- Shu, Y., Mulvey, J. M. (2024). *Dynamic Factor Allocation Leveraging Regime-Switching Signals*. arXiv:2410.14841v1.
- Nystrup et al. (2020a). Greedy online classification of persistent market states. *Journal of Financial Data Science*.
- Nystrup et al. (2021). Feature selection in jump models. *Expert Systems with Applications* 184:115558.

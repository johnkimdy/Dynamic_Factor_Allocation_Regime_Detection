# Temporal Split for SJM Hyperparameter Tuning

## Recommended Split

| Split | Period | Role |
|-------|--------|------|
| **Train** | 1993-01-01 – 2011-12-31 | Expanding window: first validation month (Jan 2012) trains on data through Dec 2011 |
| **Validation** (tuning OOS) | 2012-01-01 – 2022-12-31 | Optuna maximizes mean long-short Sharpe over this period; hyperparameters selected here |
| **Holdout** (true OOS) | 2023-04-01 – 2025-12-31 | 3‑month embargo; never seen during tuning; unbiased performance estimate |

## Rationale

- **Train ~19 years**: Enough history for regime estimation and feature stability.
- **Validation ~11 years**: Covers 2012–2022 (post‑GFC, COVID, rate hikes); diverse regimes.
- **Holdout ~3 years**: Short but still useful; true OOS for reporting and deployment.

## Caveats

### 1. Short Holdout

- Only ~3 years of holdout data.
- Higher variance in Sharpe and drawdown estimates.
- More sensitive to single events (e.g., 2023–2024 performance).
- **Mitigation**: Report both validation and holdout metrics; be conservative when interpreting holdout.

### 2. Possible Regime Shift

- Validation ends 2022; holdout starts 2023.
- Recent years may differ in rate regime, volatility, factor behavior.
- Hyperparameters tuned on 2012–2022 may not be optimal for 2023+.
- **Mitigation**: Monitor holdout vs validation divergence; consider periodic retuning.

### 3. Embargo

- Default: 3‑month embargo between validation and holdout (config: `embargo_months`).
- With `embargo_months=3`: validation ends 2022-12-31 → holdout starts 2023-04-01.
- Reduces autocorrelation across the cutoff.

### 4. ETF Inception Bias

- Some factor ETFs have shorter histories.
- 1993 start is driven by SPY; factor ETFs may start later, shortening effective train window for those factors.

### 5. Data Snooping / Multiple Testing

- We tune 6 factors independently.
- Combined “best” run is chosen by mean validation Sharpe across factors; some selection bias remains. **Per-factor best**: Each factor's best (λ, κ²) is tracked separately.
- Holdout is the main guard against overfitting.

## Lock Date Workflow

**Example:** Today is March 1, 2026; lock date is February 15, 2026.

1. **Tune & train** on all data before Feb 15, 2026 (e.g., validation 2012–2025).
2. **Validate** performance from Feb 16 to Mar 1 (post‑lock period).
3. **Deploy** if validation looks acceptable.

Lock date ensures:

- No future information is used after the cutoff.
- Post‑lock performance is true OOS.
- Regime inference and allocations are implementable at lock time.

To apply a lock date: set `validation_end` (and `data_end` if needed) to the lock date in the tune config, and evaluate post‑lock performance separately.

## Walk-Forward in analyze_strategy

Walk‑forward belongs in **analyze_strategy** (where the SJM is trained for backtest), not in hyperparameter tuning.

- **Tuning**: Fixed validation window (e.g., 2012–2022); choose (λ, κ²) once.
- **Analyze / backtest**: For each backtest date, train SJM on expanding history and infer regime. Refit monthly; that is the walk‑forward.

**Implemented**: `helix_factor_strategy.backtest(use_walk_forward=True)` refits SJMs monthly. `tune_sjm_hyperparameters` uses a fixed validation split.

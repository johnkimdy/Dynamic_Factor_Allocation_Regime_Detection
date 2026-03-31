# Helix 1.1: Factor-Based Portfolio Optimization Strategy

A lightweight daily portfolio rebalancing system using Sparse Jump Models (SJM) for regime identification across factor ETFs, inspired by the Princeton paper ["Portfolio Allocation Using Sparse Jump Model"](https://arxiv.org/html/2410.14841v1) (arXiv:2410.14841).

**[Live dashboard → justholdspy.vercel.app](https://justholdspy.vercel.app)**

## Overview

Helix 1.1 uses regime-aware factor timing to allocate across six U.S. equity factor ETFs:

- **Daily portfolio optimization** via Black-Litterman with regime-adjusted views
- **Sparse Jump Models** for interpretable binary regime classification per factor
- **Long-only constraints** for practical implementation
- **Quarterly-rebalanced EW(7) benchmark** (paper-aligned: SPY + 6 factor ETFs)

## Architecture

### System Design

- **Regime layer (SJM)**: One Sparse Jump Model per factor — classifies each factor into calm/stress regime daily
- **Allocation layer (Black-Litterman)**: Combines regime signals into portfolio weights with tracking-error targeting
- **Benchmark**: EW(7) — equal-weight of SPY, QUAL, MTUM, USMV, VLUE, SIZE, IWF, quarterly rebalanced (per paper Exhibit 1)

### Factor Universe

| Factor | ETF | Description |
|--------|-----|-------------|
| Market | SPY | SPDR S&P 500 ETF Trust |
| Quality | QUAL | iShares MSCI USA Quality Factor ETF |
| Momentum | MTUM | iShares MSCI USA Momentum Factor ETF |
| Low Volatility | USMV | iShares MSCI USA Min Vol Factor ETF |
| Value | VLUE | iShares MSCI USA Value Factor ETF |
| Size | SIZE | iShares MSCI USA Size Factor ETF |
| Growth | IWF | iShares Russell 1000 Growth ETF (benchmark only) |

## Methodology

### Regime Identification (SJM)
- One SJM per factor ETF; binary regime (calm / stress)
- Features: EWMA returns, RSI-like momentum, price momentum, volatility
- Jump penalty λ controls switching frequency; tuned per factor via Optuna on holdout Sharpe
- **Asymmetric extension (JOH-12)**: Replace scalar λ with a penalty matrix Λ where λ_enter ≠ λ_exit, capturing the empirical asymmetry that crises arrive fast but recoveries are slow (see below)

### Portfolio Construction
- **Black-Litterman** with equal-weight benchmark; views derived from per-factor regime expected active return
- Tracking-error targeting: Ω (view confidence) tuned to hit 1–4% TE (paper range)
- Long-only, fully-invested; 5 bps brokerage cost per unit turnover modeled

### Benchmarks and Fees
- **EW(7)** (paper benchmark): SPY + 6 factor ETFs, equal-weight, quarterly rebalanced
- **SPY**: buy-and-hold
- Net-of-fees analysis: ETF ER drag (blended ~13 bps/yr for 7-asset basket, 3 bps for SPY) + brokerage at each rebalance

## Performance

Multi-period backtest results (gross of fees, annualized). Periods ending 2024–2026.

| Period | Helix Return | EW(7) Return | SPY Return | Helix Sharpe | EW(7) Sharpe | Helix Sortino | Max DD |
|--------|-------------|-------------|------------|--------------|--------------|---------------|--------|
| 2024-now | — | — | — | — | — | — | — |
| 2022-2024 | — | — | — | — | — | — | — |
| 2007-2024 | — | — | — | — | — | — | — |

*Run `python analyze_strategy.py --export` to populate. Live results render in the dashboard.*

### Key Metrics Tracked
- **Total & annualized return** vs EW(7) and SPY
- **Sharpe ratio** (annualized, 0 risk-free rate)
- **Sortino ratio** (downside deviation denominator, 0 target return) — added in JOH-12
- **Annualized volatility** and **max drawdown**
- **Active return and information ratio** vs both benchmarks
- **Net-of-fees** equivalents for all three strategies

## JOH-10: Benchmark Alignment (Done)

*EW(7) benchmark, TE targeting, MTUM investigation — completed March 2026.*

### What Was Built
1. **EW(7) benchmark** — paper-aligned equal-weight of 7 indices with quarterly rebalancing; active return and IR vs EW(7) reported alongside the SPY comparison across all test periods
2. **TE targeting** — view confidence Ω tuned so realized tracking error lands in the paper's 1–4% range
3. **MTUM investigation** — per-factor holdout Sharpe of −0.28 traced to hyperparameter sensitivity; separate tuning run isolates MTUM λ/κ²
4. **SPY vs EW(7) comparison** — quantifies the harder SPY hurdle over each test window; EW(7) consistently closer to Helix vol profile
5. **Sortino ratio** — added to all analysis output (`analyze_strategy.py`) and dashboard alongside Sharpe and Max DD

### How to Reproduce
```bash
# Tune hyperparameters (paper-aligned temporal split)
python tune_sjm_hyperparameters.py -c hyperparam/sjm_hyperparameters_best.json

# Full analysis with EW(7) + SPY benchmarks
python analyze_strategy.py -c hyperparam/sjm_hyperparameters_best.json --export
```

## JOH-12: Asymmetric Jump Penalty Matrix (In Progress)

*Relax the symmetric λ assumption — replace scalar penalty with a 2×2 matrix Λ.*

### Motivation

The Princeton SJM applies the same cost to all regime transitions: entering a stress regime costs the same as exiting one. Empirically this is wrong:

- **Crises arrive fast**: VIX spikes overnight, credit spreads gap in days
- **Recoveries are slow**: volatility mean-reverts over months, the 2022 rate-shock regime took ~18 months to normalize

### The Extension

Replace scalar λ with:

```
Λ = | 0          λ_enter |
    | λ_exit     0       |
```

Where **λ_enter >> λ_exit** so the model is reluctant to declare a new stress regime (high entry cost) but, once in stress, also slow to exit (high exit cost mirrors slow recovery). This directly relaxes the symmetric assumption in the paper.

### What to Test
- Grid-search or gradient optimization of (λ_enter, λ_exit) on holdout Sharpe per factor
- OOS regime detection quality vs. symmetric baseline — does asymmetric Λ better capture 2008, 2020, 2022?
- Portfolio OOS performance: does asymmetry improve BL Sharpe/Sortino or primarily smooth the regime sequence?

### How to Run (Asymmetric Export)
```bash
# Run both symmetric and asymmetric backtests; export 4-series JSON for dashboard
python analyze_strategy.py \
    -c hyperparam/sjm_hyperparameters_best.json \
    -a hyperparam/sjm_hyperparameters_asymmetric.json \
    --export

# Dashboard auto-loads backtest_data_asymmetric.json (set in dashboard/.env)
cd dashboard && npm run dev
# Override: npm run dev -- --config backtest_data.json
```

The exported JSON includes `helix` (symmetric), `helix_asym` (asymmetric), `ew7`, and `spy` — all with Sharpe, Sortino, volatility, and max drawdown. The dashboard renders all four as overlay chart lines with a toggle.

## Usage

### Quick Start
```bash
# Install Python deps
pip install numpy pandas yfinance scikit-learn scipy optuna

# Export backtest data (quick: 3 periods)
python analyze_strategy.py --export --quick

# Start dashboard
cd dashboard && npm install && npm run dev
```
Open [http://localhost:3000](http://localhost:3000).

### Full Backtest Export
```bash
# Symmetric only (→ backtest_data.json)
python analyze_strategy.py --export

# With tuned params (→ backtest_data_<timestamp>.json)
python analyze_strategy.py -c hyperparam/sjm_hyperparameters_best.json --export

# Symmetric + asymmetric comparison (→ backtest_data_asymmetric.json)
python analyze_strategy.py \
    -c hyperparam/sjm_hyperparameters_best.json \
    -a hyperparam/sjm_hyperparameters_asymmetric.json \
    --export --quick
```

### Hyperparameter Tuning
```bash
# Optuna search (50 trials per factor by default)
python tune_sjm_hyperparameters.py

# Paper-aligned temporal split
python tune_sjm_hyperparameters.py -c hyperparam/sjm_hyperparameters_best.json

# Fewer trials for speed
python tune_sjm_hyperparameters.py --n-trials 20 --no-wandb
```

### Production Pipeline (JOH-9)
```bash
python run_sjm_pipeline.py                                          # Full: tune → store → train → validate
python run_sjm_pipeline.py --step tune --n-trials 30
python run_sjm_pipeline.py --step store
python run_sjm_pipeline.py --step train
python run_sjm_pipeline.py --step validate
python run_sjm_pipeline.py --step promote --promote-version 20240311_120000
```

### Analysis CLI Reference
```
python analyze_strategy.py [options]

  -c, --config PATH       Symmetric SJM hyperparameter JSON
  -a, --asym-config PATH  Asymmetric SJM hyperparameter JSON (runs second backtest per period;
                          default export → backtest_data_asymmetric.json)
  --export                Write JSON for dashboard
  --quick                 3 periods only (faster)
  -o, --output PATH       Override output path
  --target-te FLOAT       Target tracking error (e.g. 0.02)
  --no-sticky             Disable sticky progress bar
```

## Dashboard

The Next.js dashboard at `dashboard/` visualizes all backtest output.

```bash
cd dashboard && npm run dev
# Select JSON at runtime:
npm run dev -- --config backtest_data_asymmetric.json
```

- **Period selector**: dropdown for all exported backtest windows
- **Comparison table**: Total Return | Sharpe | Sortino | Max DD across Helix (Sym), Helix (Asym), EW7, SPY
- **Portfolio chart**: 4-line overlay with per-series toggles
- **Allocation chart**: stacked factor weights over the OOS window
- **Train tab** (local, `.env` flag): live SJM training with streaming loss curve

Default JSON is set in `dashboard/.env`:
```
NEXT_PUBLIC_BACKTEST_JSON=backtest_data_asymmetric.json
```

## Documentation

- [docs/AUDIT_PRINCETON_PAPER.md](docs/AUDIT_PRINCETON_PAPER.md) — Alignment with Princeton paper
- [docs/POMDP_interpretation.md](docs/POMDP_interpretation.md) — POMDP framing of the regime model
- [docs/TRAINING_AND_REFIT_EXAMPLE.md](docs/TRAINING_AND_REFIT_EXAMPLE.md) — Walk-forward training and refit workflow

## Installation

```bash
pip install numpy pandas yfinance scikit-learn scipy optuna mlflow
# or
conda env create -f environment.yml && conda activate helix
```

Requires Python 3.11+.

## Theoretical Foundation

This implementation extends ["Portfolio Allocation Using Sparse Jump Model"](https://arxiv.org/html/2410.14841v1) (Shu, 2024):

- Regime identification across factor indices using jump models
- Daily rebalancing with Black-Litterman portfolio construction
- Information ratios vs. EW(7) benchmark (paper Exhibit 1)

Extensions beyond the paper:
1. **Per-factor hyperparameter tuning** (Optuna) on a held-out OOS window
2. **Sortino ratio** tracking alongside Sharpe
3. **Asymmetric jump penalty matrix** Λ (JOH-12) — relaxes symmetric λ assumption

## Disclaimer

Experimental research strategy. Past performance does not guarantee future results. Not financial advice.

## References

1. Shu, Y. O. (2024). ["Portfolio Allocation Using Sparse Jump Model"](https://arxiv.org/html/2410.14841v1). arXiv:2410.14841.
2. [Jump Models GitHub Repository](https://github.com/Yizhan-Oliver-Shu/jump-models)
3. Black, F., & Litterman, R. (1992). Global portfolio optimization. *Financial Analysts Journal*, 48(5), 28–43.

---

*Helix 1.1 — Regime-aware factor allocation with asymmetric jump penalty extension.*

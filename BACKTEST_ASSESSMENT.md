# Backtest Assessment vs Princeton Paper

**Date:** March 2026  
**Setup:** Hyperparams from `hyperparam/sjm_hyperparameters_best.json` (johnkimtroll timeline), OOS inference in `analyze_strategy.py`, walk-forward backtest.

**Linear:** [JOH-6](https://linear.app/johnkimdy/issue/JOH-6) (audit) | [JOH-10](https://linear.app/johnkimdy/issue/JOH-10) (follow-up todos)

---

## Summary

Implementation is largely aligned with the paper. Negative alpha vs SPY reflects a **stricter benchmark** (SPY vs paper's EW of 7 indices) and likely **regime shift** post-2022. Per-factor holdout: 5/6 positive; MTUM negative.

---

## Benchmark Mismatch (Important)

| Paper | Our backtest |
|-------|--------------|
| EW of 7 indices (market + 6 factors), quarterly rebal | SPY only |

EW(7) underperforms SPY in strong market periods. Our comparison is harsher than the paper's.

---

## Key Results

### Strategy vs SPY (representative periods)

- **Active return:** -1.2% to -3.5% (negative)
- **IR:** -0.05 to -0.31
- **Volatility:** 0.5–2% lower than SPY (positive)
- **Max DD:** Similar or slightly better than SPY in most periods
- **Sharpe:** Sometimes competitive (e.g. 1.50 vs 1.88 in 2023–2024)

### Per-factor holdout Sharpe (sjm_hyperparameters_best.json)

| Factor | Holdout Sharpe |
|--------|----------------|
| QUAL   | 0.04           |
| MTUM   | **-0.28**      |
| USMV   | 0.80           |
| VLUE   | 0.27           |
| SIZE   | 0.65           |
| IWF    | 0.86           |

5/6 positive; MTUM misfiring in holdout.

---

## Likely Drivers

1. **Benchmark:** SPY is a harder benchmark than EW(7).
2. **Regime shift:** Validation 2012–2022; holdout 2023+. Different macro regime.
3. **MTUM:** Negative holdout Sharpe; regime model may be wrong for momentum.
4. **TE targeting:** Paper targets 1–4% TE; not yet implemented (audit open item).

---

## Follow-up (JOH-10)

1. Add EW(7) benchmark to `analyze_strategy.py`.
2. Implement tracking-error targeting (1–4%).
3. Investigate MTUM regime structure and hyperparameters.
4. Compare SPY vs EW(7) over test periods.

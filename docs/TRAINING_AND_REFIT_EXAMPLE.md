# Example: Training and Refit by Day / Month

Assume **walk-forward is on** and **backtest window** is `start_date = 2024-01-01`, `end_date = 2025-08-31`. Data is fetched from ~504 days before start (so we have history for training).

---

## Part 1: When does refit happen? (Example OOS dates)

Refits run **once per month-end** in `_build_regime_getter`, **before** the backtest loop. For each month-end we fit SJMs on data **through the previous month-end** and cache that fit.

### Month-end loop (simplified)

| Iteration | `me` (current month-end) | `prev_me` (before update) | Training window `train_ar = active_returns.loc[:prev_me]` | Stored in `fits[prev_me]` |
|-----------|---------------------------|---------------------------|----------------------------------------------------------|---------------------------|
| 1         | 2023-12-31                | 2023-12-31 (initial)      | Data through **2023-12-31**                              | `fits[2023-12-31]`        |
| 2         | 2024-01-31                | 2023-12-31                | Data through **2023-12-31**                              | (same key, overwritten)   |
| 3         | 2024-02-29                | 2024-01-31                | Data through **2024-01-31**                              | `fits[2024-01-31]`        |
| 4         | 2024-03-31                | 2024-02-29                | Data through **2024-02-29**                              | `fits[2024-02-29]`        |
| 5         | 2024-04-30                | 2024-03-31                | Data through **2024-03-31**                              | `fits[2024-03-31]`        |
| …         | …                         | …                         | …                                                        | …                         |

So:

- **Refit at “2024-01-31”** (when `me = 2024-01-31`): we fit on `active_returns.loc[:2023-12-31]` and store in `fits[2023-12-31]`. So the model “for December” is trained on data through **2023-12-31**.
- **Refit at “2024-02-29”**: we fit on `active_returns.loc[:2024-01-31]` and store in `fits[2024-01-31]`. So the model “for January” is trained on data through **2024-01-31**.
- **Refit at “2024-03-31”**: we fit on `active_returns.loc[:2024-02-29]` and store in `fits[2024-02-29]`. So the model “for February” is trained on data through **2024-02-29**.

So **refit training for OOS** works like this:

- On **2024-01-31** we **refit** using only data through **2023-12-31** (no January data). That fit is used for regime inference in **February 2024**.
- On **2024-02-29** we **refit** using only data through **2024-01-31** (no February data). That fit is used for **March 2024**.
- On **2024-03-31** we **refit** using only data through **2024-02-29** (no March data). That fit is used for **April 2024**.

So each refit uses an **expanding window** that ends at the **previous** month-end; the current month is always OOS for that refit.

---

## Part 2: What happens on each day (example days)

For each trading day, the backtest calls `regime_getter(i, current_date)`. No refit runs here; we only **choose which cached model to use** and (if online inference is on) **infer regime** for that day.

### Step 1: Which model?

- `regime_date = returns_index[max(0, i - 2)]` → “T+2”: we use the date **2 days ago** to pick the model.
- We choose `use_me` = **largest month-end ≤ regime_date**.
- We use `fits[use_me]` → that month’s cached **models** and **regime_means**.

### Step 2: Regime for today

- **If online inference is off:** we return each factor’s **last in-sample regime** from that model (same for every day in the month).
- **If online inference is on:** we take **features** for that factor from `active_returns.loc[:regime_date]`, take the last `online_lookback_days` (e.g. 63) rows, scale with that model’s `scaler`, and call `model.infer_regime_online(X_scaled)` to get **one regime per factor** for that day.

### Example days (online inference on)

| Trading day (`current_date`) | `regime_date` (i−2) | `use_me` (model key) | Model was trained on data through | Regime inference |
|-----------------------------|---------------------|----------------------|------------------------------------|-------------------|
| 2024-01-05                  | 2024-01-03          | 2023-12-31           | 2023-12-31                         | Online inference on features through 2024-01-03 (lookback 63 days), using Dec model |
| 2024-01-16                  | 2024-01-14          | 2023-12-31           | 2023-12-31                         | Same Dec model; online inference on features through 2024-01-14 |
| 2024-02-01                  | 2024-01-30          | 2023-12-31           | 2023-12-31                         | Still Dec model (Jan 31 not ≤ Jan 30); online inference through 2024-01-30 |
| 2024-02-05                  | 2024-02-02          | 2024-01-31           | 2024-01-31                         | Jan model; online inference on features through 2024-02-02 |
| 2024-04-10                  | 2024-04-08          | 2024-03-31           | 2024-03-31                         | Mar model; online inference through 2024-04-08 |

So:

- **Training for each day** = we do **not** train that day. We use the **cached** SJM for the latest month-end ≤ regime_date, and (if online inference is on) we run **one forward pass** of online inference for that day’s feature window.
- **Refit training for OOS** = once per month-end we run **full SJM fit** on `active_returns.loc[:prev_me]` (and same for market_returns), then cache under `fits[prev_me]`. The examples above show how that lines up with calendar (e.g. refit on 2024-01-31 using data through 2023-12-31, then that model is used for February 2024).

---

## Summary

- **Refit:** Monthly at month-end; each refit uses data only through the **previous** month-end (expanding window, no future data).
- **Per day:** No refit. We pick the cached model for the latest month-end ≤ regime_date (T+2) and either use that model’s last in-sample regime or run online inference for that day.

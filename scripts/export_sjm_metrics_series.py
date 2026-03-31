#!/usr/bin/env python3
"""
Export time series of paper-aligned SJM metrics for dashboard visualization.

Computes per-factor hypothetical long-short strategy (position from regime expected
active return, ±5% cap; T+2 application) and daily regime, then writes JSON with:
- daily: [{ date, QUAL_regime, QUAL_position, QUAL_ls_ret, ... }, ...]
- cum_pnl_per_factor: { QUAL: [100, 100.1, ...], ... } (cumulative PnL, base 100)
- sharpe_per_factor: { QUAL: 0.5, ... } (full-period long-short Sharpe)
- start_date, end_date, factors

Usage:
  python scripts/export_sjm_metrics_series.py --start 2017-01-01 --end 2025-12-31 -c hyperparam/sjm_hyperparameters_best.json -o dashboard/public/sjm_metrics_series.json
"""

import argparse
import json
import os
import sys

if os.path.dirname(os.path.abspath(__file__)) != os.getcwd():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from helix_factor_strategy import (
    HelixFactorStrategy,
    compute_active_returns,
    EXPECTED_RETURN_CAP,
    MARKET_ETF,
)


def long_short_position(expected_active_ann):
    cap = EXPECTED_RETURN_CAP
    if expected_active_ann >= cap:
        return 1.0
    if expected_active_ann <= -cap:
        return -1.0
    return expected_active_ann / cap


def load_sjm_config(path):
    with open(path) as f:
        data = json.load(f)
    if "results" in data:
        return {
            f: {
                "jump_penalty": float(data["results"].get(f, {}).get("lambda", 50)),
                "sparsity_param": float(data["results"].get(f, {}).get("kappa_sq", 9.5)),
            }
            for f in ("QUAL", "MTUM", "USMV", "VLUE", "SIZE", "IWF")
        }
    if "factors" in data:
        return data["factors"]
    return {
        f: {
            "jump_penalty": data.get("jump_penalty", 50),
            "sparsity_param": data.get("sparsity_param", 9.5),
        }
        for f in ("QUAL", "MTUM", "USMV", "VLUE", "SIZE", "IWF")
    }


def main():
    ap = argparse.ArgumentParser(description="Export SJM long-short and regime series for dashboard")
    ap.add_argument("--start", default="2017-01-01", help="OOS start date")
    ap.add_argument("--end", default="2025-12-31", help="OOS end date")
    ap.add_argument("-c", "--config", required=True, help="Path to hyperparam JSON (tune output or paper_aligned)")
    ap.add_argument("-o", "--output", default=None, help="Output JSON path (default: dashboard/public/sjm_metrics_series.json)")
    ap.add_argument("--min-train-days", type=int, default=252, help="Min training days before first refit")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    sjm_config = load_sjm_config(args.config)
    # Use same lookback as backtest (252) so start_idx is 252; 252*12 would require 12y of data and empty loop
    strategy = HelixFactorStrategy(lookback_days=252)
    fetch_start = (pd.Timestamp(args.start) - pd.Timedelta(days=504)).strftime("%Y-%m-%d")
    strategy.fetch_data(fetch_start, args.end)
    returns = strategy.calculate_returns()
    active_returns = compute_active_returns(returns, market_col=MARKET_ETF)
    market_returns = returns[MARKET_ETF]

    start_idx = max(strategy.lookback_days, args.min_train_days)
    backtest_start_ts = pd.Timestamp(args.start)
    returns_index = returns.index
    factors = list(active_returns.columns)

    regime_getter = strategy._build_regime_getter(
        active_returns, market_returns, returns, args.start, args.end,
        sjm_config, use_walk_forward=True
    )

    rows = []
    for i in range(start_idx, len(returns)):
        current_date = returns_index[i]
        if current_date < backtest_start_ts:
            continue
        current_regimes, regime_means = regime_getter(i, current_date)
        row = {"date": current_date.strftime("%Y-%m-%d")}
        for f in factors:
            r = int(current_regimes.get(f, 0))
            ann = regime_means.get(f, {}).get(r, 0.0)
            pos = long_short_position(ann)
            row[f"{f}_regime"] = r
            row[f"{f}_position"] = round(pos, 6)
        rows.append(row)

    if not rows:
        print(
            "No days in range. Need len(returns) > start_idx ({}). Got {} returns from {} to {}.".format(
                start_idx, len(returns), fetch_start, args.end
            )
        )
        sys.exit(1)

    df_dates = pd.DatetimeIndex([r["date"] for r in rows])
    positions = pd.DataFrame(
        {f: [r[f"{f}_position"] for r in rows] for f in factors},
        index=df_dates
    )
    regimes = pd.DataFrame(
        {f: [r[f"{f}_regime"] for r in rows] for f in factors},
        index=df_dates
    )
    ar_slice = active_returns.loc[df_dates]
    pos_t2 = positions.shift(2).reindex(ar_slice.index).ffill().fillna(0)
    ls_ret = (pos_t2 * ar_slice).dropna(how="all")

    for j in range(len(rows)):
        dt = df_dates[j]
        for f in factors:
            val = ls_ret.loc[dt, f] if dt in ls_ret.index and f in ls_ret.columns else None
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                rows[j][f"{f}_ls_ret"] = round(float(val), 8)
            else:
                rows[j][f"{f}_ls_ret"] = None

    sharpe_per_factor = {}
    cum_pnl_per_factor = {}
    cum_pnl_dates = [d.strftime("%Y-%m-%d") for d in ls_ret.index]
    for f in factors:
        s = ls_ret[f].dropna()
        if len(s) < 20 or s.std() < 1e-10:
            sharpe_per_factor[f] = 0.0
        else:
            sharpe_per_factor[f] = round(float(s.mean() / s.std() * np.sqrt(252)), 4)
        cum = (1 + ls_ret[f].fillna(0)).cumprod()
        cum_pnl_per_factor[f] = [round(100 * x, 4) for x in cum.tolist()]

    out = {
        "start_date": args.start,
        "end_date": args.end,
        "factors": factors,
        "daily": rows,
        "sharpe_per_factor": sharpe_per_factor,
        "cum_pnl_per_factor": cum_pnl_per_factor,
        "cum_pnl_dates": cum_pnl_dates,
    }

    out_path = args.output or os.path.join(root, "dashboard", "public", "sjm_metrics_series.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote {} ({} days, factors: {})".format(out_path, len(rows), factors))
    print("Sharpe per factor:", sharpe_per_factor)


if __name__ == "__main__":
    main()

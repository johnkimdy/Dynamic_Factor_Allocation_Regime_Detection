#!/usr/bin/env python3.11
"""
Comprehensive analysis of Helix 1.1 Factor Strategy

Note: Walk-forward (monthly SJM refit) should be applied here for backtest, per paper.
See docs/TUNE_TEMPORAL_SPLIT_REPORT.md. Currently helix_factor_strategy.backtest fits once;
full walk-forward would refit regime models monthly over the backtest window.

Usage:
  python analyze_strategy.py                    # Full analysis (all periods)
  python analyze_strategy.py --export           # Same + export JSON for dashboard
  python analyze_strategy.py --export --quick   # Quick export (3 periods)
  python analyze_strategy.py -c hyperparam/sjm_hyperparameters_20260313.json --export
  # With -c but no -o: output defaults to backtest_data_20260313_051222.json

  # Asymmetric jump-penalty comparison (runs both sym and asym backtests per period):
  python analyze_strategy.py -c hyperparam/sjm_hyperparameters_best.json \
      -a hyperparam/sjm_hyperparameters_asymmetric.json --export
  # Output: dashboard/public/backtest_data_asymmetric.json (Helix Sym + Asym + EW7 + SPY)
"""

import argparse
import json
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helix_factor_strategy import HelixFactorStrategy
import logging
import yfinance as yf

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent

# Paper benchmark: EW of 7 indices (market + 6 factors), quarterly rebalancing
EW7_TICKERS = ['SPY', 'QUAL', 'MTUM', 'USMV', 'VLUE', 'SIZE', 'IWF']

# Fee assumptions for net-of-fees analysis
ETF_EXPENSE_RATIOS = {  # annual, decimal
    'SPY': 0.0003,   # 3 bps
    'QUAL': 0.0015,  'MTUM': 0.0015, 'USMV': 0.0015, 'VLUE': 0.0015, 'SIZE': 0.0015, 'IWF': 0.0019,
}
BLENDED_ER_7ASSETS = (ETF_EXPENSE_RATIOS['SPY'] + 6 * 0.0015 + ETF_EXPENSE_RATIOS['IWF']) / 7  # ~0.0013
BROKERAGE_BPS = 5.0   # 5 bps per unit turnover (paper)

# CLI runtime viz
ESTIMATE_SEC_PER_PERIOD = 90  # rough: backtest + benchmarks per period
PROGRESS_BAR_WIDTH = 24


def _format_duration(seconds: float) -> str:
    """Format seconds as 'Xm Ys' or 'Xh Ym'."""
    if seconds < 60:
        return "{:.0f}s".format(seconds)
    if seconds < 3600:
        m, s = int(seconds // 60), int(seconds % 60)
        return "{}m {}s".format(m, s) if s else "{}m".format(m)
    h, rest = int(seconds // 3600), seconds % 3600
    m = int(rest // 60)
    return "{}h {}m".format(h, m) if m else "{}h".format(h)


def _progress_bar(current: int, total: int, width: int = PROGRESS_BAR_WIDTH) -> str:
    """Return a string like [████████░░░░] for current/total."""
    if total <= 0:
        filled = 0
    else:
        filled = max(0, min(width, int(round(width * current / total))))
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _print_runtime_banner(n_periods: int, quick: bool) -> None:
    """Print a small banner with expected runtime."""
    est_total = n_periods * ESTIMATE_SEC_PER_PERIOD
    est_lo, est_hi = est_total * 0.7, est_total * 1.5
    msg = "📊 {} period{} · est. {} – {} · ☕".format(
        n_periods, "s" if n_periods != 1 else "", _format_duration(est_lo), _format_duration(est_hi))
    pad = 52 - len(msg)
    inner = "  " + msg + (" " * max(0, pad)) + "  "
    print()
    print("  ┌" + "─" * 56 + "┐")
    print("  │" + inner + "│")
    print("  └" + "─" * 56 + "┘")
    print()


def _print_period_progress(
    current: int, total: int, period_name: str, start_date: str, end_date: str,
    elapsed_sec: float, eta_sec: float | None,
) -> None:
    """Print one line: progress bar, step, period name, elapsed, ETA."""
    bar = _progress_bar(current, total)
    step = " {}/{} ".format(current, total)
    period = " {} ({} → {}) ".format(period_name, start_date, end_date)
    elapsed_s = " elapsed: {}".format(_format_duration(elapsed_sec))
    eta_s = " ETA: {}".format(_format_duration(eta_sec)) if eta_sec is not None and eta_sec >= 0 else ""
    line = "  {} {} {} · {} {}".format(bar, step.rjust(6), period, elapsed_s, eta_s)
    print(line)


# --- Sticky progress bar (fixed last line, output scrolls above) ---
_STICKY_STATE: dict = {}
_STICKY_LOCK = threading.Lock()
_STICKY_HEIGHT = 24
_STICKY_THREAD: threading.Thread | None = None
_STICKY_STOP = threading.Event()


def _sticky_terminalsize() -> int:
    try:
        return shutil.get_terminal_size().lines
    except Exception:
        return _STICKY_HEIGHT


def _sticky_draw_line(h: int, current: int, total: int, period_name: str, start_date: str, end_date: str,
                      elapsed_sec: float, eta_sec: float | None, done: bool = False,
                      total_elapsed_sec: float = 0.0, n_periods: int = 0) -> None:
    """Write progress (or done) to the last line, then move cursor to line above so next print() scrolls there."""
    # Move to last line, clear it
    sys.stdout.write("\033[{};1H\033[2K".format(h))
    if done:
        sys.stdout.write("  ✓ Done in {}  ({} period{})".format(
            _format_duration(total_elapsed_sec), n_periods, "s" if n_periods != 1 else ""))
    else:
        bar = _progress_bar(current, total)
        step = " {}/{} ".format(current, total)
        period = " {} ({} → {}) ".format(period_name, start_date, end_date)
        elapsed_s = " elapsed: {}".format(_format_duration(elapsed_sec))
        eta_s = " ETA: {}".format(_format_duration(eta_sec)) if eta_sec is not None and eta_sec >= 0 else ""
        sys.stdout.write("  {} {} {} · {} {}".format(bar, step.rjust(6), period, elapsed_s, eta_s))
    # So next print() goes into the scroll region (line above progress)
    sys.stdout.write("\033[{};1H".format(h - 1))
    sys.stdout.flush()


def _sticky_progress_worker(h: int) -> None:
    while not _STICKY_STOP.wait(timeout=1.5):
        with _STICKY_LOCK:
            s = dict(_STICKY_STATE)
        if s.get("done"):
            break
        idx = s.get("idx", 0)
        n = s.get("n", 1)
        if n < 1:
            continue
        run_start = s.get("run_start", time.perf_counter())
        period_times = s.get("period_times") or []
        elapsed = time.perf_counter() - run_start
        eta_sec = None
        if period_times:
            avg = sum(period_times) / len(period_times)
            eta_sec = avg * (n - idx)
        elif n > 1:
            eta_sec = ESTIMATE_SEC_PER_PERIOD * (n - idx)
        _sticky_draw_line(
            h, idx, n,
            s.get("period_name", ""), s.get("start_date", ""), s.get("end_date", ""),
            elapsed, eta_sec, done=False,
        )
    with _STICKY_LOCK:
        s = dict(_STICKY_STATE)
    _sticky_draw_line(
        h, s.get("idx", 0), s.get("n", 1),
        s.get("period_name", ""), s.get("start_date", ""), s.get("end_date", ""),
        0, None, done=True,
        total_elapsed_sec=s.get("total_elapsed", 0), n_periods=s.get("n", 0),
    )


def _sticky_progress_start(height: int) -> None:
    """Set scroll region to 1..(height-1), start background thread that redraws last line."""
    global _STICKY_THREAD
    # Scroll region: 1-based, top;bottom (only these lines scroll)
    sys.stdout.write("\033[1;{}r".format(height - 1))
    sys.stdout.flush()
    _STICKY_STOP.clear()
    _STICKY_THREAD = threading.Thread(target=_sticky_progress_worker, args=(height,), daemon=True)
    _STICKY_THREAD.start()


def _sticky_progress_stop(height: int) -> None:
    """Signal done, wait for thread, reset scroll region."""
    global _STICKY_THREAD
    _STICKY_STOP.set()
    if _STICKY_THREAD is not None:
        _STICKY_THREAD.join(timeout=3.0)
        _STICKY_THREAD = None
    sys.stdout.write("\033[r")  # reset scroll region
    sys.stdout.flush()


def apply_er_drag(returns: pd.Series, annual_er: float) -> pd.Series:
    """Apply daily expense-ratio drag. Returns net-of-ER daily returns."""
    daily_er = annual_er / 252.0
    return returns - daily_er


def _sortino(returns: pd.Series) -> float:
    """Annualized Sortino ratio (target return = 0, downside deviation denominator)."""
    downside = returns[returns < 0]
    if len(downside) == 0:
        return 0.0
    downside_std = float(np.sqrt((downside ** 2).mean()) * np.sqrt(252))
    return float(returns.mean() * 252 / downside_std) if downside_std > 0 else 0.0


def metrics_from_returns(returns: pd.Series) -> dict:
    """Compute total_return, sharpe, sortino, vol, max_dd from daily returns."""
    if returns.empty:
        return {'total_return': 0, 'sharpe': 0, 'sortino': 0, 'vol': 0, 'max_dd': 0}
    if returns.std() == 0 or pd.isna(returns.std()):
        return {'total_return': float((1 + returns).prod() - 1), 'sharpe': 0, 'sortino': 0, 'vol': 0, 'max_dd': 0}
    cum = (1 + returns).cumprod()
    total_return = float(cum.iloc[-1] / cum.iloc[0] - 1.0)
    vol = float(returns.std() * np.sqrt(252))
    sharpe = float(returns.mean() / returns.std() * np.sqrt(252))
    sortino = _sortino(returns)
    running_max = cum.expanding().max()
    dd = (cum - running_max) / running_max
    max_dd = float(dd.min())
    return {'total_return': total_return, 'sharpe': sharpe, 'sortino': sortino, 'vol': vol, 'max_dd': max_dd}


def compute_ew7_benchmark(start_date, end_date, apply_fees=False):
    """
    Compute equally weighted portfolio of 7 indices, rebalanced quarterly (paper benchmark).
    Returns dict with total_return, annualized_volatility, sharpe_ratio, max_drawdown, portfolio_values.
    If apply_fees=True, applies ER drag + brokerage at quarter-end rebalances.
    """
    raw = yf.download(EW7_TICKERS, start=start_date, end=end_date, auto_adjust=True, progress=False, threads=False)
    if raw.empty:
        return None
    if hasattr(raw.columns, 'nlevels') and raw.columns.nlevels > 1:
        prices = raw['Close'].copy()
    else:
        prices = raw[['Close']] if isinstance(raw, pd.DataFrame) else raw.to_frame()
    if isinstance(prices, pd.DataFrame):
        avail = [c for c in EW7_TICKERS if c in prices.columns]
        if len(avail) < 7:
            return None
        prices = prices[avail].ffill().dropna()
    if len(prices) < 10:
        return None

    returns = prices.pct_change().dropna()
    n = returns.shape[1]

    # Quarterly rebalancing: hold weights, let them drift, reset to 1/7 at quarter end
    quarter_labels = returns.index.to_period('Q')
    port_returns = []
    weights = np.ones(n) / n
    txn_cost_bps = BROKERAGE_BPS / 10000.0
    for i in range(len(returns)):
        r = returns.iloc[i].values
        port_ret = np.dot(weights, r)
        # Apply ER drag (daily)
        if apply_fees:
            port_ret -= BLENDED_ER_7ASSETS / 252.0
        port_returns.append(port_ret)
        # Update weights (drift)
        weights = weights * (1 + r)
        weights = weights / weights.sum()
        # Rebalance at quarter end; apply brokerage (cost on turnover from drifted weights to 1/7)
        if i + 1 < len(returns) and quarter_labels[i] != quarter_labels[i + 1]:
            if apply_fees:
                weight_chg = np.sum(np.abs(np.ones(n) / n - weights))
                port_returns[-1] -= txn_cost_bps * weight_chg  # brokerage reduces that day's return
            weights = np.ones(n) / n

    port_returns = pd.Series(port_returns, index=returns.index)
    port_values = (1 + port_returns).cumprod()
    port_values = port_values / port_values.iloc[0]

    total_return = float(port_values.iloc[-1] - 1.0)
    vol = float(port_returns.std() * np.sqrt(252))
    sharpe = float(port_returns.mean() / port_returns.std() * np.sqrt(252)) if port_returns.std() > 0 else 0.0
    sortino = _sortino(port_returns)
    running_max = port_values.expanding().max()
    dd = (port_values - running_max) / running_max
    max_dd = float(dd.min())

    return {
        'total_return': total_return,
        'annualized_volatility': vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_dd,
        'portfolio_values': port_values,
        'returns': port_returns,
    }


def compute_spy_benchmark(start_date, end_date):
    """SPY buy-and-hold. Returns dict with total_return, sharpe_ratio, volatility, max_drawdown, portfolio_values."""
    raw = yf.download('SPY', start=start_date, end=end_date, auto_adjust=True, progress=False)
    if raw.empty:
        return None
    prices = raw['Close'].squeeze()
    if len(prices) < 10:
        return None
    port_values = prices / prices.iloc[0]
    returns = port_values.pct_change().dropna()
    total_return = float(port_values.iloc[-1] - 1.0)
    vol = float(returns.std() * np.sqrt(252))
    sharpe = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0
    sortino = _sortino(returns)
    running_max = port_values.expanding().max()
    dd = (port_values - running_max) / running_max
    max_dd = float(dd.min())
    return {
        'total_return': total_return,
        'annualized_volatility': vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_dd,
        'portfolio_values': port_values,
    }


def _serialize_period_for_export(start_date, end_date, period_name, results, ew7_data, spy_data, asym_results=None):
    """Convert backtest results to JSON-serializable format for dashboard.

    asym_results: optional second backtest results dict (asymmetric hyperparams).
    When provided, adds asym_values and benchmarks.helix_asym to the export.
    """
    pv = results['portfolio_values']
    dates = [d.strftime('%Y-%m-%d') for d in pv.index]
    values = [float(v) for v in pv.values]
    weights_history = [
        {'date': date.strftime('%Y-%m-%d'), 'weights': {k: float(v) for k, v in w.items()}}
        for date, w in results['weights_history']
    ]
    rebalance_dates = [d.strftime('%Y-%m-%d') for d in results['rebalance_dates']]
    n_days = len(pv)
    years = n_days / 252.0 if n_days > 0 else 0
    ann_ret = (pv.iloc[-1] / pv.iloc[0]) ** (1 / years) - 1 if years > 0 and pv.iloc[0] > 0 else 0
    helix_returns = results['portfolio_values'].pct_change().dropna()
    helix_metrics = {
        'total_return': float(results['total_return']),
        'annualized_return': float(ann_ret),
        'sharpe_ratio': float(results['sharpe_ratio']),
        'sortino_ratio': float(_sortino(helix_returns)),
        'volatility': float(results['annualized_volatility']),
        'max_drawdown': float(results['max_drawdown']),
        'n_rebalances': int(results['n_rebalances']),
    }
    ew7_values = []
    spy_values = []
    ew7_metrics = None
    spy_metrics = None
    if ew7_data is not None:
        ew7_pv = ew7_data['portfolio_values']
        ew7_metrics = {
            'total_return': float(ew7_data['total_return']),
            'sharpe_ratio': float(ew7_data['sharpe_ratio']),
            'sortino_ratio': float(ew7_data['sortino_ratio']),
            'volatility': float(ew7_data['annualized_volatility']),
            'max_drawdown': float(ew7_data['max_drawdown']),
        }
        ew7_aligned = ew7_pv.reindex(pv.index).ffill().bfill()
        if len(ew7_aligned) > 0 and ew7_aligned.notna().all():
            base = ew7_aligned.iloc[0]
            ew7_values = [
                [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)[:10], float(v / base * 100)]
                for d, v in ew7_aligned.items()
            ]
    if spy_data is not None:
        spy_pv = spy_data['portfolio_values']
        spy_metrics = {
            'total_return': float(spy_data['total_return']),
            'sharpe_ratio': float(spy_data['sharpe_ratio']),
            'sortino_ratio': float(spy_data['sortino_ratio']),
            'volatility': float(spy_data['annualized_volatility']),
            'max_drawdown': float(spy_data['max_drawdown']),
        }
        spy_aligned = spy_pv.reindex(pv.index).ffill().bfill()
        if len(spy_aligned) > 0 and spy_aligned.notna().all():
            base = spy_aligned.iloc[0]
            spy_values = [
                [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)[:10], float(v / base * 100)]
                for d, v in spy_aligned.items()
            ]

    # Asymmetric backtest (optional second run with different SJM config)
    asym_values = []
    asym_metrics = None
    if asym_results is not None:
        asym_pv = asym_results['portfolio_values']
        asym_n_days = len(asym_pv)
        asym_years = asym_n_days / 252.0 if asym_n_days > 0 else 0
        asym_ann_ret = (
            (asym_pv.iloc[-1] / asym_pv.iloc[0]) ** (1 / asym_years) - 1
            if asym_years > 0 and asym_pv.iloc[0] > 0 else 0
        )
        asym_returns = asym_pv.pct_change().dropna()
        asym_metrics = {
            'total_return': float(asym_results['total_return']),
            'annualized_return': float(asym_ann_ret),
            'sharpe_ratio': float(asym_results['sharpe_ratio']),
            'sortino_ratio': float(_sortino(asym_returns)),
            'volatility': float(asym_results['annualized_volatility']),
            'max_drawdown': float(asym_results['max_drawdown']),
            'n_rebalances': int(asym_results['n_rebalances']),
        }
        asym_aligned = asym_pv.reindex(pv.index).ffill().bfill()
        if len(asym_aligned) > 0 and asym_aligned.notna().all():
            base = asym_aligned.iloc[0]
            asym_values = [
                [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)[:10], float(v / base * 100)]
                for d, v in asym_aligned.items()
            ]

    return {
        'period': period_name,
        'start_date': start_date,
        'end_date': end_date,
        'metrics': helix_metrics,
        'benchmarks': {'helix': helix_metrics, 'helix_asym': asym_metrics, 'ew7': ew7_metrics, 'spy': spy_metrics},
        'portfolio_values': list(zip(dates, values)),
        'asym_values': asym_values,
        'ew7_values': ew7_values,
        'spy_values': spy_values,
        'weights_history': weights_history,
        'rebalance_dates': rebalance_dates,
    }


def run_analysis(use_tuned_params=True, sjm_config_path=None, asym_sjm_config_path=None,
                 target_tracking_error=None, export_output=None, quick_export=False, no_sticky=False):
    """Run comprehensive analysis of the Helix 1.1 strategy.

    sjm_config_path: optional path to tune output JSON (symmetric / primary config).
    asym_sjm_config_path: optional path to a second (asymmetric jump-penalty) config.
      When set, each period runs a second backtest and the export JSON includes both
      helix (sym) and helix_asym results plus asym_values for the chart.
    target_tracking_error: optional (e.g. 0.02 for 2%). Paper targets 1-4% TE.
    no_sticky: if True, disable sticky progress bar (use when terminal doesn't support it).
    """
    strategy = HelixFactorStrategy(target_tracking_error=target_tracking_error)
    sjm_config = None
    if use_tuned_params:
        if sjm_config_path:
            import os
            from hyperparam_io import load_run
            path = sjm_config_path
            if not os.path.exists(path) and not os.path.isabs(path):
                path = os.path.join("hyperparam", os.path.basename(path))
            sjm_config, _ = load_run(path)
            if sjm_config:
                print("Using SJM params from: {}".format(path))
        if not sjm_config:
            from hyperparam_io import load_sjm_config
            sjm_config = load_sjm_config(use_best=True)
            if sjm_config:
                print("Using tuned SJM params from hyperparam/ (best)")
        if not sjm_config:
            try:
                from sjm_params_store import load
                sjm_config = load(source="local", version_or_alias="production")
                if not sjm_config:
                    sjm_config = load(source="local", version_or_alias="latest")
                if sjm_config:
                    print("Using tuned SJM params from param store")
            except Exception as e:
                print("Param store load failed ({}), using paper defaults".format(e))
        if not sjm_config:
            print("Using paper defaults (λ=50, κ²=9.5)")

    # Load asymmetric config (optional second backtest per period)
    asym_sjm_config = None
    if asym_sjm_config_path:
        try:
            from hyperparam_io import load_run
            asym_path = asym_sjm_config_path
            if not os.path.exists(asym_path) and not os.path.isabs(asym_path):
                asym_path = os.path.join("hyperparam", os.path.basename(asym_path))
            asym_sjm_config, _ = load_run(asym_path)
            if asym_sjm_config:
                print("Using asymmetric SJM params from: {}".format(asym_path))
            else:
                print("Warning: could not load asymmetric config from {}".format(asym_path))
        except Exception as e:
            print("Warning: failed to load asymmetric config ({}), skipping asym backtest".format(e))

    # End date for "current" periods: use today so data is always up to date
    _end_now = pd.Timestamp.now().strftime('%Y-%m-%d')
    # Test periods: grouped by ending year, sorted by starting year desc within each group
    QUICK_PERIODS = [
        ('2024-01-01', _end_now, '2024-now'),
        ('2022-01-01', '2024-12-31', '2022-2024'),
        ('2021-01-01', '2023-12-31', '2021-2023'),
    ]
    _raw_periods = [
        # Ending now (latest data)
        ('2024-01-01', _end_now, '2024-now'),
        ('2023-01-01', _end_now, '2023-now'),
        ('2022-01-01', _end_now, '2022-now'),
        ('2021-01-01', _end_now, '2021-now'),
        ('2020-01-01', _end_now, '2020-now'),
        ('2019-01-01', _end_now, '2019-now'),
        ('2018-01-01', _end_now, '2018-now'),
        ('2017-01-01', _end_now, '2017-now'),
        # Ending 2024
        ('2023-01-01', '2024-12-31', '2023-2024'),
        ('2022-01-01', '2024-12-31', '2022-2024'),
        ('2021-01-01', '2024-12-31', '2021-2024'),
        # Paper test period (Exhibit 1: 2007-2024)
        ('2007-01-01', '2024-12-31', '2007-2024'),
        ('2007-01-01', '2023-12-31', '2007-2023'),
        ('2007-01-01', '2022-12-31', '2007-2022'),
        # Ending 2023
        ('2022-01-01', '2023-12-31', '2022-2023'),
        ('2021-01-01', '2023-12-31', '2021-2023'),
        # Ending 2022
        ('2021-01-01', '2022-12-31', '2021-2022'),
    ]
    _all_periods = sorted(_raw_periods, key=lambda p: (-int(p[1][:4]), -int(p[0][:4])))
    test_periods = sorted(QUICK_PERIODS, key=lambda p: (-int(p[1][:4]), -int(p[0][:4]))) if quick_export else _all_periods

    results_summary = []
    export_periods = []  # for --export
    
    print("=" * 60)
    print("HELIX 1.1 FACTOR STRATEGY ANALYSIS")
    print("=" * 60)

    _print_runtime_banner(len(test_periods), quick_export)
    run_start = time.perf_counter()
    period_times: list[float] = []
    use_sticky = sys.stdout.isatty() and not no_sticky
    sticky_height = _sticky_terminalsize() if use_sticky else 0
    if use_sticky:
        _sticky_progress_start(sticky_height)

    for idx, (start_date, end_date, period_name) in enumerate(test_periods, start=1):
        n = len(test_periods)
        elapsed = time.perf_counter() - run_start
        eta_sec = None
        if period_times:
            avg = sum(period_times) / len(period_times)
            eta_sec = avg * (n - idx)
        elif n > 1:
            eta_sec = ESTIMATE_SEC_PER_PERIOD * (n - idx)
        if use_sticky:
            with _STICKY_LOCK:
                _STICKY_STATE.update({
                    "idx": idx, "n": n, "period_name": period_name,
                    "start_date": start_date, "end_date": end_date,
                    "run_start": run_start, "period_times": period_times,
                })
        else:
            _print_period_progress(idx, n, period_name, start_date, end_date, elapsed, eta_sec)
        print("-" * 40)

        period_start = time.perf_counter()
        try:
            results = strategy.backtest(start_date, end_date, sjm_config=sjm_config)
            
            # Extract key metrics (gross for Helix = already includes 5bps brokerage in backtest)
            helix_returns_gross = results['portfolio_values'].pct_change().dropna()
            metrics = {
                'period': period_name,
                'start_date': start_date,
                'end_date': end_date,
                'total_return': results['total_return'],
                'sharpe_ratio': results['sharpe_ratio'],
                'sortino_ratio': _sortino(helix_returns_gross),
                'volatility': results['annualized_volatility'],
                'max_drawdown': results['max_drawdown'],
                'n_rebalances': results['n_rebalances']
            }

            # Fee-adjusted metrics: Helix gets ER drag (brokerage already in backtest)
            helix_returns = results['portfolio_values'].pct_change().dropna()
            helix_returns_net = apply_er_drag(helix_returns, BLENDED_ER_7ASSETS)
            helix_net = metrics_from_returns(helix_returns_net)
            metrics['helix_net_return'] = helix_net['total_return']
            metrics['helix_net_sharpe'] = helix_net['sharpe']
            metrics['helix_net_sortino'] = helix_net['sortino']
            metrics['helix_net_vol'] = helix_net['vol']
            metrics['helix_net_dd'] = helix_net['max_dd']

            results_summary.append(metrics)

            if export_output is not None:
                ew7_export = compute_ew7_benchmark(start_date, end_date)
                spy_export = compute_spy_benchmark(start_date, end_date)
                asym_results_export = None
                if asym_sjm_config is not None:
                    try:
                        asym_results_export = strategy.backtest(start_date, end_date, sjm_config=asym_sjm_config)
                        print("  Asym backtest done: Sharpe={:.2f}, Sortino={:.2f}".format(
                            asym_results_export['sharpe_ratio'],
                            _sortino(asym_results_export['portfolio_values'].pct_change().dropna()),
                        ))
                    except Exception as e:
                        print("  Asym backtest failed for {}: {}".format(period_name, e))
                export_periods.append(_serialize_period_for_export(
                    start_date, end_date, period_name, results,
                    ew7_export, spy_export, asym_results_export,
                ))

            # Print results
            print("Total Return: {:.2%}".format(results['total_return']))
            print("Sharpe Ratio: {:.2f}".format(results['sharpe_ratio']))
            print("Sortino Ratio: {:.2f}".format(metrics['sortino_ratio']))
            print("Annualized Volatility: {:.2%}".format(results['annualized_volatility']))
            print("Max Drawdown: {:.2%}".format(results['max_drawdown']))
            print("Number of Rebalances: {}".format(results['n_rebalances']))
            
            # Calculate annualized return
            portfolio_values = results['portfolio_values']
            n_days = len(portfolio_values)
            years = n_days / 252.0
            annualized_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (1/years) - 1
            print("Annualized Return: {:.2%}".format(annualized_return))
            
            # Calculate active return metrics vs SPY and vs EW(7) (paper benchmark)
            try:
                spy_raw = yf.download('SPY', start=start_date, end=end_date, auto_adjust=True, progress=False)
                spy_data = spy_raw['Close'].squeeze()

                spy_total_return = (spy_data.iloc[-1] / spy_data.iloc[0]) - 1
                spy_annualized = (1 + spy_total_return) ** (1/years) - 1
                strategy_annualized = annualized_return
                spy_returns_gross = spy_data.pct_change().dropna()
                metrics['spy_sharpe'] = float(spy_returns_gross.mean() / spy_returns_gross.std() * np.sqrt(252)) if spy_returns_gross.std() > 0 else 0
                metrics['spy_sortino'] = _sortino(spy_returns_gross)
                metrics['spy_vol'] = float(spy_returns_gross.std() * np.sqrt(252))
                spy_cum = (1 + spy_returns_gross).cumprod()
                metrics['spy_max_dd'] = float(((spy_cum - spy_cum.expanding().max()) / spy_cum.expanding().max()).min())

                # SPY net of fees (ER only; buy-and-hold = no brokerage)
                spy_returns = spy_data.pct_change().dropna()
                spy_returns_net = apply_er_drag(spy_returns, ETF_EXPENSE_RATIOS['SPY'])
                spy_net = metrics_from_returns(spy_returns_net)
                metrics['spy_net_return'] = spy_net['total_return']
                metrics['spy_net_sharpe'] = spy_net['sharpe']
                metrics['spy_net_sortino'] = spy_net['sortino']
                metrics['spy_net_vol'] = spy_net['vol']
                metrics['spy_net_dd'] = spy_net['max_dd']

                active_return = strategy_annualized - spy_annualized
                information_ratio = active_return / results['annualized_volatility'] if results['annualized_volatility'] > 0 else 0

                print("Active Return (vs SPY): {:.2%}".format(float(active_return)))
                print("Information Ratio vs SPY (approx): {:.2f}".format(float(information_ratio)))

                metrics['active_return'] = active_return
                metrics['information_ratio'] = information_ratio
                metrics['active_volatility'] = results['annualized_volatility']
                metrics['spy_return'] = spy_total_return

                ew7 = compute_ew7_benchmark(start_date, end_date)
                ew7_net = compute_ew7_benchmark(start_date, end_date, apply_fees=True) if ew7 is not None else None
                if ew7 is not None:
                    ew7_annualized = (1 + ew7['total_return']) ** (1/years) - 1
                    active_return_ew7 = strategy_annualized - ew7_annualized
                    ir_ew7 = active_return_ew7 / results['annualized_volatility'] if results['annualized_volatility'] > 0 else 0
                    metrics['ew7_return'] = ew7['total_return']
                    metrics['active_return_ew7'] = active_return_ew7
                    metrics['information_ratio_ew7'] = ir_ew7
                    metrics['ew7_sharpe'] = ew7['sharpe_ratio']
                    metrics['ew7_sortino'] = ew7['sortino_ratio']
                    metrics['ew7_vol'] = ew7['annualized_volatility']
                    metrics['ew7_max_dd'] = ew7['max_drawdown']
                    metrics['ew7_net_return'] = ew7_net['total_return'] if ew7_net else None
                    metrics['ew7_net_sharpe'] = ew7_net['sharpe_ratio'] if ew7_net else None
                    metrics['ew7_net_sortino'] = ew7_net['sortino_ratio'] if ew7_net else None
                    metrics['ew7_net_vol'] = ew7_net['annualized_volatility'] if ew7_net else None
                    metrics['ew7_net_dd'] = ew7_net['max_drawdown'] if ew7_net else None
                    print("Active Return (vs EW7, paper bmk): {:.2%}".format(float(active_return_ew7)))
                    print("Information Ratio vs EW7 (approx): {:.2f}".format(float(ir_ew7)))
                else:
                    metrics['ew7_return'] = None
                    metrics['active_return_ew7'] = None
                    metrics['information_ratio_ew7'] = None
                    metrics['ew7_sharpe'] = None
                    metrics['ew7_sortino'] = None
                    metrics['ew7_vol'] = None
                    metrics['ew7_max_dd'] = None
                    metrics['ew7_net_return'] = None
                    metrics['ew7_net_sharpe'] = None
                    metrics['ew7_net_sortino'] = None
                    metrics['ew7_net_vol'] = None
                    metrics['ew7_net_dd'] = None

            except Exception as e:
                print("Could not calculate active return metrics: {}".format(e))
                metrics['active_return'] = 0
                metrics['information_ratio'] = 0
                metrics['active_volatility'] = 0
                metrics['spy_return'] = None
                metrics['spy_sharpe'] = None
                metrics['spy_sortino'] = None
                metrics['spy_vol'] = None
                metrics['spy_max_dd'] = None
                metrics['ew7_return'] = None
                metrics['active_return_ew7'] = None
                metrics['information_ratio_ew7'] = None
                metrics['ew7_sharpe'] = None
                metrics['ew7_sortino'] = None
                metrics['ew7_vol'] = None
                metrics['ew7_max_dd'] = None
                metrics['spy_net_return'] = None
                metrics['spy_net_sharpe'] = None
                metrics['spy_net_sortino'] = None
                metrics['spy_net_vol'] = None
                metrics['spy_net_dd'] = None
                metrics['ew7_net_return'] = None
                metrics['ew7_net_sharpe'] = None
                metrics['ew7_net_sortino'] = None
                metrics['ew7_net_vol'] = None
                metrics['ew7_net_dd'] = None
            
        except Exception as e:
            print("Error during backtest: {}".format(e))
        finally:
            period_times.append(time.perf_counter() - period_start)

    total_elapsed = time.perf_counter() - run_start
    if use_sticky:
        with _STICKY_LOCK:
            _STICKY_STATE.update({
                "done": True, "total_elapsed": total_elapsed, "n": len(test_periods),
            })
        _sticky_progress_stop(sticky_height)
        print()
    else:
        print()
        print("  ✓ Done in {}  ({} period{})".format(
            _format_duration(total_elapsed), len(test_periods), "s" if len(test_periods) != 1 else ""))

    # Create summary table
    if results_summary:
        print("\n" + "=" * 110)
        print("SUMMARY TABLE")
        print("=" * 110)

        df_results = pd.DataFrame(results_summary)

        print("Period       | Total Ret | Sharpe | Sortino | Vol   | Max DD | Act Ret(SPY) | IR(SPY) | Act Ret(EW7) | IR(EW7) | Rebal")
        print("-" * 120)
        for _, row in df_results.iterrows():
            ar_ew7 = row.get('active_return_ew7')
            ir_ew7 = row.get('information_ratio_ew7')
            ar_ew7_val = float(ar_ew7) if ar_ew7 is not None and not (isinstance(ar_ew7, float) and pd.isna(ar_ew7)) else None
            ir_ew7_val = float(ir_ew7) if ir_ew7 is not None and not (isinstance(ir_ew7, float) and pd.isna(ir_ew7)) else None
            ar_ew7_s = "{:>11.2%}".format(ar_ew7_val) if ar_ew7_val is not None else "        n/a"
            ir_ew7_s = "{:>7.2f}".format(ir_ew7_val) if ir_ew7_val is not None else "    n/a"
            print("{:<12} | {:>8.2%} | {:>6.2f} | {:>7.2f} | {:>6.2%} | {:>6.2%} | {:>11.2%} | {:>7.2f} | {:>13} | {:>7} | {:>5}".format(
                str(row['period']),
                float(row['total_return']),
                float(row['sharpe_ratio']),
                float(row.get('sortino_ratio', 0)),
                float(row['volatility']),
                float(row['max_drawdown']),
                float(row.get('active_return', 0)),
                float(row.get('information_ratio', 0)),
                ar_ew7_s,
                ir_ew7_s,
                int(row['n_rebalances'])
            ))
    
    # Benchmark comparison vs SPY for all timeframes
    print("\n" + "=" * 110)
    print("BENCHMARK COMPARISON vs SPY (ALL PERIODS)")
    print("=" * 110)
    
    try:
        comparison_rows = []
        for strat in results_summary:
            start_date = strat['start_date']
            end_date = strat['end_date']
            period_name = strat['period']
            try:
                spy_raw = yf.download('SPY', start=start_date, end=end_date, auto_adjust=True, progress=False)
                spy_data = spy_raw['Close'].squeeze()
                spy_returns = spy_data.pct_change().dropna()
                
                spy_total_return = float((spy_data.iloc[-1] / spy_data.iloc[0]) - 1)
                spy_volatility = float(spy_returns.std() * np.sqrt(252))
                spy_sharpe = float(spy_returns.mean() / spy_returns.std() * np.sqrt(252))
                
                spy_cumulative = spy_data / spy_data.iloc[0]
                spy_running_max = spy_cumulative.expanding().max()
                spy_drawdown = (spy_cumulative - spy_running_max) / spy_running_max
                spy_max_dd = float(spy_drawdown.min())
                
                spy_sortino = _sortino(spy_returns)
                comparison_rows.append({
                    'period': period_name,
                    'helix_return': strat['total_return'],
                    'spy_return': spy_total_return,
                    'return_diff': strat['total_return'] - spy_total_return,
                    'helix_sharpe': strat['sharpe_ratio'],
                    'spy_sharpe': spy_sharpe,
                    'sharpe_diff': strat['sharpe_ratio'] - spy_sharpe,
                    'helix_sortino': strat.get('sortino_ratio', 0),
                    'spy_sortino': spy_sortino,
                    'sortino_diff': strat.get('sortino_ratio', 0) - spy_sortino,
                    'helix_vol': strat['volatility'],
                    'spy_vol': spy_volatility,
                    'vol_diff': strat['volatility'] - spy_volatility,
                    'helix_dd': strat['max_drawdown'],
                    'spy_dd': spy_max_dd,
                })
            except Exception as e:
                print("Could not fetch SPY for {}: {}".format(period_name, e))
        
        if comparison_rows:
            print("\n{:<12} | {:>12} | {:>12} | {:>10} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8}".format(
                "Period", "Helix Ret", "SPY Ret", "Ret Diff", "Helix SR", "SPY SR", "SR Diff", "Helix Vol", "SPY Vol", "Vol Diff"
            ))
            print("-" * 110)
            for r in comparison_rows:
                print("{:<12} | {:>11.2%} | {:>11.2%} | {:>+9.2%} | {:>7.2f} | {:>7.2f} | {:>+7.2f} | {:>7.2%} | {:>7.2%} | {:>+7.2%}".format(
                    r['period'],
                    r['helix_return'],
                    r['spy_return'],
                    r['return_diff'],
                    r['helix_sharpe'],
                    r['spy_sharpe'],
                    r['sharpe_diff'],
                    r['helix_vol'],
                    r['spy_vol'],
                    r['vol_diff']
                ))
            print("\n{:<12} | {:>12} | {:>12} | {:>10}".format("Period", "Helix Max DD", "SPY Max DD", "DD Diff"))
            print("-" * 55)
            for r in comparison_rows:
                dd_diff = r['helix_dd'] - r['spy_dd']
                print("{:<12} | {:>11.2%} | {:>11.2%} | {:>+9.2%}".format(
                    r['period'], r['helix_dd'], r['spy_dd'], dd_diff
                ))
            print("\n{:<12} | {:>14} | {:>14} | {:>12}".format("Period", "Helix Sortino", "SPY Sortino", "Sortino Diff"))
            print("-" * 60)
            for r in comparison_rows:
                print("{:<12} | {:>13.2f} | {:>13.2f} | {:>+11.2f}".format(
                    r['period'], r['helix_sortino'], r['spy_sortino'], r['sortino_diff']
                ))

    except Exception as e:
        print("Error getting SPY benchmark: {}".format(e))

    # Helix vs EW7 vs SPY — Gross and Net of Fees (ER + brokerage)
    print("\n" + "=" * 130)
    print("HELIX vs EW(7) vs SPY — Gross vs Net of Fees (ER + brokerage)")
    print("Fee assumptions: Helix/EW7 ER ~0.13%% pa (7-ETF blend); SPY ER 0.03%%; brokerage 5 bps per turnover; Helix=100s rebalances, EW7=4/yr, SPY=0")
    print("=" * 130)
    if results_summary:
        print("\nGROSS (before fees):")
        print("{:<12} | {:>10} {:>10} {:>10} | {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8} | {:>5}".format(
            "Period", "Helix Ret", "EW7 Ret", "SPY Ret", "Helix SR", "EW7 SR", "SPY SR",
            "Helix Vol", "EW7 Vol", "SPY Vol", "Helix DD", "EW7 DD", "SPY DD", "Rebal"
        ))
        print("-" * 130)
        for s in results_summary:
            ew7_r = s.get('ew7_return')
            ew7_s = s.get('ew7_sharpe')
            ew7_v = s.get('ew7_vol')
            ew7_d = s.get('ew7_max_dd')
            spy_r = s.get('spy_return')
            spy_s = s.get('spy_sharpe')
            spy_v = s.get('spy_vol')
            spy_d = s.get('spy_max_dd')
            ew7_ret_s = "{:>9.2%}".format(ew7_r) if ew7_r is not None else "      n/a"
            spy_ret_s = "{:>9.2%}".format(spy_r) if spy_r is not None else "      n/a"
            print("{:<12} | {:>9.2%} {:>10} {:>10} | {:>7.2f} {:>7.2f} {:>7.2f} | {:>7.2%} {:>7.2%} {:>7.2%} | {:>7.2%} {:>7.2%} {:>7.2%} | {:>5}".format(
                s['period'],
                s['total_return'], ew7_ret_s, spy_ret_s,
                s['sharpe_ratio'], ew7_s or 0, spy_s or 0,
                s['volatility'], ew7_v or 0, spy_v or 0,
                s['max_drawdown'], ew7_d or 0, spy_d or 0,
                s['n_rebalances']
            ))
        print("-" * 130)
        print("GROSS SORTINO: Helix | EW7 | SPY")
        print("{:<12} | {:>12} | {:>12} | {:>12}".format("Period", "Helix Sortino", "EW7 Sortino", "SPY Sortino"))
        print("-" * 58)
        for s in results_summary:
            print("{:<12} | {:>12.2f} | {:>12.2f} | {:>12.2f}".format(
                s['period'],
                s.get('sortino_ratio') or 0,
                s.get('ew7_sortino') or 0,
                s.get('spy_sortino') or 0,
            ))
        print("-" * 130)
        print("NET (after ER + brokerage):")
        print("{:<12} | {:>10} {:>10} {:>10} | {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8} | {:>5}".format(
            "Period", "Helix Ret", "EW7 Ret", "SPY Ret", "Helix SR", "EW7 SR", "SPY SR",
            "Helix Vol", "EW7 Vol", "SPY Vol", "Helix DD", "EW7 DD", "SPY DD", "Rebal"
        ))
        print("-" * 130)
        for s in results_summary:
            e_net = s.get('ew7_net_return')
            sp_net = s.get('spy_net_return')
            e_ret_s = "{:>9.2%}".format(e_net) if e_net is not None else "      n/a"
            sp_ret_s = "{:>9.2%}".format(sp_net) if sp_net is not None else "      n/a"
            print("{:<12} | {:>9.2%} {:>10} {:>10} | {:>7.2f} {:>7.2f} {:>7.2f} | {:>7.2%} {:>7.2%} {:>7.2%} | {:>7.2%} {:>7.2%} {:>7.2%} | {:>5}".format(
                s['period'],
                s.get('helix_net_return') or 0, e_ret_s, sp_ret_s,
                s.get('helix_net_sharpe') or 0, s.get('ew7_net_sharpe') or 0, s.get('spy_net_sharpe') or 0,
                s.get('helix_net_vol') or 0, s.get('ew7_net_vol') or 0, s.get('spy_net_vol') or 0,
                s.get('helix_net_dd') or 0, s.get('ew7_net_dd') or 0, s.get('spy_net_dd') or 0,
                s['n_rebalances']
            ))
        print("NET SORTINO: Helix | EW7 | SPY")
        print("{:<12} | {:>12} | {:>12} | {:>12}".format("Period", "Helix Sortino", "EW7 Sortino", "SPY Sortino"))
        print("-" * 58)
        for s in results_summary:
            print("{:<12} | {:>12.2f} | {:>12.2f} | {:>12.2f}".format(
                s['period'],
                s.get('helix_net_sortino') or 0,
                s.get('ew7_net_sortino') or 0,
                s.get('spy_net_sortino') or 0,
            ))

    # Benchmark comparison vs EW(7) (paper benchmark)
    print("\n" + "=" * 110)
    print("BENCHMARK COMPARISON vs EW(7) [PAPER BENCHMARK] (ALL PERIODS)")
    print("=" * 110)

    try:
        ew7_rows = []
        for strat in results_summary:
            start_date = strat['start_date']
            end_date = strat['end_date']
            period_name = strat['period']
            ew7 = compute_ew7_benchmark(start_date, end_date)
            if ew7 is None:
                continue
            ew7_rows.append({
                'period': period_name,
                'helix_return': strat['total_return'],
                'ew7_return': ew7['total_return'],
                'return_diff': strat['total_return'] - ew7['total_return'],
                'helix_sharpe': strat['sharpe_ratio'],
                'ew7_sharpe': ew7['sharpe_ratio'],
                'sharpe_diff': strat['sharpe_ratio'] - ew7['sharpe_ratio'],
                'helix_sortino': strat.get('sortino_ratio', 0),
                'ew7_sortino': ew7['sortino_ratio'],
                'sortino_diff': strat.get('sortino_ratio', 0) - ew7['sortino_ratio'],
                'helix_vol': strat['volatility'],
                'ew7_vol': ew7['annualized_volatility'],
                'vol_diff': strat['volatility'] - ew7['annualized_volatility'],
                'helix_dd': strat['max_drawdown'],
                'ew7_dd': ew7['max_drawdown'],
            })

        if ew7_rows:
            print("\n{:<12} | {:>12} | {:>12} | {:>10} | {:>8} | {:>8} | {:>8}".format(
                "Period", "Helix Ret", "EW7 Ret", "Ret Diff", "Helix SR", "EW7 SR", "SR Diff"
            ))
            print("-" * 90)
            for r in ew7_rows:
                print("{:<12} | {:>11.2%} | {:>11.2%} | {:>+9.2%} | {:>7.2f} | {:>7.2f} | {:>+7.2f}".format(
                    r['period'], r['helix_return'], r['ew7_return'], r['return_diff'],
                    r['helix_sharpe'], r['ew7_sharpe'], r['sharpe_diff']
                ))
            print("\n{:<12} | {:>12} | {:>12} | {:>10}".format("Period", "Helix Max DD", "EW7 Max DD", "DD Diff"))
            print("-" * 55)
            for r in ew7_rows:
                dd_diff = r['helix_dd'] - r['ew7_dd']
                print("{:<12} | {:>11.2%} | {:>11.2%} | {:>+9.2%}".format(
                    r['period'], r['helix_dd'], r['ew7_dd'], dd_diff
                ))
            print("\n{:<12} | {:>14} | {:>14} | {:>12}".format("Period", "Helix Sortino", "EW7 Sortino", "Sortino Diff"))
            print("-" * 60)
            for r in ew7_rows:
                print("{:<12} | {:>13.2f} | {:>13.2f} | {:>+11.2f}".format(
                    r['period'], r['helix_sortino'], r['ew7_sortino'], r['sortino_diff']
                ))

        # SPY vs EW(7) – how much harder is SPY as benchmark?
        print("\n" + "-" * 60)
        print("SPY vs EW(7) – benchmark gap (SPY typically harder to beat)")
        print("-" * 60)
        spy_ew7_rows = []
        for strat in results_summary:
            start_date = strat['start_date']
            end_date = strat['end_date']
            period_name = strat['period']
            try:
                spy_raw = yf.download('SPY', start=start_date, end=end_date, auto_adjust=True, progress=False)
                spy_data = spy_raw['Close'].squeeze()
                spy_ret = float((spy_data.iloc[-1] / spy_data.iloc[0]) - 1)
            except Exception:
                continue
            ew7 = compute_ew7_benchmark(start_date, end_date)
            if ew7 is None:
                continue
            spy_ew7_rows.append({
                'period': period_name,
                'spy_return': spy_ret,
                'ew7_return': ew7['total_return'],
                'diff': spy_ret - ew7['total_return'],
            })
        if spy_ew7_rows:
            print("{:<12} | {:>12} | {:>12} | {:>10}".format("Period", "SPY Ret", "EW7 Ret", "SPY - EW7"))
            print("-" * 55)
            for r in spy_ew7_rows:
                print("{:<12} | {:>11.2%} | {:>11.2%} | {:>+9.2%}".format(
                    r['period'], r['spy_return'], r['ew7_return'], r['diff']
                ))

    except Exception as e:
        print("Error getting EW(7) benchmark: {}".format(e))

    if export_output is not None and export_periods:
        out_path = Path(export_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump({'periods': export_periods}, f, indent=2)
        print("\nExported {} periods to {}".format(len(export_periods), out_path))

    return results_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Helix 1.1 factor strategy analysis")
    parser.add_argument("--config", "-c", help="Path to tune output JSON (symmetric / primary SJM params)")
    parser.add_argument("--asym-config", "-a", default=None,
                        help="Path to asymmetric hyperparameter JSON. Runs a second backtest per period "
                             "and includes helix_asym + asym_values in the export. "
                             "Default output becomes backtest_data_asymmetric.json when set.")
    parser.add_argument("--target-te", type=float, default=None,
                        help="Target tracking error (e.g. 0.02 for 2%%). Paper: 1-4%%")
    parser.add_argument("--export", action="store_true", help="Export JSON for dashboard (writes to -o path)")
    parser.add_argument("--quick", action="store_true", help="With --export: only 3 periods (faster)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output JSON path for --export. Defaults: --asym-config → backtest_data_asymmetric.json; "
                             "-c → backtest_data_<datetime>.json; otherwise backtest_data.json")
    parser.add_argument("--no-sticky", action="store_true",
                        help="Disable sticky progress bar (progress scrolls with output)")
    args = parser.parse_args()

    export_output = None
    if args.export:
        if args.output is None:
            if args.asym_config:
                export_output = str(PROJECT_ROOT / "dashboard" / "public" / "backtest_data_asymmetric.json")
            elif args.config:
                base = Path(args.config).stem
                m = re.search(r"(\d{8}_\d{6})$", base)
                if m:
                    export_output = str(PROJECT_ROOT / "dashboard" / "public" / "backtest_data_{}.json".format(m.group(1)))
                else:
                    export_output = str(PROJECT_ROOT / "dashboard" / "public" / "backtest_data.json")
            else:
                export_output = str(PROJECT_ROOT / "dashboard" / "public" / "backtest_data.json")
        else:
            export_output = args.output

    results = run_analysis(
        sjm_config_path=args.config,
        asym_sjm_config_path=args.asym_config,
        target_tracking_error=args.target_te,
        export_output=export_output,
        quick_export=args.quick and args.export,
        no_sticky=args.no_sticky,
    )
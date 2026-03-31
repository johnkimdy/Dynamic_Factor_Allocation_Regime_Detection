#!/usr/bin/env python3
"""
Phase 6: SJM hyperparameter tuning via Optuna.
Per paper: λ=50, κ²=9.5. OOS validation with monthly refit.

Config-based: pass --config <path> with a JSON file (see hyperparam/tune_template.json).
Temporal split: train (implicit before validation), validation (tuning OOS), holdout (true OOS).
"""

import json
import logging
import os
import time
import argparse

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Silence noisy loggers before imports
logging.basicConfig(level=logging.WARNING)
for _name in ('helix_factor_strategy', 'data.market_data', 'yfinance', 'urllib3'):
    logging.getLogger(_name).setLevel(logging.WARNING)

import numpy as np
import pandas as pd

from helix_factor_strategy import (
    HelixFactorStrategy,
    compute_active_returns,
    EXPECTED_RETURN_CAP,
    MARKET_ETF,
)

# Re-apply: helix_factor_strategy sets INFO on import
logging.getLogger().setLevel(logging.WARNING)
for _name in ('helix_factor_strategy', 'data.market_data', 'yfinance', 'urllib3', 'optuna'):
    logging.getLogger(_name).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

FACTORS = ['QUAL', 'MTUM', 'USMV', 'VLUE', 'SIZE', 'IWF']

# Defaults (used when config missing fields)
DATA_START = '1993-01-01'
DATA_END = '2024-12-31'
VALIDATION_START = '2012-01-01'
VALIDATION_END = '2022-12-31'
HOLDOUT_START = '2023-01-01'
HOLDOUT_END = '2025-12-31'

PAPER_LAMBDA = 50.0
PAPER_KAPPA_SQ = 9.5

WANDB_PROJECT = 'helix-sjm-tuning'
MLFLOW_EXPERIMENT = 'helix-sjm-tuning'
HYPERPARAM_DIR = 'hyperparam'

LAMBDA_LOW, LAMBDA_HIGH = 5.0, 150.0
KAPPA_LOW, KAPPA_HIGH = 2.0, 18.0

# Asymmetric penalty search space (JOH-12)
LAMBDA_ENTER_LOW, LAMBDA_ENTER_HIGH = 5.0, 200.0
LAMBDA_EXIT_LOW, LAMBDA_EXIT_HIGH = 5.0, 200.0


def long_short_position(expected_active_ann):
    cap = EXPECTED_RETURN_CAP
    if expected_active_ann >= cap:
        return 1.0
    if expected_active_ann <= -cap:
        return -1.0
    return expected_active_ann / cap


def _compute_oos_sharpe(strategy, active_returns, market_returns, factor, lam, k2, test_start, test_end,
                        min_train_days=252):
    """OOS Sharpe: refit monthly with expanding window.
    min_train_days: paper uses 8 years (2016); default 252 (1 year) for backward compat."""
    test_dates = active_returns.loc[test_start:test_end].index
    if len(test_dates) < 60:
        return 0.0
    month_ends = pd.date_range(start=test_start, end=test_end, freq='ME')
    positions = pd.Series(index=test_dates, dtype=float)
    prev_me = pd.Timestamp(test_start) - pd.offsets.MonthEnd(1)
    for me in month_ends:
        train_end = prev_me
        train_ar = active_returns.loc[:train_end]
        train_mr = market_returns.loc[:train_end]
        if len(train_ar) < min_train_days:
            prev_me = me
            continue
        cfg = {factor: {'jump_penalty': lam, 'sparsity_param': k2}}
        strategy.fit_regime_models(train_ar, train_mr, sjm_config=cfg)
        model = strategy.regime_models.get(factor)
        if model is None or model.regimes_ is None:
            prev_me = me
            continue
        last_regime = int(model.regimes_.iloc[-1])
        means = strategy.regime_means.get(factor, {0: 0.0})
        pos = long_short_position(means.get(last_regime, 0.0))
        mask = (test_dates > prev_me) & (test_dates <= me)
        positions.loc[mask] = pos
        prev_me = me
    positions = positions.ffill().fillna(0)
    ar = active_returns.loc[test_start:test_end, factor]
    pos = positions.reindex(ar.index).ffill().fillna(0).shift(2)
    strat_ret = (pos * ar).dropna()
    if len(strat_ret) < 20 or strat_ret.std() < 1e-10:
        return 0.0
    return float(strat_ret.mean() / strat_ret.std() * np.sqrt(252))


def _compute_oos_sharpe_asymmetric(
    strategy, active_returns, market_returns, factor,
    lambda_enter, lambda_exit, kappa_sq, test_start, test_end,
    min_train_days=252,
):
    """OOS Sharpe with asymmetric jump penalty matrix (JOH-12).
    Identical monthly expanding-window loop as _compute_oos_sharpe but uses
    jump_penalty_matrix instead of scalar jump_penalty."""
    L = [[0.0, float(lambda_enter)], [float(lambda_exit), 0.0]]
    test_dates = active_returns.loc[test_start:test_end].index
    if len(test_dates) < 60:
        return 0.0
    month_ends = pd.date_range(start=test_start, end=test_end, freq='ME')
    positions = pd.Series(index=test_dates, dtype=float)
    prev_me = pd.Timestamp(test_start) - pd.offsets.MonthEnd(1)
    for me in month_ends:
        train_end = prev_me
        train_ar = active_returns.loc[:train_end]
        train_mr = market_returns.loc[:train_end]
        if len(train_ar) < min_train_days:
            prev_me = me
            continue
        cfg = {factor: {'jump_penalty_matrix': L, 'sparsity_param': kappa_sq}}
        strategy.fit_regime_models(train_ar, train_mr, sjm_config=cfg)
        model = strategy.regime_models.get(factor)
        if model is None or model.regimes_ is None:
            prev_me = me
            continue
        last_regime = int(model.regimes_.iloc[-1])
        means = strategy.regime_means.get(factor, {0: 0.0})
        pos = long_short_position(means.get(last_regime, 0.0))
        mask = (test_dates > prev_me) & (test_dates <= me)
        positions.loc[mask] = pos
        prev_me = me
    positions = positions.ffill().fillna(0)
    ar = active_returns.loc[test_start:test_end, factor]
    pos = positions.reindex(ar.index).ffill().fillna(0).shift(2)
    strat_ret = (pos * ar).dropna()
    if len(strat_ret) < 20 or strat_ret.std() < 1e-10:
        return 0.0
    return float(strat_ret.mean() / strat_ret.std() * np.sqrt(252))


def tune_factor_asymmetric_optuna(
    strategy,
    active_returns,
    market_returns,
    factor,
    validation_start,
    validation_end,
    kappa_sq=None,
    n_trials=80,
    n_jobs=4,
    lambda_enter_low=LAMBDA_ENTER_LOW,
    lambda_enter_high=LAMBDA_ENTER_HIGH,
    lambda_exit_low=LAMBDA_EXIT_LOW,
    lambda_exit_high=LAMBDA_EXIT_HIGH,
    min_train_days=252,
    use_mlflow=False,
):
    """Optuna study: search (lambda_enter, lambda_exit) for one factor (JOH-12).

    kappa_sq: fixed from prior symmetric best run; defaults to paper value 9.5.
    Runs annually: call once per year with the updated validation window.
    Returns dict with lambda_enter, lambda_exit, kappa_sq, sharpe.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    k2 = float(kappa_sq) if kappa_sq is not None else PAPER_KAPPA_SQ

    def objective(trial):
        le = trial.suggest_float('lambda_enter', lambda_enter_low, lambda_enter_high, log=True)
        lx = trial.suggest_float('lambda_exit', lambda_exit_low, lambda_exit_high, log=True)
        return _compute_oos_sharpe_asymmetric(
            strategy, active_returns, market_returns, factor,
            le, lx, k2, validation_start, validation_end,
            min_train_days=min_train_days,
        )

    callbacks = [_make_trial_logger(factor)]
    if use_mlflow:
        callbacks.append(_make_mlflow_callback(factor))

    study = optuna.create_study(
        direction='maximize',
        study_name='asymmetric_{}'.format(factor),
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, callbacks=callbacks, n_jobs=n_jobs,
                   show_progress_bar=True)
    best = study.best_trial
    return {
        'lambda_enter': best.params['lambda_enter'],
        'lambda_exit': best.params['lambda_exit'],
        'kappa_sq': k2,
        'sharpe': best.value,
        '_study': study,   # caller may want trial data for plotting
    }


def tune_asymmetric_hyperparameters(
    start_date,
    end_date,
    validation_start,
    validation_end,
    holdout_start,
    holdout_end,
    prior_kappa_sq=None,        # dict {factor: kappa_sq} from symmetric best, or None → paper default
    n_trials_per_factor=80,
    n_jobs=4,
    lambda_enter_low=LAMBDA_ENTER_LOW,
    lambda_enter_high=LAMBDA_ENTER_HIGH,
    lambda_exit_low=LAMBDA_EXIT_LOW,
    lambda_exit_high=LAMBDA_EXIT_HIGH,
    min_train_days=252,
    use_mlflow=False,
):
    """Annual asymmetric Λ tuning (JOH-12).

    Phase 2 of the yearly hyperparameter schedule:
      1. Run tune_hyperparameters (symmetric) → get best kappa_sq per factor.
      2. Run tune_asymmetric_hyperparameters with those kappa_sq values fixed →
         find per-factor (lambda_enter, lambda_exit) on the same validation window.
      3. Save results to sjm_hyperparameters_asymmetric_best.json.

    prior_kappa_sq: {factor: float} from symmetric run. If None, uses paper default 9.5.
    Returns (results, study_map) where study_map {factor: optuna.Study} for plotting.
    """
    strategy = HelixFactorStrategy(lookback_days=252 * 10)
    strategy.fetch_data(start_date, end_date)
    returns = strategy.calculate_returns()
    active_returns = compute_active_returns(returns, market_col=MARKET_ETF)
    market_returns = returns[MARKET_ETF]

    results = {}
    study_map = {}
    kappas = prior_kappa_sq or {}

    for factor in tqdm(FACTORS, desc="Asymmetric tuning", unit="factor"):
        k2 = kappas.get(factor, PAPER_KAPPA_SQ)
        print("\n--- Asymmetric Optuna {} ({} trials, kappa_sq={:.2f} fixed) ---".format(
            factor, n_trials_per_factor, k2))
        r = tune_factor_asymmetric_optuna(
            strategy, active_returns, market_returns, factor,
            validation_start, validation_end,
            kappa_sq=k2,
            n_trials=n_trials_per_factor,
            n_jobs=n_jobs,
            lambda_enter_low=lambda_enter_low,
            lambda_enter_high=lambda_enter_high,
            lambda_exit_low=lambda_exit_low,
            lambda_exit_high=lambda_exit_high,
            min_train_days=min_train_days,
            use_mlflow=use_mlflow,
        )
        study_map[factor] = r.pop('_study')
        print("{}: λ_enter={:.1f}, λ_exit={:.1f}, κ²={:.2f}, validation Sharpe={:.3f}".format(
            factor, r['lambda_enter'], r['lambda_exit'], r['kappa_sq'], r['sharpe']))
        results[factor] = r

    # Holdout evaluation with best asymmetric params
    print("\n--- Holdout evaluation ({}-{}) ---".format(holdout_start[:10], holdout_end[:10]))
    for factor in FACTORS:
        r = results[factor]
        holdout_sharpe = _compute_oos_sharpe_asymmetric(
            strategy, active_returns, market_returns, factor,
            r['lambda_enter'], r['lambda_exit'], r['kappa_sq'],
            holdout_start, holdout_end,
            min_train_days=min_train_days,
        )
        r['holdout_sharpe'] = holdout_sharpe
        print("{}: holdout Sharpe={:.3f}".format(factor, holdout_sharpe))

    return results, study_map, strategy, active_returns, market_returns


def _make_mlflow_callback(factor):
    """Custom MLflow callback (no optuna-integration needed). Logs each trial; works with n_jobs>1."""

    def callback(study, trial):
        try:
            import mlflow
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            with mlflow.start_run(run_name=f"{factor}-trial{trial.number}"):
                mlflow.log_params(trial.params)
                mlflow.log_metric("oos_sharpe", trial.value)
                mlflow.set_tag("factor", factor)
        except Exception as e:
            logger.warning("MLflow log failed: %s", e)

    return callback


def _make_trial_logger(factor, max_log=5):
    """Callback to log first few trials for JOH-8 convergence debugging."""
    seen = [0]

    def callback(study, trial):
        if seen[0] < max_log:
            p = trial.params
            v = trial.value
            logger.info(
                "[%s] Trial %d: λ=%.2f κ²=%.2f → OOS Sharpe=%.4f",
                factor, trial.number, p.get('lambda', 0), p.get('kappa_sq', 0), v
            )
            seen[0] += 1
    return callback


def tune_factor_optuna(
    strategy,
    active_returns,
    market_returns,
    factor,
    validation_start,
    validation_end,
    n_trials=100,
    n_jobs=4,
    lambda_low=LAMBDA_LOW,
    lambda_high=LAMBDA_HIGH,
    kappa_low=KAPPA_LOW,
    kappa_high=KAPPA_HIGH,
    min_train_days=252,
    use_wandb=True,
    use_mlflow=False,
):
    """Optuna study for one factor."""

    def objective(trial):
        lam = trial.suggest_float('lambda', lambda_low, lambda_high, log=True)
        k2 = trial.suggest_float('kappa_sq', kappa_low, kappa_high)
        sharpe = _compute_oos_sharpe(
            strategy, active_returns, market_returns, factor, lam, k2,
            validation_start, validation_end,
            min_train_days=min_train_days,
        )
        return sharpe

    callbacks = [_make_trial_logger(factor)]
    if use_wandb:
        try:
            import wandb
            try:
                from optuna.integration.wandb import WeightsAndBiasesCallback
            except ImportError:
                from optuna_integration.wandb import WeightsAndBiasesCallback
            wandb.login()
            callbacks.append(WeightsAndBiasesCallback(
                project=WANDB_PROJECT,
                wandb_kwargs={'tags': ['sjm', 'optuna', factor], 'name': f'sjm-{factor}'},
            ))
        except Exception as e:
            logger.warning("W&B callback failed: %s", e)
    if use_mlflow:
        callbacks.append(_make_mlflow_callback(factor))

    import optuna
    study = optuna.create_study(direction='maximize', study_name=f'sjm_{factor}')
    study.optimize(objective, n_trials=n_trials, callbacks=callbacks, n_jobs=n_jobs, show_progress_bar=True)
    best = study.best_trial
    return {
        'lambda': best.params['lambda'],
        'kappa_sq': best.params['kappa_sq'],
        'sharpe': best.value,
    }


def tune_hyperparameters(
    start_date,
    end_date,
    validation_start,
    validation_end,
    holdout_start,
    holdout_end,
    n_trials_per_factor=100,
    n_jobs=4,
    lambda_low=LAMBDA_LOW,
    lambda_high=LAMBDA_HIGH,
    kappa_low=KAPPA_LOW,
    kappa_high=KAPPA_HIGH,
    min_train_days=252,
    use_wandb=True,
    use_mlflow=False,
):
    """
    Optuna search for λ and κ² per factor on validation period.
    Then compute holdout Sharpe with best params.
    Returns (results, strategy, active_returns, market_returns) for holdout eval.
    """
    strategy = HelixFactorStrategy(lookback_days=252 * 10)
    strategy.fetch_data(start_date, end_date)
    returns = strategy.calculate_returns()
    active_returns = compute_active_returns(returns, market_col=MARKET_ETF)
    market_returns = returns[MARKET_ETF]

    paper_sharpes = {}
    for factor in FACTORS:
        s = _compute_oos_sharpe(
            strategy, active_returns, market_returns, factor,
            PAPER_LAMBDA, PAPER_KAPPA_SQ, validation_start, validation_end,
            min_train_days=min_train_days,
        )
        paper_sharpes[factor] = s
    paper_mean = np.mean(list(paper_sharpes.values()))
    print("Paper baseline (λ=50, κ²=9.5) validation Sharpe: mean={:.3f}".format(paper_mean))
    print("  per factor: {}".format({f: round(s, 3) for f, s in paper_sharpes.items()}))

    results = {}
    for factor in tqdm(FACTORS, desc="Tuning factors", unit="factor"):
        print("\n--- Optuna tuning {} ({} trials, validation {}-{}) ---".format(
            factor, n_trials_per_factor, validation_start[:10], validation_end[:10]))
        results[factor] = tune_factor_optuna(
            strategy,
            active_returns,
            market_returns,
            factor,
            validation_start,
            validation_end,
            n_trials=n_trials_per_factor,
            n_jobs=n_jobs,
            lambda_low=lambda_low,
            lambda_high=lambda_high,
            kappa_low=kappa_low,
            kappa_high=kappa_high,
            min_train_days=min_train_days,
            use_wandb=use_wandb,
            use_mlflow=use_mlflow,
        )
        r = results[factor]
        print("{}: λ={:.1f}, κ²={:.1f}, validation Sharpe={:.3f}".format(
            factor, r['lambda'], r['kappa_sq'], r['sharpe']))

    # Holdout evaluation
    print("\n--- Holdout evaluation ({}-{}) ---".format(holdout_start[:10], holdout_end[:10]))
    for factor in FACTORS:
        r = results[factor]
        holdout_sharpe = _compute_oos_sharpe(
            strategy,
            active_returns,
            market_returns,
            factor,
            r['lambda'],
            r['kappa_sq'],
            holdout_start,
            holdout_end,
            min_train_days=min_train_days,
        )
        r['holdout_sharpe'] = holdout_sharpe
        print("{}: holdout Sharpe={:.3f}".format(factor, holdout_sharpe))

    return results, strategy, active_returns, market_returns


def verify_hyperparameter_sensitivity():
    """
    JOH-8: Assert that different (λ, κ²) produce different SJM regime outputs.
    Uses synthetic data - no network. Run with: python tune_sjm_hyperparameters.py --verify
    """
    import numpy as np
    import pandas as pd
    from helix_factor_strategy import SparseJumpModel

    np.random.seed(42)
    n = 500
    ar = pd.Series(0.001 * np.random.randn(n), index=pd.date_range('2020-01-01', periods=n, freq='B'))

    configs = [(5.0, 2.0), (50.0, 9.5), (150.0, 18.0)]
    regimes_list = []
    for lam, k2 in configs:
        model = SparseJumpModel(n_regimes=2, jump_penalty=lam, sparsity_param=k2) # Explicit JM state (K = 2 regimes) from paper's page 7.
        model.fit(ar)
        regimes_list.append(model.regimes_)

    # At least two configs must differ
    r0, r1, r2 = regimes_list
    diff_01 = (r0 != r1).sum()
    diff_12 = (r1 != r2).sum()
    diff_02 = (r0 != r2).sum()
    if diff_01 == 0 and diff_12 == 0 and diff_02 == 0:
        raise AssertionError(
            "JOH-8: All (λ, κ²) configs produced identical regimes. "
            "Hyperparameters have no effect on SJM output."
        )
    # Sparsity: low κ² should produce fewer non-zero feature weights
    m_lo = SparseJumpModel(jump_penalty=50, sparsity_param=2)
    m_hi = SparseJumpModel(jump_penalty=50, sparsity_param=18)
    m_lo.fit(ar)
    m_hi.fit(ar)
    nz_lo = (np.abs(m_lo.feature_weights_) > 1e-6).sum()
    nz_hi = (np.abs(m_hi.feature_weights_) > 1e-6).sum()
    if nz_lo >= nz_hi:
        raise AssertionError(
            "JOH-8: sparsity_param has no effect: κ²=2 gave {} non-zero weights, "
            "κ²=18 gave {} (expected fewer for lower κ²)".format(nz_lo, nz_hi)
        )
    print("JOH-8 verify OK: regimes and sparsity vary with (λ, κ²)")


def load_config(path):
    """Load and validate config JSON. Returns metadata dict.
    If embargo_months is set, holdout_start = validation_end + embargo_months."""
    with open(path) as f:
        doc = json.load(f)
    meta = doc.get("metadata", doc)
    defaults = {
        "data_start": DATA_START,
        "data_end": DATA_END,
        "validation_start": VALIDATION_START,
        "validation_end": VALIDATION_END,
        "holdout_start": HOLDOUT_START,
        "holdout_end": HOLDOUT_END,
        "embargo_months": 0,
        "min_train_days": 252,
        "n_trials_per_factor": 100,
        "n_jobs": 4,
        "lambda_low": LAMBDA_LOW,
        "lambda_high": LAMBDA_HIGH,
        "kappa_low": KAPPA_LOW,
        "kappa_high": KAPPA_HIGH,
    }
    for k, v in defaults.items():
        if k not in meta or meta[k] is None:
            meta[k] = v
    embargo = int(meta.get("embargo_months", 0))
    if embargo > 0:
        val_end = pd.Timestamp(meta["validation_end"])
        holdout_start = (val_end + pd.DateOffset(months=embargo) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        meta["holdout_start"] = holdout_start
    return meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SJM hyperparameter tuning with Optuna. Pass --config (copy hyperparam/tune_template.json).'
    )
    parser.add_argument('--config', required=True, help='Path to config JSON (see hyperparam/tune_template.json)')
    parser.add_argument('--mode', choices=['symmetric', 'asymmetric'], default='symmetric',
                        help='symmetric: tune (lambda, kappa); asymmetric: tune (lambda_enter, lambda_exit) '
                             'with kappa fixed from prior best. Run symmetric first, then asymmetric annually.')
    parser.add_argument('--n-trials', type=int, default=None, help='Override n_trials_per_factor from config')
    parser.add_argument('--n-jobs', type=int, default=None, help='Override n_jobs from config')
    parser.add_argument('--no-wandb', action='store_true', help='Disable W&B logging')
    parser.add_argument('--mlflow', action='store_true', help='Log to MLflow (works with n_jobs>1; saves to ./mlruns)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable INFO logging')
    parser.add_argument('--verify', action='store_true', help='Run JOH-8 sensitivity check (no network)')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
        for _n in ('helix_factor_strategy', 'data.market_data', 'yfinance', 'urllib3', 'optuna'):
            logging.getLogger(_n).setLevel(logging.INFO)

    if args.verify:
        verify_hyperparameter_sensitivity()
        print("Done.")
        exit(0)

    meta = load_config(args.config)
    run_date = time.strftime('%Y-%m-%dT%H:%M:%S')
    meta["run_date"] = run_date

    if args.mode == 'asymmetric':
        n_trials = args.n_trials if args.n_trials is not None else int(meta.get("n_trials_asymmetric", meta["n_trials_per_factor"]))
    else:
        n_trials = args.n_trials if args.n_trials is not None else int(meta["n_trials_per_factor"])
    n_jobs = args.n_jobs if args.n_jobs is not None else int(meta["n_jobs"])

    use_wb = not args.no_wandb and os.environ.get('WANDB_MODE') != 'disabled'
    print("SJM Optuna tuning: config={}".format(args.config))
    print("  n_trials={}, n_jobs={}, W&B={}, MLflow={}".format(n_trials, n_jobs, use_wb, args.mlflow))
    print("  data: {} to {}".format(meta["data_start"], meta["data_end"]))
    print("  validation (tuning OOS): {} to {}".format(meta["validation_start"], meta["validation_end"]))
    print("  holdout (true OOS): {} to {}".format(meta["holdout_start"], meta["holdout_end"]))
    min_train = int(meta.get("min_train_days", 252))
    print("  min_train_days: {} ({:.1f} years)".format(min_train, min_train / 252.0))

    t0 = time.perf_counter()

    if args.mode == 'asymmetric':
        # Phase 2: load prior symmetric best kappa_sq per factor, then tune lambda_enter/lambda_exit
        from hyperparam_io import load_best, save_asymmetric_run_with_doc
        _, best_doc = load_best()
        prior_kappas = {}
        if best_doc and 'results' in best_doc:
            for f, r in best_doc['results'].items():
                prior_kappas[f] = float(r.get('kappa_sq', PAPER_KAPPA_SQ))
        if prior_kappas:
            print("  Using prior kappa_sq from best symmetric run: {}".format(
                {f: round(v, 2) for f, v in prior_kappas.items()}))
        else:
            print("  No symmetric best found; using paper kappa_sq={} for all factors.".format(PAPER_KAPPA_SQ))

        asym_results, study_map, _, _, _ = tune_asymmetric_hyperparameters(
            start_date=meta["data_start"],
            end_date=meta["data_end"],
            validation_start=meta["validation_start"],
            validation_end=meta["validation_end"],
            holdout_start=meta["holdout_start"],
            holdout_end=meta["holdout_end"],
            prior_kappa_sq=prior_kappas or None,
            n_trials_per_factor=n_trials,
            n_jobs=n_jobs,
            lambda_enter_low=float(meta.get("lambda_enter_low", LAMBDA_ENTER_LOW)),
            lambda_enter_high=float(meta.get("lambda_enter_high", LAMBDA_ENTER_HIGH)),
            lambda_exit_low=float(meta.get("lambda_exit_low", LAMBDA_EXIT_LOW)),
            lambda_exit_high=float(meta.get("lambda_exit_high", LAMBDA_EXIT_HIGH)),
            min_train_days=int(meta.get("min_train_days", 252)),
            use_mlflow=args.mlflow,
        )
        elapsed = time.perf_counter() - t0
        meta["elapsed_sec"] = round(elapsed, 1)
        meta["mode"] = "asymmetric"

        validation_sharpes = [r["sharpe"] for r in asym_results.values()]
        holdout_sharpes = [r["holdout_sharpe"] for r in asym_results.values()]
        mean_validation_sharpe = np.mean(validation_sharpes)
        mean_holdout_sharpe = np.mean(holdout_sharpes)

        print("\nTotal time: {:.1f}s ({:.1f} min)".format(elapsed, elapsed / 60))
        print("\nBest asymmetric penalties (validation Sharpe):")
        for f, r in asym_results.items():
            print("  {}: λ_enter={:.1f}, λ_exit={:.1f}, κ²={:.2f}, "
                  "validation={:.3f}, holdout={:.3f}".format(
                      f, r['lambda_enter'], r['lambda_exit'], r['kappa_sq'],
                      r['sharpe'], r['holdout_sharpe']))
        print("\nMean validation Sharpe: {:.4f}".format(mean_validation_sharpe))
        print("Mean holdout Sharpe: {:.4f}".format(mean_holdout_sharpe))

        out_doc = {
            "metadata": meta,
            "mean_oos_sharpe": mean_validation_sharpe,
            "mean_holdout_sharpe": mean_holdout_sharpe,
            "results": {f: {k: v for k, v in r.items()} for f, r in asym_results.items()},
        }
        out_path, factors_updated, _ = save_asymmetric_run_with_doc(out_doc)
        print("\nSaved to {}".format(out_path))
        if factors_updated:
            print("*** ASYMMETRIC BEST UPDATED *** factors improved: {}".format(", ".join(factors_updated)))
        else:
            print("Run saved to backlog. No factor beat its current asymmetric best.")
        exit(0)

    results, _, _, _ = tune_hyperparameters(
        start_date=meta["data_start"],
        end_date=meta["data_end"],
        validation_start=meta["validation_start"],
        validation_end=meta["validation_end"],
        holdout_start=meta["holdout_start"],
        holdout_end=meta["holdout_end"],
        n_trials_per_factor=n_trials,
        n_jobs=n_jobs,
        lambda_low=float(meta["lambda_low"]),
        lambda_high=float(meta["lambda_high"]),
        kappa_low=float(meta["kappa_low"]),
        kappa_high=float(meta["kappa_high"]),
        min_train_days=int(meta.get("min_train_days", 252)),
        use_wandb=use_wb,
        use_mlflow=args.mlflow,
    )
    elapsed = time.perf_counter() - t0
    meta["elapsed_sec"] = round(elapsed, 1)

    validation_sharpes = [r["sharpe"] for r in results.values()]
    holdout_sharpes = [r["holdout_sharpe"] for r in results.values()]
    mean_validation_sharpe = np.mean(validation_sharpes)
    mean_holdout_sharpe = np.mean(holdout_sharpes)

    print("\nTotal time: {:.1f}s ({:.1f} min)".format(elapsed, elapsed / 60))
    print("\nBest hyperparameters (validation Sharpe):")
    for f, r in results.items():
        print("  {}: λ={:.1f}, κ²={:.1f}, validation={:.3f}, holdout={:.3f}".format(
            f, r['lambda'], r['kappa_sq'], r['sharpe'], r['holdout_sharpe']))
    print("\nMean validation Sharpe: {:.4f}".format(mean_validation_sharpe))
    print("Mean holdout Sharpe: {:.4f}".format(mean_holdout_sharpe))

    # Build output doc: input metadata + run_date + results
    out_doc = {
        "metadata": meta,
        "mean_oos_sharpe": mean_validation_sharpe,
        "mean_holdout_sharpe": mean_holdout_sharpe,
        "results": {f: {k: v for k, v in r.items()} for f, r in results.items()},
    }

    from hyperparam_io import save_run_with_doc
    out_path, factors_updated, _ = save_run_with_doc(out_doc)

    print("\nSaved to {}".format(out_path))
    if factors_updated:
        print("*** PER-FACTOR BEST UPDATED *** factors improved: {}".format(", ".join(factors_updated)))
    else:
        print("Run saved to backlog. No factor beat its current best.")

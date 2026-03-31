#!/usr/bin/env python3
"""
Stream training + optional allocation for dashboard "Train your own SJM" (JOH-11).

Prints machine-readable lines to stdout for live streaming:
  LOG\t<message>   -- log line
  LOSS\t<factor>\t<iter>\t<value>  -- SJM objective per iteration
  DONE\t<summary>  -- run finished
  ERR\t<message>   -- error

With --mlflow: logs a single MLflow run (experiment "helix-sjm-train") with params and
per-factor metrics objective_QUAL, objective_MTUM, ... so the loss curve is viewable in MLflow UI.

Usage (from repo root):
  python scripts/run_training_stream.py --train-start 2015-01-01 --train-end 2023-12-31 [--oos-start 2024-01-01 --oos-end 2024-12-31] [--lambda 50 --kappa2 9.5] [--config hyperparam/paper_aligned.json] [--mlflow]
"""

import argparse
import json
import os
import sys
from datetime import datetime

if os.path.dirname(os.path.abspath(__file__)) != os.getcwd():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Stream handler that prints LOG\t so API can forward
def _stream_log(msg):
    print("LOG\t" + msg, flush=True)


def main():
    ap = argparse.ArgumentParser(description="SJM training + allocation stream (JOH-11)")
    ap.add_argument("--train-start", required=True, help="Training start date YYYY-MM-DD")
    ap.add_argument("--train-end", required=True, help="Training end date YYYY-MM-DD")
    ap.add_argument("--oos-start", default=None, help="Out-of-sample start (optional)")
    ap.add_argument("--oos-end", default=None, help="Out-of-sample end (optional)")
    ap.add_argument("--lambda", dest="jump_penalty", type=float, default=50.0, help="SJM jump penalty (default 50)")
    ap.add_argument("--kappa2", dest="sparsity_param", type=float, default=9.5, help="SJM kappa^2 (default 9.5)")
    ap.add_argument("--config", default=None, help="Optional JSON path for per-factor overrides")
    ap.add_argument("--mlflow", action="store_true", help="Log run and loss curves to MLflow (./mlruns)")
    args = ap.parse_args()

    try:
        from helix_factor_strategy import (
            HelixFactorStrategy,
            compute_active_returns,
            MARKET_ETF,
        )
    except ImportError as e:
        print("ERR\tImport failed: " + str(e), flush=True)
        sys.exit(1)

    # Optional per-factor config (does not call analyze_strategy.py; uses HelixFactorStrategy only).
    # Accept: "results" (tune output: lambda, kappa_sq per factor), "factors", or flat jump_penalty/sparsity_param.
    sjm_config = None
    if args.config and os.path.isfile(args.config):
        try:
            with open(args.config) as f:
                data = json.load(f)
            if "results" in data:
                # Tune-output format: results.QUAL.lambda, results.QUAL.kappa_sq
                sjm_config = {
                    f: {
                        "jump_penalty": float(data["results"].get(f, {}).get("lambda", args.jump_penalty)),
                        "sparsity_param": float(data["results"].get(f, {}).get("kappa_sq", args.sparsity_param)),
                    }
                    for f in ("QUAL", "MTUM", "USMV", "VLUE", "SIZE", "IWF")
                }
            elif "factors" in data:
                sjm_config = data["factors"]
            else:
                sjm_config = {
                    f: {
                        "jump_penalty": data.get("jump_penalty", args.jump_penalty),
                        "sparsity_param": data.get("sparsity_param", args.sparsity_param),
                    }
                    for f in ("QUAL", "MTUM", "USMV", "VLUE", "SIZE", "IWF")
                }
            if sjm_config:
                _stream_log("Loaded config from " + args.config)
        except Exception as e:
            _stream_log("Config load failed, using defaults: " + str(e))
    if sjm_config is None:
        sjm_config = {
            f: {"jump_penalty": args.jump_penalty, "sparsity_param": args.sparsity_param}
            for f in ("QUAL", "MTUM", "USMV", "VLUE", "SIZE", "IWF")
        }

    strategy = HelixFactorStrategy(
        lookback_days=252 * 12,
        record_sjm_loss_curve=True,
    )

    mlflow_run = None
    if args.mlflow:
        try:
            import mlflow
            mlflow.set_experiment("helix-sjm-train")
            mlflow_run = mlflow.start_run(run_name="train-{}".format(datetime.now().strftime("%Y%m%d_%H%M%S")))
            mlflow.log_params({
                "train_start": args.train_start,
                "train_end": args.train_end,
                "oos_start": args.oos_start or "",
                "oos_end": args.oos_end or "",
                "jump_penalty": args.jump_penalty,
                "sparsity_param": args.sparsity_param,
                "config": args.config or "",
            })
            _stream_log("MLflow run started: {} (experiment helix-sjm-train)".format(mlflow_run.info.run_id))
        except Exception as e:
            _stream_log("MLflow start failed (continuing without MLflow): " + str(e))
            mlflow_run = None

    def on_iter(factor, iteration, objective):
        print("LOSS\t{}\t{}\t{}".format(factor, iteration, objective), flush=True)
        if mlflow_run is not None:
            try:
                import mlflow
                mlflow.log_metric("objective_{}".format(factor), float(objective), step=int(iteration))
            except Exception:
                pass

    strategy.on_sjm_iteration = on_iter

    _stream_log("Fetching data {} to {}".format(args.train_start, args.train_end))
    try:
        strategy.fetch_data(args.train_start, args.train_end)
    except Exception as e:
        print("ERR\tFetch failed: " + str(e), flush=True)
        sys.exit(1)

    returns = strategy.calculate_returns()
    active_returns = compute_active_returns(returns, market_col=MARKET_ETF)
    market_returns = returns[MARKET_ETF]

    _stream_log("Fitting 6 SJMs (live loss stream above)...")
    strategy.fit_regime_models(active_returns, market_returns, sjm_config=sjm_config)

    _stream_log("Training complete.")

    if args.oos_start and args.oos_end:
        _stream_log("OOS allocation {} to {}".format(args.oos_start, args.oos_end))
        try:
            strategy.fetch_data(args.oos_start, args.oos_end)
            oos_returns = strategy.calculate_returns()
            oos_active = compute_active_returns(oos_returns, market_col=MARKET_ETF)
            # Use last known regimes from training (or re-infer on OOS lookback)
            current_regimes = {}
            for factor in strategy.regime_models:
                model = strategy.regime_models[factor]
                if hasattr(model, "regimes_") and model.regimes_ is not None and len(model.regimes_):
                    current_regimes[factor] = int(model.regimes_.iloc[-1])
                else:
                    current_regimes[factor] = 0
            expected = strategy.generate_expected_returns(current_regimes)
            recent = oos_returns.iloc[-252:] if len(oos_returns) >= 252 else oos_returns
            weights = strategy.optimize_portfolio(recent, expected)
            if weights is not None:
                _stream_log("Allocation: " + ", ".join("{}={:.1%}".format(a, w) for a, w in weights.items()))
        except Exception as e:
            print("ERR\tOOS allocation failed: " + str(e), flush=True)

    if args.mlflow and mlflow_run is not None:
        try:
            import mlflow
            mlflow.end_run()
            _stream_log("MLflow run ended. View: mlflow ui (in repo root, open experiment helix-sjm-train)")
        except Exception as e:
            _stream_log("MLflow end_run failed: " + str(e))

    print("DONE\tTraining and (optional) allocation finished.", flush=True)


if __name__ == "__main__":
    main()

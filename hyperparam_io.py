"""
Hyperparameter run storage and best selection.

Stores each tune run as JSON with full metadata (timeframe, n_trials, search space).
Compares new runs to current best by mean OOS Sharpe; promotes if better.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

HYPERPARAM_DIR = "hyperparam"
BEST_POINTER = "best"  # filename: contains name of best run file
BEST_MERGED_FILE = "sjm_hyperparameters_best.json"  # per-factor merged best
RUN_PREFIX = "sjm_hyperparameters_"

# Asymmetric penalty store (JOH-12)
ASYM_RUN_PREFIX = "sjm_asym_"
ASYM_BEST_MERGED_FILE = "sjm_hyperparameters_asymmetric_best.json"

FACTORS = ["QUAL", "MTUM", "USMV", "VLUE", "SIZE", "IWF"]


def _ensure_dir():
    Path(HYPERPARAM_DIR).mkdir(parents=True, exist_ok=True)
    return HYPERPARAM_DIR


def _run_path(run_label):
    return os.path.join(HYPERPARAM_DIR, "{}{}.json".format(RUN_PREFIX, run_label))


def _results_to_sjm_config(results):
    """Convert tune results dict to sjm_config {factor: {jump_penalty|jump_penalty_matrix, sparsity_param}}.
    If results contain lambda_enter/lambda_exit (asymmetric run), builds jump_penalty_matrix.
    Otherwise uses scalar jump_penalty."""
    if not results:
        return {}
    cfg = {}
    for factor, r in results.items():
        entry = {"sparsity_param": float(r.get("kappa_sq", 9.5))}
        if "lambda_enter" in r and "lambda_exit" in r:
            entry["jump_penalty_matrix"] = [
                [0.0, float(r["lambda_enter"])],
                [float(r["lambda_exit"]), 0.0],
            ]
        else:
            entry["jump_penalty"] = float(r.get("lambda", 50.0))
        cfg[factor] = entry
    return cfg


def _merge_per_factor_best(doc, run_filename):
    """
    Merge new run into per-factor best. For each factor, if new validation sharpe
    beats current best for that factor, update. Writes sjm_hyperparameters_best.json.
    Returns (factors_updated, list of factor names that were improved).
    """
    _ensure_dir()
    best_path = os.path.join(HYPERPARAM_DIR, BEST_MERGED_FILE)
    new_results = doc.get("results", {})
    run_meta = doc.get("metadata", {})

    if os.path.exists(best_path):
        with open(best_path) as f:
            best_doc = json.load(f)
        best_results = best_doc.get("results", {})
    else:
        best_doc = {
            "metadata": {"note": "Per-factor best (merged from runs)", "last_updated": run_meta.get("run_date", "")},
            "results": {},
        }
        best_results = {}

    factors_updated = []
    for factor in FACTORS:
        r = new_results.get(factor)
        if not r or "sharpe" not in r:
            continue
        new_sharpe = float(r["sharpe"])
        cur = best_results.get(factor)
        cur_sharpe = float(cur["sharpe"]) if cur and "sharpe" in cur else -1e9
        if new_sharpe > cur_sharpe:
            best_results[factor] = dict(r)
            best_results[factor]["source_run"] = run_filename
            factors_updated.append(factor)

    if factors_updated:
        best_doc["results"] = best_results
        best_doc["metadata"]["last_updated"] = run_meta.get("run_date", "")
        best_doc["metadata"]["last_merge_from"] = run_filename
        best_doc["mean_oos_sharpe"] = sum(
            float(best_results[f]["sharpe"]) for f in FACTORS if f in best_results
        ) / max(1, sum(1 for f in FACTORS if f in best_results))
        with open(best_path, "w") as f:
            json.dump(best_doc, f, indent=2)

    return factors_updated


def save_run_with_doc(doc):
    """
    Save a full run doc to hyperparam/sjm_hyperparameters_YYYYMMDD_HHMMSS.json.
    Per-factor merge: for each factor where this run's validation sharpe beats
    current best for that factor, update sjm_hyperparameters_best.json.
    doc must have: metadata, mean_oos_sharpe, results.
    Returns (path, factors_updated, prev_best_factors).
    """
    _ensure_dir()
    run_label = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = _run_path(run_label)
    run_filename = os.path.basename(path)

    with open(path, "w") as f:
        json.dump(doc, f, indent=2)

    factors_updated = _merge_per_factor_best(doc, run_filename)
    prev_best_factors = []  # for backward compat in return
    return path, factors_updated, prev_best_factors


def save_run(
    results,
    metadata,
):
    """
    Save a tune run to hyperparam/sjm_hyperparameters_YYYYMMDD_HHMMSS.json.
    Compares mean OOS Sharpe to current best; updates best pointer if better.
    Always saves the run (backlog).
    Returns (path, is_new_best, mean_sharpe, prev_best_sharpe).
    """
    _ensure_dir()
    run_label = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = _run_path(run_label)

    sharpes = [r["sharpe"] for r in results.values()]
    mean_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0.0

    doc = {
        "metadata": dict(metadata),
        "mean_oos_sharpe": mean_sharpe,
        "results": {f: dict(r) for f, r in results.items()},
    }

    with open(path, "w") as f:
        json.dump(doc, f, indent=2)

    best_path = os.path.join(HYPERPARAM_DIR, BEST_POINTER)
    is_new_best = False
    prev_best_sharpe = None

    if os.path.exists(best_path):
        with open(best_path) as f:
            prev_best_name = f.read().strip()
        prev_best_file = os.path.join(HYPERPARAM_DIR, prev_best_name)
        if os.path.exists(prev_best_file):
            try:
                with open(prev_best_file) as f:
                    prev = json.load(f)
                prev_best_sharpe = prev.get("mean_oos_sharpe", 0.0)
                if mean_sharpe > prev_best_sharpe:
                    is_new_best = True
                    with open(best_path, "w") as f:
                        f.write(os.path.basename(path))
            except Exception as e:
                logger.warning("Could not read previous best: %s", e)
        else:
            is_new_best = True
            with open(best_path, "w") as f:
                f.write(os.path.basename(path))
    else:
        is_new_best = True
        with open(best_path, "w") as f:
            f.write(os.path.basename(path))

    return path, is_new_best, mean_sharpe, prev_best_sharpe


def load_best():
    """Load sjm_config from per-factor best (sjm_hyperparameters_best.json) or pointer. Returns (cfg, metadata) or (None, None)."""
    merged_path = os.path.join(HYPERPARAM_DIR, BEST_MERGED_FILE)
    if os.path.exists(merged_path):
        return load_run(merged_path)
    best_path = os.path.join(HYPERPARAM_DIR, BEST_POINTER)
    if not os.path.exists(best_path):
        return None, None
    with open(best_path) as f:
        best_name = f.read().strip()
    full_path = os.path.join(HYPERPARAM_DIR, best_name)
    if not os.path.exists(full_path):
        return None, None
    return load_run(full_path)


def load_latest():
    """Load sjm_config from the most recent run (by filename timestamp)."""
    import glob

    pattern = os.path.join(HYPERPARAM_DIR, "{}*.json".format(RUN_PREFIX))
    candidates = glob.glob(pattern)
    if not candidates:
        return None, None
    candidates.sort(reverse=True)
    return load_run(candidates[0])


def load_run(path):
    """Load a specific run file. Returns (sjm_config, full_doc) or (None, None)."""
    if not os.path.exists(path):
        return None, None
    try:
        with open(path) as f:
            doc = json.load(f)
        results = doc.get("results", {})
        cfg = _results_to_sjm_config(results)
        return cfg, doc
    except Exception as e:
        logger.warning("Could not load %s: %s", path, e)
        return None, None


def load_sjm_config(use_best=True):
    """
    Load sjm_config for backtest/analysis.
    use_best=True: load from best pointer, else latest JSON.
    use_best=False: load latest run by filename.
    Falls back to legacy CSV if no JSON found.
    """
    if use_best:
        cfg, _ = load_best()
        if cfg:
            return cfg
    cfg, _ = load_latest()
    if cfg:
        return cfg
    return _load_legacy_csv()


def _load_legacy_csv():
    """Fallback: load from legacy CSV (hyperparam/sjm_hyperparameters_*.csv)."""
    import glob

    pattern = os.path.join(HYPERPARAM_DIR, "{}*.csv".format(RUN_PREFIX))
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    candidates.sort(reverse=True)
    path = candidates[0]
    try:
        import pandas as pd

        df = pd.read_csv(path, index_col=0)
        cfg = {}
        for factor in df.index:
            cfg[factor] = {
                "jump_penalty": float(df.loc[factor, "lambda"]),
                "sparsity_param": float(df.loc[factor, "kappa_sq"]),
            }
        return cfg
    except Exception as e:
        logger.warning("Could not load legacy CSV %s: %s", path, e)
        return None


def _merge_asymmetric_per_factor_best(doc, run_filename):
    """Merge asymmetric run into sjm_hyperparameters_asymmetric_best.json.
    Promotes per-factor if new validation sharpe beats current best for that factor."""
    _ensure_dir()
    best_path = os.path.join(HYPERPARAM_DIR, ASYM_BEST_MERGED_FILE)
    new_results = doc.get("results", {})
    run_meta = doc.get("metadata", {})

    if os.path.exists(best_path):
        with open(best_path) as f:
            best_doc = json.load(f)
        best_results = best_doc.get("results", {})
    else:
        best_doc = {
            "metadata": {"note": "Per-factor asymmetric best (JOH-12)", "last_updated": run_meta.get("run_date", "")},
            "results": {},
        }
        best_results = {}

    factors_updated = []
    for factor in FACTORS:
        r = new_results.get(factor)
        if not r or "sharpe" not in r:
            continue
        new_sharpe = float(r["sharpe"])
        cur = best_results.get(factor)
        cur_sharpe = float(cur["sharpe"]) if cur and "sharpe" in cur else -1e9
        if new_sharpe > cur_sharpe:
            best_results[factor] = dict(r)
            best_results[factor]["source_run"] = run_filename
            factors_updated.append(factor)

    if factors_updated:
        best_doc["results"] = best_results
        best_doc["metadata"]["last_updated"] = run_meta.get("run_date", "")
        best_doc["metadata"]["last_merge_from"] = run_filename
        best_doc["mean_oos_sharpe"] = sum(
            float(best_results[f]["sharpe"]) for f in FACTORS if f in best_results
        ) / max(1, sum(1 for f in FACTORS if f in best_results))
        with open(best_path, "w") as f:
            json.dump(best_doc, f, indent=2)

    return factors_updated


def save_asymmetric_run_with_doc(doc):
    """Save asymmetric tuning run to hyperparam/sjm_asym_YYYYMMDD_HHMMSS.json.
    Merges per-factor bests into sjm_hyperparameters_asymmetric_best.json.
    Returns (path, factors_updated, [])."""
    _ensure_dir()
    from datetime import datetime
    run_label = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(HYPERPARAM_DIR, "{}{}.json".format(ASYM_RUN_PREFIX, run_label))
    run_filename = os.path.basename(path)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    factors_updated = _merge_asymmetric_per_factor_best(doc, run_filename)
    return path, factors_updated, []


def load_asymmetric_best():
    """Load asymmetric best from sjm_hyperparameters_asymmetric_best.json.
    Returns (sjm_config, full_doc) or (None, None)."""
    path = os.path.join(HYPERPARAM_DIR, ASYM_BEST_MERGED_FILE)
    return load_run(path)


def load_results_for_store():
    """
    Load full results (for run_sjm_pipeline step_store) from best, then latest, then legacy CSV.
    Returns dict {factor: {lambda, kappa_sq, sharpe}} or None.
    """
    cfg, doc = load_best()
    if doc and "results" in doc:
        return doc["results"]
    cfg, doc = load_latest()
    if doc and "results" in doc:
        return doc["results"]
    # Legacy CSV
    import glob
    pattern = os.path.join(HYPERPARAM_DIR, "{}*.csv".format(RUN_PREFIX))
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    candidates.sort(reverse=True)
    try:
        import pandas as pd
        df = pd.read_csv(candidates[0], index_col=0)
        return {
            f: {"lambda": float(df.loc[f, "lambda"]), "kappa_sq": float(df.loc[f, "kappa_sq"]),
                "sharpe": float(df.loc[f, "sharpe"]) if "sharpe" in df.columns else 0.0}
            for f in df.index
        }
    except Exception as e:
        logger.warning("Could not load legacy CSV: %s", e)
        return None

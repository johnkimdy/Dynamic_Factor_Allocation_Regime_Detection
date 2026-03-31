#!/usr/bin/env python3
"""
SJM Hyperparameter Parameter Store (JOH-9)

Persists (λ, κ²) per factor with versioning. Supports:
- Local YAML config (offline/CI, rollback)
- W&B Artifacts (cloud versioning, production)
- Paper defaults fallback
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Artifact naming
WANDB_ARTIFACT_NAME = "sjm-hyperparameters"
WANDB_ARTIFACT_TYPE = "params"
WANDB_PROJECT = "helix-sjm-tuning"

# Default config dir (relative to project root)
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent / "config"
PRODUCTION_POINTER = "production"
LATEST_POINTER = "latest"

# Paper defaults (fallback)
PAPER_DEFAULTS = {"jump_penalty": 50.0, "sparsity_param": 9.5}


def _results_to_sjm_config(results):
    """Convert tune_sjm_hyperparameters results dict to sjm_config format."""
    if not results:
        return {}
    cfg = {}
    for factor, r in results.items():
        cfg[factor] = {
            "jump_penalty": float(r.get("lambda", PAPER_DEFAULTS["jump_penalty"])),
            "sparsity_param": float(r.get("kappa_sq", PAPER_DEFAULTS["sparsity_param"])),
        }
    return cfg


def _sjm_config_to_results(cfg):
    """Convert sjm_config to tune results format (for serialization)."""
    results = {}
    for factor, p in cfg.items():
        results[factor] = {
            "lambda": p.get("jump_penalty", PAPER_DEFAULTS["jump_penalty"]),
            "kappa_sq": p.get("sparsity_param", PAPER_DEFAULTS["sparsity_param"]),
            "sharpe": p.get("sharpe"),
        }
    return results


def _ensure_config_dir():
    Path(DEFAULT_CONFIG_DIR).mkdir(parents=True, exist_ok=True)
    return DEFAULT_CONFIG_DIR


def _version_path(version):
    return DEFAULT_CONFIG_DIR / f"sjm_params_v{version}.json"


def _pointer_path(alias):
    return DEFAULT_CONFIG_DIR / f"sjm_params_{alias}.json"


def save_local(results, metadata=None):
    """
    Save params to local JSON. Creates new version and updates 'latest' pointer.
    Returns version string (e.g. '20240311_143022').
    """
    cfg = _results_to_sjm_config(results)
    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    doc = {
        "version": version,
        "metadata": metadata or {},
        "tuning_date": datetime.utcnow().isoformat() + "Z",
        "params": cfg,
        "results": _sjm_config_to_results(cfg),
    }
    # Add sharpe from results if present
    for f, r in results.items():
        if "sharpe" in r and f in doc["params"]:
            doc["params"][f]["sharpe"] = r["sharpe"]

    _ensure_config_dir()
    path = _version_path(version)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    # Update latest pointer
    latest_path = _pointer_path(LATEST_POINTER)
    with open(latest_path, "w") as f:
        json.dump({"version": version, "path": str(path)}, f, indent=2)
    logger.info("Saved params to %s (version=%s)", path, version)
    return version


def load_local(version_or_alias=None):
    """
    Load params from local store.
    version_or_alias: 'latest', 'production', or version string (e.g. '20240311_143022').
    Returns (sjm_config dict, metadata dict) or (paper defaults, {}) if not found.
    """
    _ensure_config_dir()
    if version_or_alias is None:
        version_or_alias = LATEST_POINTER

    if version_or_alias in (LATEST_POINTER, PRODUCTION_POINTER):
        ptr = _pointer_path(version_or_alias)
        if not ptr.exists():
            logger.warning("Pointer %s not found, using paper defaults", version_or_alias)
            return _paper_default_config(), {}
        with open(ptr) as f:
            data = json.load(f)
        version = data.get("version")
        path = data.get("path")
        if path and Path(path).exists():
            with open(path) as fp:
                doc = json.load(fp)
        else:
            path = _version_path(version)
            if not path.exists():
                logger.warning("Version %s not found, using paper defaults", version)
                return _paper_default_config(), {}
            with open(path) as fp:
                doc = json.load(fp)
    else:
        path = _version_path(version_or_alias)
        if not path.exists():
            logger.warning("Version %s not found, using paper defaults", version_or_alias)
            return _paper_default_config(), {}
        with open(path) as f:
            doc = json.load(f)

    params = doc.get("params", {})
    cfg = {}
    for factor, p in params.items():
        cfg[factor] = {
            "jump_penalty": float(p.get("jump_penalty", PAPER_DEFAULTS["jump_penalty"])),
            "sparsity_param": float(p.get("sparsity_param", PAPER_DEFAULTS["sparsity_param"])),
        }
    return cfg, doc.get("metadata", {})


FACTORS = ["QUAL", "MTUM", "USMV", "VLUE", "SIZE", "IWF"]


def _paper_default_config():
    """Return sjm_config with paper defaults for all 6 factors."""
    return {f: dict(PAPER_DEFAULTS) for f in FACTORS}


def promote_local(version):
    """Mark a version as production (rollback target)."""
    path = _version_path(version)
    if not path.exists():
        raise FileNotFoundError("Version not found: %s" % version)
    ptr = _pointer_path(PRODUCTION_POINTER)
    _ensure_config_dir()
    with open(ptr, "w") as f:
        json.dump({"version": version, "path": str(path)}, f, indent=2)
    logger.info("Promoted version %s to production", version)


def save_wandb(results, metadata=None, project=None):
    """
    Save params to W&B Artifact. Requires wandb and login.
    Returns artifact version (e.g. 'v0', 'v1').
    """
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed, skipping W&B save")
        return None
    if os.environ.get("WANDB_MODE") == "disabled":
        logger.info("WANDB_MODE=disabled, skipping W&B save")
        return None

    cfg = _results_to_sjm_config(results)
    doc = {
        "version": datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
        "metadata": metadata or {},
        "tuning_date": datetime.utcnow().isoformat() + "Z",
        "params": cfg,
        "results": _sjm_config_to_results(cfg),
    }
    for f, r in results.items():
        if "sharpe" in r and f in doc["params"]:
            doc["params"][f]["sharpe"] = r["sharpe"]

    proj = project or WANDB_PROJECT
    with wandb.init(project=proj, job_type="params_store") as run:
        artifact = wandb.Artifact(
            name=WANDB_ARTIFACT_NAME,
            type=WANDB_ARTIFACT_TYPE,
            metadata={
                "tuning_date": doc["tuning_date"],
                "version": doc["version"],
                **(metadata or {}),
            },
        )
        # Write to temp file and add to artifact
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
            json.dump(doc, tf, indent=2)
            tf.flush()
            artifact.add_file(tf.name, "sjm_params.json")
        try:
            os.unlink(tf.name)
        except Exception:
            pass
        run.log_artifact(artifact)
        # Version is assigned by W&B (v0, v1, ...)
        return artifact.version


def load_wandb(alias="latest", project=None):
    """
    Load params from W&B Artifact.
    alias: 'latest', 'production', or 'v0', 'v1', etc.
    Returns (sjm_config dict, metadata dict) or (None, None) if not found.
    """
    try:
        import wandb
        api = wandb.Api()
    except ImportError:
        logger.warning("wandb not installed, cannot load from W&B")
        return None, None
    if os.environ.get("WANDB_MODE") == "disabled":
        return None, None

    proj = project or WANDB_PROJECT
    artifact_ref = f"{proj}/{WANDB_ARTIFACT_NAME}:{alias}"
    try:
        artifact = api.artifact(artifact_ref, type=WANDB_ARTIFACT_TYPE)
    except Exception as e:
        logger.warning("W&B artifact not found %s: %s", artifact_ref, e)
        return None, None

    path = artifact.download()
    params_file = Path(path) / "sjm_params.json"
    if not params_file.exists():
        logger.warning("sjm_params.json not in artifact")
        return None, None
    with open(params_file) as f:
        doc = json.load(f)
    params = doc.get("params", {})
    cfg = {}
    for factor, p in params.items():
        cfg[factor] = {
            "jump_penalty": float(p.get("jump_penalty", PAPER_DEFAULTS["jump_penalty"])),
            "sparsity_param": float(p.get("sparsity_param", PAPER_DEFAULTS["sparsity_param"])),
        }
    return cfg, doc.get("metadata", {})


def load(source="local", version_or_alias=None):
    """
    Unified load. source: 'local' | 'wandb' | 'paper'.
    Returns sjm_config dict for HelixFactorStrategy.fit_regime_models(sjm_config=...).
    """
    if source == "paper":
        return _paper_default_config()

    if source == "local":
        cfg, _ = load_local(version_or_alias)
        return cfg

    if source == "wandb":
        cfg, _ = load_wandb(version_or_alias or "latest")
        if cfg is None:
            logger.warning("W&B load failed, falling back to local")
            cfg, _ = load_local(version_or_alias or LATEST_POINTER)
        if cfg is None or not cfg:
            logger.warning("Local load failed, using paper defaults")
            return _paper_default_config()
        return cfg

    raise ValueError("source must be 'local', 'wandb', or 'paper'")


def save(results, metadata=None, use_wandb=True, use_local=True):
    """
    Save tuning results to store(s). Returns (local_version, wandb_version).
    """
    local_ver = save_local(results, metadata) if use_local else None
    wb_ver = save_wandb(results, metadata) if use_wandb else None
    return local_ver, wb_ver

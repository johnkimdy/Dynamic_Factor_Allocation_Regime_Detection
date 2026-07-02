"""Unit tests for sjm_params_store module."""

import json
import os
import sys

import pytest
from unittest.mock import patch

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import sjm_params_store as sps


@pytest.fixture(autouse=True)
def _use_tmpdir(tmp_path, monkeypatch):
    """Redirect config dir and disable W&B for all tests."""
    config_dir = tmp_path / "config"
    monkeypatch.setattr(sps, "DEFAULT_CONFIG_DIR", config_dir)
    monkeypatch.setenv("WANDB_MODE", "disabled")


# ---------------------------------------------------------------------------
# Config conversion helpers
# ---------------------------------------------------------------------------

class TestConfigConversion:
    def test_results_to_sjm_config(self):
        results = {"QUAL": {"lambda": 40.0, "kappa_sq": 8.0}}
        cfg = sps._results_to_sjm_config(results)
        assert cfg["QUAL"]["jump_penalty"] == 40.0
        assert cfg["QUAL"]["sparsity_param"] == 8.0

    def test_results_to_sjm_config_defaults(self):
        results = {"QUAL": {}}
        cfg = sps._results_to_sjm_config(results)
        assert cfg["QUAL"]["jump_penalty"] == sps.PAPER_DEFAULTS["jump_penalty"]
        assert cfg["QUAL"]["sparsity_param"] == sps.PAPER_DEFAULTS["sparsity_param"]

    def test_results_to_sjm_config_empty(self):
        assert sps._results_to_sjm_config({}) == {}
        assert sps._results_to_sjm_config(None) == {}

    def test_sjm_config_to_results(self):
        cfg = {"QUAL": {"jump_penalty": 40.0, "sparsity_param": 8.0, "sharpe": 0.5}}
        results = sps._sjm_config_to_results(cfg)
        assert results["QUAL"]["lambda"] == 40.0
        assert results["QUAL"]["kappa_sq"] == 8.0
        assert results["QUAL"]["sharpe"] == 0.5

    def test_paper_default_config(self):
        cfg = sps._paper_default_config()
        assert len(cfg) == len(sps.FACTORS)
        for f in sps.FACTORS:
            assert cfg[f]["jump_penalty"] == 50.0
            assert cfg[f]["sparsity_param"] == 9.5


# ---------------------------------------------------------------------------
# save_local / load_local
# ---------------------------------------------------------------------------

class TestLocalStore:
    def _make_results(self, sharpe=0.5):
        return {f: {"lambda": 50.0, "kappa_sq": 9.5, "sharpe": sharpe} for f in sps.FACTORS}

    def test_save_local_creates_version(self):
        version = sps.save_local(self._make_results())
        assert version is not None
        path = sps._version_path(version)
        assert path.exists()

    def test_save_local_updates_latest_pointer(self):
        version = sps.save_local(self._make_results())
        ptr = sps._pointer_path(sps.LATEST_POINTER)
        assert ptr.exists()
        with open(ptr) as f:
            data = json.load(f)
        assert data["version"] == version

    def test_load_local_latest(self):
        sps.save_local(self._make_results())
        cfg, meta = sps.load_local("latest")
        assert "QUAL" in cfg
        assert cfg["QUAL"]["jump_penalty"] == 50.0

    def test_load_local_specific_version(self):
        version = sps.save_local(self._make_results(0.6))
        cfg, meta = sps.load_local(version)
        assert cfg is not None
        assert "QUAL" in cfg

    def test_load_local_nonexistent_version(self):
        cfg, meta = sps.load_local("99999999_000000")
        # Falls back to paper defaults
        assert cfg is not None
        assert cfg["QUAL"]["jump_penalty"] == sps.PAPER_DEFAULTS["jump_penalty"]

    def test_load_local_no_pointer(self):
        cfg, meta = sps.load_local()
        assert cfg is not None  # paper defaults
        assert all(cfg[f]["jump_penalty"] == 50.0 for f in sps.FACTORS)


# ---------------------------------------------------------------------------
# promote_local
# ---------------------------------------------------------------------------

class TestPromoteLocal:
    def test_promote_success(self):
        results = {f: {"lambda": 50.0, "kappa_sq": 9.5, "sharpe": 0.5} for f in sps.FACTORS}
        version = sps.save_local(results)
        sps.promote_local(version)
        ptr = sps._pointer_path(sps.PRODUCTION_POINTER)
        assert ptr.exists()
        with open(ptr) as f:
            data = json.load(f)
        assert data["version"] == version

    def test_promote_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            sps.promote_local("nonexistent_version")


# ---------------------------------------------------------------------------
# load (unified)
# ---------------------------------------------------------------------------

class TestUnifiedLoad:
    def test_load_paper(self):
        cfg = sps.load(source="paper")
        assert len(cfg) == len(sps.FACTORS)
        assert cfg["QUAL"]["jump_penalty"] == 50.0

    def test_load_local(self):
        results = {f: {"lambda": 40.0, "kappa_sq": 8.0, "sharpe": 0.3} for f in sps.FACTORS}
        sps.save_local(results)
        cfg = sps.load(source="local", version_or_alias="latest")
        assert cfg["QUAL"]["jump_penalty"] == 40.0

    def test_load_invalid_source(self):
        with pytest.raises(ValueError, match="source must be"):
            sps.load(source="invalid")

    def test_load_wandb_falls_back_to_local(self):
        results = {f: {"lambda": 42.0, "kappa_sq": 7.0, "sharpe": 0.4} for f in sps.FACTORS}
        sps.save_local(results)
        with patch.object(sps, "load_wandb", return_value=(None, None)):
            cfg = sps.load(source="wandb", version_or_alias="latest")
        assert cfg["QUAL"]["jump_penalty"] == 42.0


# ---------------------------------------------------------------------------
# save (combined local + wandb)
# ---------------------------------------------------------------------------

class TestSaveCombined:
    def test_save_local_only(self):
        results = {f: {"lambda": 50.0, "kappa_sq": 9.5, "sharpe": 0.5} for f in sps.FACTORS}
        local_ver, wb_ver = sps.save(results, use_wandb=False, use_local=True)
        assert local_ver is not None
        assert wb_ver is None

    def test_save_skip_both(self):
        results = {f: {"lambda": 50.0, "kappa_sq": 9.5, "sharpe": 0.5} for f in sps.FACTORS}
        local_ver, wb_ver = sps.save(results, use_wandb=False, use_local=False)
        assert local_ver is None
        assert wb_ver is None


# ---------------------------------------------------------------------------
# W&B stubs (disabled mode)
# ---------------------------------------------------------------------------

class TestWandbDisabled:
    def test_save_wandb_disabled(self):
        results = {f: {"lambda": 50.0, "kappa_sq": 9.5, "sharpe": 0.5} for f in sps.FACTORS}
        version = sps.save_wandb(results)
        assert version is None

    def test_load_wandb_disabled(self):
        with patch.object(sps, "load_wandb", return_value=(None, None)):
            cfg, meta = sps.load_wandb()
        assert cfg is None
        assert meta is None

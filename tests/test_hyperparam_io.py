"""Unit tests for hyperparam_io module."""

import json
import os
import sys
import tempfile

import pytest

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import hyperparam_io as hpio


@pytest.fixture(autouse=True)
def _use_tmpdir(tmp_path, monkeypatch):
    """Redirect HYPERPARAM_DIR to a temp directory for every test."""
    d = str(tmp_path / "hyperparam")
    monkeypatch.setattr(hpio, "HYPERPARAM_DIR", d)


# ---------------------------------------------------------------------------
# _results_to_sjm_config
# ---------------------------------------------------------------------------

class TestResultsToSjmConfig:
    def test_scalar_lambda(self):
        results = {"QUAL": {"lambda": 40.0, "kappa_sq": 8.0, "sharpe": 0.5}}
        cfg = hpio._results_to_sjm_config(results)
        assert cfg["QUAL"]["sparsity_param"] == 8.0
        assert cfg["QUAL"]["jump_penalty"] == 40.0
        assert "jump_penalty_matrix" not in cfg["QUAL"]

    def test_asymmetric_lambda(self):
        results = {"QUAL": {"lambda_enter": 80.0, "lambda_exit": 20.0, "kappa_sq": 9.5}}
        cfg = hpio._results_to_sjm_config(results)
        matrix = cfg["QUAL"]["jump_penalty_matrix"]
        assert matrix == [[0.0, 80.0], [20.0, 0.0]]
        assert "jump_penalty" not in cfg["QUAL"]

    def test_empty_results(self):
        assert hpio._results_to_sjm_config({}) == {}
        assert hpio._results_to_sjm_config(None) == {}

    def test_defaults_when_keys_missing(self):
        results = {"QUAL": {}}
        cfg = hpio._results_to_sjm_config(results)
        assert cfg["QUAL"]["sparsity_param"] == 9.5
        assert cfg["QUAL"]["jump_penalty"] == 50.0


# ---------------------------------------------------------------------------
# save_run / load_best / load_latest / load_run
# ---------------------------------------------------------------------------

class TestSaveAndLoadRun:
    def _make_results(self, sharpe=0.5):
        return {
            f: {"lambda": 50.0, "kappa_sq": 9.5, "sharpe": sharpe}
            for f in hpio.FACTORS
        }

    def test_save_run_creates_file(self):
        results = self._make_results()
        path, is_new, mean_s, prev_s = hpio.save_run(results, metadata={"note": "test"})
        assert os.path.exists(path)
        assert is_new is True
        assert prev_s is None
        assert abs(mean_s - 0.5) < 1e-6

    def test_save_run_updates_best_pointer(self):
        import time
        r1 = self._make_results(sharpe=0.3)
        hpio.save_run(r1, metadata={})
        time.sleep(1.1)  # ensure different timestamp
        r2 = self._make_results(sharpe=0.6)
        path2, is_new, mean_s, prev_s = hpio.save_run(r2, metadata={})
        assert is_new is True
        assert abs(prev_s - 0.3) < 1e-6
        assert abs(mean_s - 0.6) < 1e-6

    def test_save_run_does_not_promote_worse(self):
        r1 = self._make_results(sharpe=0.8)
        hpio.save_run(r1, metadata={})
        r2 = self._make_results(sharpe=0.2)
        _, is_new, _, _ = hpio.save_run(r2, metadata={})
        assert is_new is False

    def test_load_run_returns_config(self):
        results = self._make_results()
        path, _, _, _ = hpio.save_run(results, metadata={})
        cfg, doc = hpio.load_run(path)
        assert "QUAL" in cfg
        assert cfg["QUAL"]["jump_penalty"] == 50.0

    def test_load_run_nonexistent(self):
        cfg, doc = hpio.load_run("/nonexistent/path.json")
        assert cfg is None
        assert doc is None

    def test_load_latest(self):
        hpio.save_run(self._make_results(0.1), metadata={})
        hpio.save_run(self._make_results(0.2), metadata={})
        cfg, doc = hpio.load_latest()
        assert cfg is not None

    def test_load_latest_empty(self):
        cfg, doc = hpio.load_latest()
        assert cfg is None

    def test_load_best(self):
        hpio.save_run(self._make_results(0.5), metadata={})
        cfg, doc = hpio.load_best()
        assert cfg is not None


# ---------------------------------------------------------------------------
# save_run_with_doc / per-factor merge
# ---------------------------------------------------------------------------

class TestSaveRunWithDoc:
    def test_per_factor_merge(self):
        doc1 = {
            "metadata": {"run_date": "2024-01-01"},
            "mean_oos_sharpe": 0.3,
            "results": {
                "QUAL": {"lambda": 40.0, "kappa_sq": 8.0, "sharpe": 0.6},
                "MTUM": {"lambda": 60.0, "kappa_sq": 10.0, "sharpe": 0.2},
            },
        }
        path1, updated1, _ = hpio.save_run_with_doc(doc1)
        assert "QUAL" in updated1
        assert "MTUM" in updated1

        doc2 = {
            "metadata": {"run_date": "2024-02-01"},
            "mean_oos_sharpe": 0.35,
            "results": {
                "QUAL": {"lambda": 45.0, "kappa_sq": 9.0, "sharpe": 0.4},  # worse
                "MTUM": {"lambda": 55.0, "kappa_sq": 11.0, "sharpe": 0.5},  # better
            },
        }
        path2, updated2, _ = hpio.save_run_with_doc(doc2)
        assert "QUAL" not in updated2  # not improved
        assert "MTUM" in updated2  # improved

        # Verify merged best has QUAL from doc1, MTUM from doc2
        best_path = os.path.join(hpio.HYPERPARAM_DIR, hpio.BEST_MERGED_FILE)
        with open(best_path) as f:
            best = json.load(f)
        assert best["results"]["QUAL"]["sharpe"] == 0.6
        assert best["results"]["MTUM"]["sharpe"] == 0.5


# ---------------------------------------------------------------------------
# load_sjm_config
# ---------------------------------------------------------------------------

class TestLoadSjmConfig:
    def test_load_sjm_config_best(self):
        results = {f: {"lambda": 50.0, "kappa_sq": 9.5, "sharpe": 0.5} for f in hpio.FACTORS}
        hpio.save_run(results, metadata={})
        cfg = hpio.load_sjm_config(use_best=True)
        assert cfg is not None
        assert "QUAL" in cfg

    def test_load_sjm_config_latest(self):
        results = {f: {"lambda": 50.0, "kappa_sq": 9.5, "sharpe": 0.5} for f in hpio.FACTORS}
        hpio.save_run(results, metadata={})
        cfg = hpio.load_sjm_config(use_best=False)
        assert cfg is not None

    def test_load_sjm_config_no_data(self):
        cfg = hpio.load_sjm_config(use_best=True)
        assert cfg is None


# ---------------------------------------------------------------------------
# Asymmetric runs
# ---------------------------------------------------------------------------

class TestAsymmetricRuns:
    def test_save_and_load_asymmetric(self):
        doc = {
            "metadata": {"run_date": "2024-03-01"},
            "mean_oos_sharpe": 0.4,
            "results": {
                "QUAL": {"lambda_enter": 80.0, "lambda_exit": 20.0, "kappa_sq": 9.5, "sharpe": 0.4},
            },
        }
        path, updated, _ = hpio.save_asymmetric_run_with_doc(doc)
        assert os.path.exists(path)
        assert "QUAL" in updated

        cfg, loaded_doc = hpio.load_asymmetric_best()
        assert cfg is not None
        assert "QUAL" in cfg
        assert "jump_penalty_matrix" in cfg["QUAL"]


# ---------------------------------------------------------------------------
# load_results_for_store
# ---------------------------------------------------------------------------

class TestLoadResultsForStore:
    def test_returns_results_from_best(self):
        doc = {
            "metadata": {"run_date": "2024-01-01"},
            "mean_oos_sharpe": 0.5,
            "results": {"QUAL": {"lambda": 50.0, "kappa_sq": 9.5, "sharpe": 0.5}},
        }
        hpio.save_run_with_doc(doc)
        results = hpio.load_results_for_store()
        assert results is not None
        assert "QUAL" in results

    def test_returns_none_when_empty(self):
        results = hpio.load_results_for_store()
        assert results is None

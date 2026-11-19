import numpy as np
import pytest

from uplift import causal


def test_aipw_removes_confounding_that_biases_the_naive_estimate(monkeypatch):
    """Two pooled 'tests' with different treatment shares and base rates."""
    monkeypatch.setattr(causal, "NUISANCE_PARAMS",
                        {**causal.NUISANCE_PARAMS, "n_estimators": 60, "min_child_samples": 200, "n_jobs": 2})
    rng = np.random.default_rng(0)
    n = 40000
    test_id = rng.integers(0, 2, n)
    X = np.column_stack([test_id + rng.normal(0, 0.1, n), rng.normal(size=n)]).astype(np.float32)
    share = np.where(test_id == 1, 0.95, 0.60)
    base = np.where(test_id == 1, 0.05, 0.30)
    t = rng.binomial(1, share)
    y = rng.binomial(1, base + 0.03 * t)
    est = causal.aipw(X, t, {"y": y}, groups=np.arange(n), n_folds=2, seed=1)["y"]
    assert abs(est["difference_in_means"] - 0.03) > 0.05      # naive is badly biased
    assert est["aipw_ci"][0] < 0.03 < est["aipw_ci"][1]        # AIPW covers the truth


def test_difference_in_means():
    t = np.array([1, 1, 0, 0])
    y = np.array([1.0, 0.0, 0.0, 0.0])
    ate, se = causal.difference_in_means(t, y)
    assert ate == pytest.approx(0.5)
    assert se > 0

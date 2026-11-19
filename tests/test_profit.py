import numpy as np
import pytest

from uplift import metrics, profit


@pytest.fixture
def trial():
    rng = np.random.default_rng(0)
    n = 60000
    x = rng.normal(size=n)
    tau = np.where(x > 0.5, 0.06, -0.02)
    t = rng.binomial(1, 0.85, n)
    y = rng.binomial(1, 0.05 + tau * t)
    return x, tau, t, y


def test_treat_all_matches_end_of_profit_curve(trial):
    x, tau, t, y = trial
    c = metrics.curve(tau, t, y)
    p = profit.profit_curve(c, cost=0.01, value=1.0)
    blanket = profit.blanket_policies(t, y, cost=0.01, value=1.0)
    assert p[-1] == pytest.approx(blanket["treat_all"])
    assert p[0] == 0.0


def test_policy_thresholds(trial):
    x, tau, t, y = trial
    every = profit.evaluate_policy(tau, t, y, -np.inf, cost=0.01, value=1.0)
    nobody = profit.evaluate_policy(tau, t, y, np.inf, cost=0.01, value=1.0)
    assert every["fraction"] == pytest.approx(1.0)
    assert every["profit_per_user"] == pytest.approx(
        profit.blanket_policies(t, y, 0.01, 1.0)["treat_all"])
    assert nobody["profit_per_user"] == 0.0


def test_best_policy_targets_the_positive_segment(trial):
    x, tau, t, y = trial
    c = metrics.curve(tau, t, y)
    pol = profit.best_policy(tau, c, cost=0.01, value=1.0)
    # only users with tau = 0.06 are worth targeting (0.06 > 0.01 > -0.02)
    assert pol.fraction == pytest.approx((tau > 0).mean(), abs=1e-9)
    assert pol.threshold == pytest.approx(0.06)
    best = profit.evaluate_policy(tau, t, y, pol.threshold, 0.01, 1.0)
    assert best["profit_per_user"] > profit.blanket_policies(t, y, 0.01, 1.0)["treat_all"]


def test_expensive_campaign_targets_nobody(trial):
    x, tau, t, y = trial
    c = metrics.curve(tau, t, y)
    pol = profit.best_policy(tau, c, cost=10.0, value=1.0)
    assert pol.fraction == 0.0 and pol.threshold == np.inf

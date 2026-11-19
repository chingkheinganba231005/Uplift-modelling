import numpy as np
import pytest

from uplift import metrics


def test_hand_computed_curve():
    score = np.array([4.0, 3.0, 2.0, 1.0])
    t = np.array([1, 0, 1, 0])
    y = np.array([1, 0, 0, 0])
    c = metrics.curve(score, t, y)
    np.testing.assert_allclose(c.x, [0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(c.uplift, [0, 0.25, 0.5, 0.375, 0.5])
    np.testing.assert_allclose(c.qini, [0, 0.25, 0.25, 0.25, 0.25])
    assert metrics.auuc(c) == pytest.approx(0.34375)
    # Qini area 0.21875 minus random-line area 0.125
    assert metrics.qini_coefficient(c) == pytest.approx(0.21875 - 0.125)


def test_curve_ends_at_ate_for_any_ranking():
    rng = np.random.default_rng(0)
    t = rng.integers(0, 2, 5000)
    y = rng.binomial(1, 0.1 + 0.05 * t)
    ate = y[t == 1].mean() - y[t == 0].mean()
    for score in (rng.normal(size=5000), np.zeros(5000), np.arange(5000.0)):
        c = metrics.curve(score, t, y)
        assert c.uplift[-1] == pytest.approx(ate)
        assert c.x[-1] == pytest.approx(1.0)


def test_ties_are_kept_together():
    rng = np.random.default_rng(1)
    t = rng.integers(0, 2, 1000)
    y = rng.binomial(1, 0.2, 1000)
    c = metrics.curve(np.ones(1000), t, y)
    assert len(c.x) == 2  # origin and the single block of ties
    assert metrics.auuc(c) == pytest.approx(c.uplift[-1] / 2)
    assert metrics.qini_coefficient(c) == pytest.approx(0.0)


def test_ordering_matters_in_the_right_direction():
    rng = np.random.default_rng(2)
    n = 40000
    x = rng.normal(size=n)
    tau = np.where(x > 0, 0.10, -0.05)
    t = rng.binomial(1, 0.85, n)
    y = rng.binomial(1, 0.2 + tau * t)
    good = metrics.summary(metrics.curve(tau, t, y))
    bad = metrics.summary(metrics.curve(-tau, t, y))
    assert good["auuc"] > good["auuc_random"] > bad["auuc"]
    assert good["qini_coefficient"] > 0 > bad["qini_coefficient"]


def test_monotone_transform_invariance():
    rng = np.random.default_rng(3)
    s = rng.normal(size=2000)
    t = rng.integers(0, 2, 2000)
    y = rng.binomial(1, 0.3, 2000)
    a = metrics.curve(s, t, y)
    b = metrics.curve(np.exp(3 * s) + 7, t, y)
    np.testing.assert_allclose(a.uplift, b.uplift)


def test_integer_weights_equal_row_replication():
    rng = np.random.default_rng(4)
    s = rng.normal(size=300)
    t = rng.integers(0, 2, 300)
    y = rng.binomial(1, 0.4, 300)
    w = rng.integers(0, 4, 300)
    weighted = metrics.curve(s, t, y, weights=w)
    rep = np.repeat(np.arange(300), w)
    replicated = metrics.curve(s[rep], t[rep], y[rep])
    assert metrics.auuc(weighted) == pytest.approx(metrics.auuc(replicated))
    assert metrics.qini_coefficient(weighted) == pytest.approx(metrics.qini_coefficient(replicated))


def test_uplift_at_fraction():
    score = np.array([4.0, 3.0, 2.0, 1.0])
    t = np.array([1, 0, 1, 0])
    y = np.array([1, 0, 0, 0])
    c = metrics.curve(score, t, y)
    # top half: one treated converter, one control non-converter -> effect 1.0
    assert metrics.uplift_at(c, 0.5) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        metrics.uplift_at(c, 0.0)


def test_bootstrap_is_reproducible_and_paired():
    rng = np.random.default_rng(5)
    n = 5000
    s = rng.normal(size=n)
    t = rng.integers(0, 2, n)
    y = rng.binomial(1, 0.2, n)
    groups = np.arange(n)
    a = metrics.bootstrap({"m": s, "same": s.copy()}, t, y, groups, n_boot=20, seed=9)
    b = metrics.bootstrap({"m": s, "same": s.copy()}, t, y, groups, n_boot=20, seed=9)
    np.testing.assert_array_equal(a["metrics"]["m"]["auuc"], b["metrics"]["m"]["auuc"])
    # identical scores under the same weights give identical replicates
    np.testing.assert_array_equal(a["metrics"]["m"]["auuc"], a["metrics"]["same"]["auuc"])


def test_group_weights_are_shared_within_group():
    rng = np.random.default_rng(6)
    groups = np.array([0, 0, 1, 2, 2, 2])
    w = metrics.group_poisson_weights(groups, 3, rng)
    assert w[0] == w[1] and w[3] == w[4] == w[5]


def test_deciles_break_ties_at_random_not_by_row_order():
    # all scores tied and rows sorted by treatment, as in the raw file
    n = 10000
    t = np.r_[np.ones(8500, int), np.zeros(1500, int)]
    y = np.zeros(n, int)
    d = metrics.deciles(np.zeros(n), t, y, seed=0)
    shares = [r["n_treated"] / r["n"] for r in d]
    assert min(shares) > 0.8 and max(shares) < 0.9


def test_deciles_values():
    rng = np.random.default_rng(7)
    n = 20000
    s = rng.normal(size=n)
    t = rng.integers(0, 2, n)
    y = rng.binomial(1, np.where(s > np.quantile(s, 0.9), 0.3, 0.1) + 0.05 * t)
    d = metrics.deciles(s, t, y)
    assert len(d) == 10 and sum(r["n"] for r in d) == n
    assert d[0]["rate_control"] > d[-1]["rate_control"]
    assert all(abs(r["uplift"] - 0.05) < 0.05 for r in d)

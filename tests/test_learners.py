import numpy as np
import pytest
from scipy.stats import spearmanr

from uplift import learners


@pytest.fixture(scope="module")
def rct():
    """Synthetic randomised trial with a known effect, for checking the learners' logic."""
    rng = np.random.default_rng(42)
    n = 60000
    X = rng.normal(size=(n, 4)).astype(np.float32)
    tau = 0.15 * (X[:, 0] > 0) - 0.05 * (X[:, 0] <= 0)
    base = 0.25 + 0.1 * np.tanh(X[:, 1])
    t = rng.binomial(1, 0.85, n).astype(np.int8)
    y = rng.binomial(1, np.clip(base + tau * t, 0, 1)).astype(np.int8)
    return X, t, y, tau


def _split(X, t, y):
    k = int(0.8 * len(t))
    return (X[:k], t[:k], y[:k]), (X[k:], t[k:], y[k:])


@pytest.mark.parametrize("name", ["s_learner", "t_learner", "x_learner", "class_transformation"])
def test_learner_recovers_effect_ranking(rct, name, monkeypatch):
    monkeypatch.setitem(learners.LGB_PARAMS, "min_child_samples", 200)
    X, t, y, tau = rct
    train, val = _split(X, t, y)
    model = learners.make_learners(seed=0)[name].fit(*train, val=val)
    pred = model.predict(val[0])
    true = tau[len(train[1]):]
    assert spearmanr(pred, true).statistic > 0.5
    # segment means have the right sign and roughly the right size
    assert pred[true > 0].mean() > 0.05
    assert pred[true < 0].mean() < 0.02


def test_response_model_ignores_control_group(rct, monkeypatch):
    monkeypatch.setitem(learners.LGB_PARAMS, "min_child_samples", 200)
    X, t, y, tau = rct
    train, val = _split(X, t, y)
    model = learners.ResponseModel(seed=0).fit(*train, val=val)
    pred = model.predict(val[0])
    assert pred.min() >= 0 and pred.max() <= 1


def test_class_transformation_weights_balance_arms():
    ct = learners.ClassTransformation(propensity=0.85)
    t = np.array([1] * 85 + [0] * 15)
    z, w = ct._zw(t, np.zeros(100, dtype=int))
    assert w[t == 1].sum() == pytest.approx(w[t == 0].sum())
    np.testing.assert_array_equal(z, 1 - t)


def test_causal_forest_runs_and_ranks(rct):
    X, t, y, tau = rct
    cf = learners.CausalForest(seed=0, n_estimators=40, min_samples_leaf=200, max_rows=20000)
    cf.fit(X, t, y)
    pred = cf.predict(X[:5000])
    assert spearmanr(pred, tau[:5000]).statistic > 0.5

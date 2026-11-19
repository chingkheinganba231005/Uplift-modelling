"""Uplift learners built on LightGBM.

Every learner exposes ``fit(X, t, y, val=(X, t, y))`` and ``predict(X)``,
which returns the estimated individual uplift tau(x) = P(y=1 | x, treated) -
P(y=1 | x, control), or a score that ranks users in the same order.

The validation set is only used for early stopping of each component on its
own loss (log-loss or squared error); model comparison happens elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np

from . import config

ValSet = tuple[np.ndarray, np.ndarray, np.ndarray] | None

LGB_PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=2000,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    verbose=-1,
    n_jobs=4,
)
EARLY_STOPPING = 50


def _classifier(seed: int, **overrides) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(random_state=seed, **{**LGB_PARAMS, **overrides})


def _regressor(seed: int, **overrides) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(random_state=seed, **{**LGB_PARAMS, **overrides})


def _fit(model, X, y, sample_weight=None, eval_set=None, eval_weight=None):
    kwargs = {}
    if eval_set is not None:
        kwargs = dict(
            eval_X=(eval_set[0],),
            eval_y=(eval_set[1],),
            eval_sample_weight=None if eval_weight is None else [eval_weight],
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
        )
    model.fit(X, y, sample_weight=sample_weight, **kwargs)
    return model


def _p1(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


@dataclass
class ResponseModel:
    """Baseline: rank users by P(convert | x, treated), ignoring the control group."""

    seed: int = config.SEED
    model: lgb.LGBMClassifier | None = None

    def fit(self, X, t, y, val: ValSet = None):
        m = t == 1
        ev = None if val is None else (val[0][val[1] == 1], val[2][val[1] == 1])
        self.model = _fit(_classifier(self.seed), X[m], y[m], eval_set=ev)
        return self

    def predict(self, X) -> np.ndarray:
        return _p1(self.model, X)


@dataclass
class SLearner:
    """One model of y on (x, t); uplift = f(x, 1) - f(x, 0)."""

    seed: int = config.SEED
    model: lgb.LGBMClassifier | None = None

    @staticmethod
    def _with_t(X, t):
        return np.column_stack([X, t]).astype(np.float32)

    def fit(self, X, t, y, val: ValSet = None):
        ev = None if val is None else (self._with_t(val[0], val[1]), val[2])
        self.model = _fit(_classifier(self.seed), self._with_t(X, t), y, eval_set=ev)
        return self

    def predict(self, X) -> np.ndarray:
        ones = np.ones(len(X), dtype=np.float32)
        return _p1(self.model, self._with_t(X, ones)) - _p1(self.model, self._with_t(X, 0 * ones))


@dataclass
class TLearner:
    """Separate outcome models for treated and control users."""

    seed: int = config.SEED
    model_t: lgb.LGBMClassifier | None = None
    model_c: lgb.LGBMClassifier | None = None

    def fit(self, X, t, y, val: ValSet = None):
        for arm in (1, 0):
            m = t == arm
            ev = None if val is None else (val[0][val[1] == arm], val[2][val[1] == arm])
            model = _fit(_classifier(self.seed + arm), X[m], y[m], eval_set=ev)
            if arm == 1:
                self.model_t = model
            else:
                self.model_c = model
        return self

    def mu1(self, X):
        return _p1(self.model_t, X)

    def mu0(self, X):
        return _p1(self.model_c, X)

    def predict(self, X) -> np.ndarray:
        return self.mu1(X) - self.mu0(X)


@dataclass
class XLearner:
    """Kunzel et al. (2019).

    Stage 1: the T-learner outcome models mu1, mu0.
    Stage 2: impute individual effects, D1 = y - mu0(x) for treated users and
    D0 = mu1(x) - y for control users, and regress them on x.
    Stage 3: tau(x) = g * tau0(x) + (1 - g) * tau1(x) with g the propensity.
    With 85% of users treated, mu1 is learned from far more data than mu0, so
    tau0 (built from mu1) gets the larger weight.
    """

    seed: int = config.SEED
    propensity: float = config.PROPENSITY
    stage1: TLearner = field(default_factory=TLearner)
    tau1: lgb.LGBMRegressor | None = None
    tau0: lgb.LGBMRegressor | None = None

    def _imputed(self, X, t, y):
        d = np.empty(len(t), dtype=np.float64)
        m1 = t == 1
        d[m1] = y[m1] - self.stage1.mu0(X[m1])
        d[~m1] = self.stage1.mu1(X[~m1]) - y[~m1]
        return d

    def fit(self, X, t, y, val: ValSet = None):
        # An already fitted T-learner can be passed in as stage 1 to avoid
        # fitting the same two outcome models twice.
        if self.stage1.model_t is None:
            self.stage1 = TLearner(seed=self.seed).fit(X, t, y, val=val)
        d = self._imputed(X, t, y)
        d_val = None if val is None else self._imputed(*val)
        for arm in (1, 0):
            m = t == arm
            ev = None if val is None else (val[0][val[1] == arm], d_val[val[1] == arm])
            model = _fit(_regressor(self.seed + 10 + arm), X[m], d[m], eval_set=ev)
            if arm == 1:
                self.tau1 = model
            else:
                self.tau0 = model
        return self

    def predict(self, X) -> np.ndarray:
        g = self.propensity
        return g * self.tau0.predict(X) + (1 - g) * self.tau1.predict(X)


@dataclass
class ClassTransformation:
    """Class-variable transformation (Jaskowski and Jaroszewicz, 2012).

    Define z = 1 for treated converters and control non-converters. With the
    arms reweighted to equal size, P(z=1 | x) = (1 + tau(x)) / 2, so a single
    classifier on z recovers tau(x) = 2 P(z=1 | x) - 1. The reweighting is what
    makes the method valid when treatment is not split 50/50 (here 85/15).
    """

    seed: int = config.SEED
    propensity: float = config.PROPENSITY
    model: lgb.LGBMClassifier | None = None

    def _zw(self, t, y):
        z = (y * t + (1 - y) * (1 - t)).astype(np.int8)
        w = np.where(t == 1, 0.5 / self.propensity, 0.5 / (1 - self.propensity))
        return z, w

    def fit(self, X, t, y, val: ValSet = None):
        z, w = self._zw(t, y)
        ev, ew = None, None
        if val is not None:
            zv, wv = self._zw(val[1], val[2])
            ev, ew = (val[0], zv), wv
        self.model = _fit(_classifier(self.seed), X, z, sample_weight=w, eval_set=ev, eval_weight=ew)
        return self

    def predict(self, X) -> np.ndarray:
        return 2 * _p1(self.model, X) - 1


@dataclass
class CausalForest:
    """Causal forest (Athey, Tibshirani and Wager, 2019) via econml's CausalForestDML.

    The outcome nuisance model is LightGBM. The treatment nuisance model is the
    known design propensity: in a randomised trial there is nothing to learn,
    and estimating it would only add noise. Fitted on a subsample of the
    training set because the forest does not scale to 8 million rows on a
    laptop.
    """

    seed: int = config.SEED
    n_estimators: int = 200
    min_samples_leaf: int = 1000
    max_rows: int = 1_000_000
    model: object | None = None

    def fit(self, X, t, y, val: ValSet = None):
        from econml.dml import CausalForestDML
        from sklearn.dummy import DummyClassifier

        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(t), size=min(self.max_rows, len(t)), replace=False)
        self.model = CausalForestDML(
            model_y=_classifier(self.seed, n_estimators=300),
            model_t=DummyClassifier(strategy="prior"),
            discrete_treatment=True,
            discrete_outcome=True,
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_samples=0.3,
            honest=True,
            cv=2,
            n_jobs=4,
            random_state=self.seed,
        )
        self.model.fit(y[idx], t[idx], X=X[idx])
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.effect(X).ravel()


def make_learners(seed: int = config.SEED) -> dict[str, object]:
    return {
        "response": ResponseModel(seed=seed),
        "s_learner": SLearner(seed=seed),
        "t_learner": TLearner(seed=seed),
        "x_learner": XLearner(seed=seed),
        "class_transformation": ClassTransformation(seed=seed),
        "causal_forest": CausalForest(seed=seed),
    }


LABELS = {
    "random": "Random targeting",
    "response": "Response model",
    "s_learner": "S-learner",
    "t_learner": "T-learner",
    "x_learner": "X-learner",
    "class_transformation": "Class transformation",
    "causal_forest": "Causal forest",
}

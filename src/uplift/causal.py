"""Average treatment effect estimators used as robustness checks.

The headline ATE is the difference in means (sql/06_ate.sql), which is
unbiased if every user had the same chance of treatment. The file pools
several tests whose treatment shares differ, so the augmented inverse
propensity weighted (AIPW, doubly robust) estimator is computed alongside:

    psi_i = mu1(x) - mu0(x) + t (y - mu1(x)) / e(x) - (1 - t)(y - mu0(x)) / (1 - e(x))
    ATE   = mean(psi),  SE = sd(psi) / sqrt(n)

with e, mu1, mu0 fitted by LightGBM using K-fold cross-fitting (each row is
predicted by models that never saw it). It is consistent if either the
propensity model or the outcome models are right.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np

NUISANCE_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=63,
                       min_child_samples=1000, subsample=0.8, subsample_freq=1,
                       verbose=-1, n_jobs=4)


def difference_in_means(t: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    y1, y0 = y[t == 1], y[t == 0]
    ate = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    return float(ate), float(se)


def aipw(X: np.ndarray, t: np.ndarray, ys: dict[str, np.ndarray], groups: np.ndarray,
         n_folds: int = 2, seed: int = 0, clip: float = 0.01) -> dict:
    """Cross-fitted AIPW estimates for several outcomes sharing one propensity model.

    Folds are assigned by group so that identical feature vectors are always
    predicted out of fold.
    """
    rng = np.random.default_rng(seed)
    fold_of_group = rng.integers(0, n_folds, size=int(groups.max()) + 1)
    fold = fold_of_group[groups]
    e = np.empty(len(t))
    mu1 = {k: np.empty(len(t)) for k in ys}
    mu0 = {k: np.empty(len(t)) for k in ys}
    for k in range(n_folds):
        fit, pred = fold != k, fold == k
        m = lgb.LGBMClassifier(random_state=seed + k, **NUISANCE_PARAMS).fit(X[fit], t[fit])
        e[pred] = m.predict_proba(X[pred])[:, 1]
        for name, y in ys.items():
            for arm, store in ((1, mu1[name]), (0, mu0[name])):
                sel = fit & (t == arm)
                mo = lgb.LGBMClassifier(random_state=seed + 10 * k + arm, **NUISANCE_PARAMS).fit(X[sel], y[sel])
                store[pred] = mo.predict_proba(X[pred])[:, 1]
    e = np.clip(e, clip, 1 - clip)
    out = {"propensity_range": [float(e.min()), float(e.max())], "n": int(len(t)), "folds": n_folds}
    for name, y in ys.items():
        psi = (mu1[name] - mu0[name] + t * (y - mu1[name]) / e
               - (1 - t) * (y - mu0[name]) / (1 - e))
        ate, se = float(psi.mean()), float(psi.std(ddof=1) / np.sqrt(len(psi)))
        dm, dm_se = difference_in_means(t, y)
        out[name] = {
            "aipw": ate, "aipw_se": se, "aipw_ci": [ate - 1.959964 * se, ate + 1.959964 * se],
            "difference_in_means": dm, "dm_se": dm_se,
            "dm_ci": [dm - 1.959964 * dm_se, dm + 1.959964 * dm_se],
        }
    return out

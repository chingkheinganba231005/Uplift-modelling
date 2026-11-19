"""Uplift and Qini curves, areas under them, and a grouped bootstrap.

Conventions
-----------
Users are sorted by predicted uplift, highest first. For the top fraction
``x`` of the population, with n_t, n_c treated and control users and y_t, y_c
outcomes among them:

* uplift curve:  u(x) = (y_t / n_t - y_c / n_c) * (n_t + n_c) / N
  the extra outcomes per user of the whole population if only the top x were
  targeted (the effect inside the segment times its size).
* Qini curve:    q(x) = (y_t - y_c * n_t / n_c) / N
  the same quantity measured in treated-group units, as in Radcliffe (2007).

Both curves start at 0 and end at the average treatment effect (ATE) at x = 1.
Random targeting gives the straight line x * ATE, so the area under it is
ATE / 2. The AUUC is the area under u; the Qini coefficient is the area
between q and its random line. Users with tied scores are kept together:
the curves are only evaluated at the end of each block of ties.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Curve:
    x: np.ndarray       # fraction of the population targeted, starts at 0
    uplift: np.ndarray  # u(x)
    qini: np.ndarray    # q(x)
    n_treated: np.ndarray
    n_control: np.ndarray


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a, dtype=float)
    np.divide(a, b, out=out, where=b > 0)
    return out


def curve(score: np.ndarray, t: np.ndarray, y: np.ndarray,
          weights: np.ndarray | None = None, order: np.ndarray | None = None) -> Curve:
    """Uplift and Qini curves for one ranking.

    ``order`` (indices sorting ``score`` from high to low) can be passed to
    avoid sorting again inside a bootstrap loop.
    """
    score = np.asarray(score)
    if order is None:
        order = np.argsort(-score, kind="stable")
    s = score[order]
    tt = np.asarray(t, dtype=float)[order]
    yy = np.asarray(y, dtype=float)[order]
    w = np.ones_like(tt) if weights is None else np.asarray(weights, dtype=float)[order]

    n_t = np.cumsum(w * tt)
    n_c = np.cumsum(w * (1.0 - tt))
    y_t = np.cumsum(w * yy * tt)
    y_c = np.cumsum(w * yy * (1.0 - tt))

    end_of_block = np.r_[s[1:] != s[:-1], True]
    n_t, n_c, y_t, y_c = n_t[end_of_block], n_c[end_of_block], y_t[end_of_block], y_c[end_of_block]

    total = n_t[-1] + n_c[-1]
    n = n_t + n_c
    uplift = (_safe_div(y_t, n_t) - _safe_div(y_c, n_c)) * n / total
    qini = (y_t - _safe_div(y_c * n_t, n_c)) / total
    zero = np.zeros(1)
    return Curve(
        x=np.r_[zero, n / total],
        uplift=np.r_[zero, uplift],
        qini=np.r_[zero, qini],
        n_treated=np.r_[zero, n_t],
        n_control=np.r_[zero, n_c],
    )


def _area(x: np.ndarray, v: np.ndarray) -> float:
    return float(np.sum((x[1:] - x[:-1]) * (v[1:] + v[:-1]) / 2.0))


def auuc(c: Curve) -> float:
    """Area under the uplift curve (outcomes per user, averaged over x)."""
    return _area(c.x, c.uplift)


def qini_coefficient(c: Curve) -> float:
    """Area between the Qini curve and the random-targeting line."""
    return _area(c.x, c.qini) - c.qini[-1] / 2.0


def uplift_at(c: Curve, fraction: float) -> float:
    """Average treatment effect inside the top ``fraction`` of users."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    return float(np.interp(fraction, c.x, c.uplift) / fraction)


def on_grid(c: Curve, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Uplift and Qini curves linearly interpolated onto ``grid``."""
    return np.interp(grid, c.x, c.uplift), np.interp(grid, c.x, c.qini)


def summary(c: Curve) -> dict[str, float]:
    ate = float(c.uplift[-1])
    return {
        "auuc": auuc(c),
        "auuc_random": ate / 2.0,
        "qini_coefficient": qini_coefficient(c),
        "ate": ate,
        "uplift_top10": uplift_at(c, 0.10),
        "uplift_top30": uplift_at(c, 0.30),
    }


def group_poisson_weights(groups: np.ndarray, n_groups: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Poisson(1) bootstrap weights drawn per group and copied to its rows.

    Rows with identical feature vectors are resampled together, which keeps
    the bootstrap honest about their dependence.
    """
    return rng.poisson(1.0, size=n_groups).astype(np.float64)[groups]


def bootstrap(scores: dict[str, np.ndarray], t: np.ndarray, y: np.ndarray,
              groups: np.ndarray, n_boot: int, seed: int,
              grid: np.ndarray | None = None) -> dict:
    """Paired bootstrap of curve summaries for several rankings.

    Every replicate uses the same weights for all models, so differences
    between models are paired. Returns, for each model, an array of metric
    values per replicate and (if ``grid`` is given) the curves on that grid.
    """
    rng = np.random.default_rng(seed)
    n_groups = int(groups.max()) + 1
    orders = {k: np.argsort(-np.asarray(s), kind="stable") for k, s in scores.items()}
    metrics = {k: {m: np.empty(n_boot) for m in ("auuc", "qini_coefficient", "ate")}
               for k in scores}
    curves = {k: np.empty((n_boot, len(grid))) for k in scores} if grid is not None else None
    for b in range(n_boot):
        w = group_poisson_weights(groups, n_groups, rng)
        for k, s in scores.items():
            c = curve(s, t, y, weights=w, order=orders[k])
            metrics[k]["auuc"][b] = auuc(c)
            metrics[k]["qini_coefficient"][b] = qini_coefficient(c)
            metrics[k]["ate"][b] = c.uplift[-1]
            if curves is not None:
                curves[k][b] = np.interp(grid, c.x, c.uplift)
    return {"metrics": metrics, "curves": curves}


def percentile_ci(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    lo, hi = np.quantile(samples, [a, 1 - a])
    return float(lo), float(hi)

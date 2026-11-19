"""Turn uplift rankings into targeting decisions and money.

Profit of targeting a set S of users, relative to targeting nobody:

    profit(S) = value * incremental_conversions(S) - cost * |S|

incremental_conversions(S) is estimated from the randomised test set as the
difference in conversion rates between treated and control users inside S,
times |S|. The cost is charged per targeted user. Treat-none earns 0 by
definition; random targeting of a fraction x earns x * (value * ATE - cost)
per user.

The targeting threshold is chosen on the validation set and then applied,
unchanged, to the test set. Choosing it on the test set would report the
best point of a noisy curve, which is optimistic (the optimiser's curse).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import Curve


def profit_curve(c: Curve, cost: float, value: float, population: float = 1.0) -> np.ndarray:
    """Profit for each point of the uplift curve, for a population of ``population`` users."""
    return population * (value * c.uplift - cost * c.x)


@dataclass
class Policy:
    threshold: float   # target users with score >= threshold
    fraction: float    # share of the selection population targeted


def best_policy(score: np.ndarray, c: Curve, cost: float, value: float) -> Policy:
    """Profit-maximising cut-off on a curve computed from ``score``.

    ``c`` must be the curve of the same ``score`` (the selection set). The
    returned threshold is the score of the last user included.
    """
    p = profit_curve(c, cost, value)
    k = int(np.argmax(p))
    frac = float(c.x[k])
    if k == 0:
        return Policy(threshold=np.inf, fraction=0.0)
    s = np.sort(np.asarray(score))[::-1]
    n_included = int(round(frac * len(s)))
    return Policy(threshold=float(s[n_included - 1]), fraction=frac)


def evaluate_policy(score: np.ndarray, t: np.ndarray, y: np.ndarray, threshold: float,
                    cost: float, value: float, weights: np.ndarray | None = None) -> dict:
    """Per-user profit of targeting everyone with score >= threshold."""
    w = np.ones(len(t)) if weights is None else weights
    sel = np.asarray(score) >= threshold
    n = w.sum()
    n_t = (w * sel * (t == 1)).sum()
    n_c = (w * sel * (t == 0)).sum()
    if n_t == 0 or n_c == 0:
        return {"fraction": float((w * sel).sum() / n), "incremental_per_user": 0.0,
                "profit_per_user": 0.0}
    rate_t = (w * sel * (t == 1) * y).sum() / n_t
    rate_c = (w * sel * (t == 0) * y).sum() / n_c
    frac = (n_t + n_c) / n
    inc = (rate_t - rate_c) * frac
    return {
        "fraction": float(frac),
        "incremental_per_user": float(inc),
        "profit_per_user": float(value * inc - cost * frac),
    }


def blanket_policies(t: np.ndarray, y: np.ndarray, cost: float, value: float,
                     weights: np.ndarray | None = None) -> dict[str, float]:
    """Per-user profit of treating everyone and of treating nobody."""
    w = np.ones(len(t)) if weights is None else weights
    rate_t = (w * (t == 1) * y).sum() / (w * (t == 1)).sum()
    rate_c = (w * (t == 0) * y).sum() / (w * (t == 0)).sum()
    return {"treat_all": float(value * (rate_t - rate_c) - cost), "treat_none": 0.0}

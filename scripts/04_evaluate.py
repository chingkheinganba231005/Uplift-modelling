"""Stage 4: compare the learners on validation, then evaluate once on test.

Model selection uses the validation split only (Qini coefficient). The test
split is then scored once for every model, with a grouped, paired bootstrap
giving confidence intervals for each model and for differences between
models. Writes results/evaluation.json and figures/qini_*.png, deciles_*.png.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np

from _common import read_json, write_json
from uplift import config, data, metrics, plots
from uplift.learners import LABELS

log = logging.getLogger("evaluate")

UPLIFT_MODELS = ["s_learner", "t_learner", "x_learner", "class_transformation", "causal_forest"]
ALL_MODELS = ["random", "response"] + UPLIFT_MODELS
GRID = np.linspace(0, 1, 101)


def evaluate_outcome(outcome: str, val: data.Split, test: data.Split, e_test: np.ndarray) -> dict:
    s = np.load(config.ARTIFACTS_DIR / f"scores_{outcome}.npz")

    val_metrics = {m: metrics.summary(metrics.curve(s[f"val_{m}"], val.t, val.y[outcome]))
                   for m in ALL_MODELS}
    selected = max(UPLIFT_MODELS, key=lambda m: val_metrics[m]["qini_coefficient"])
    log.info("%s: selected %s on validation", outcome, selected)

    y = test.y[outcome]
    scores = {m: s[f"test_{m}"] for m in ALL_MODELS}
    point = {m: metrics.summary(metrics.curve(scores[m], test.t, y)) for m in ALL_MODELS}
    boot = metrics.bootstrap(scores, test.t, y, test.groups, n_boot=config.N_BOOTSTRAP,
                             seed=config.SEED, grid=GRID)
    bm = boot["metrics"]

    test_metrics = {}
    for m in ALL_MODELS:
        test_metrics[m] = dict(point[m])
        for k in ("auuc", "qini_coefficient"):
            lo, hi = metrics.percentile_ci(bm[m][k])
            test_metrics[m][f"{k}_ci"] = [lo, hi]
        # paired differences against the baselines
        for base in ("random", "response"):
            if m == base:
                continue
            diff = bm[m]["qini_coefficient"] - bm[base]["qini_coefficient"]
            test_metrics[m][f"qini_minus_{base}"] = float(point[m]["qini_coefficient"] - point[base]["qini_coefficient"])
            test_metrics[m][f"qini_minus_{base}_ci"] = list(metrics.percentile_ci(diff))
            test_metrics[m][f"p_not_better_than_{base}"] = float(np.mean(diff <= 0))

    curves = {}
    for m in ALL_MODELS:
        c = metrics.curve(scores[m], test.t, y)
        u, q = metrics.on_grid(c, GRID)
        lo, hi = np.quantile(boot["curves"][m], [0.025, 0.975], axis=0)
        curves[m] = {"uplift": u, "qini": q, "uplift_lo": lo, "uplift_hi": hi}

    # Robustness: the pooled file mixes tests with different treatment shares.
    # Re-weight test users by inverse estimated propensity (fitted on the
    # training split) and check that the comparison between models holds.
    w = np.where(test.t == 1, 1 / e_test, 1 / (1 - e_test))
    ipw = {m: metrics.summary(metrics.curve(scores[m], test.t, y, weights=w)) for m in ALL_MODELS}

    return {
        "ipw_robustness": {m: {"qini_coefficient": ipw[m]["qini_coefficient"], "auuc": ipw[m]["auuc"],
                               "ate": ipw[m]["ate"]} for m in ALL_MODELS},
        "validation": val_metrics,
        "selected_model": selected,
        "test": test_metrics,
        "test_ranking_by_qini": sorted(UPLIFT_MODELS + ["response"],
                                       key=lambda m: -point[m]["qini_coefficient"]),
        "deciles": {m: metrics.deciles(scores[m], test.t, y, seed=config.SEED)
                    for m in ALL_MODELS if m != "random"},
        "curves": {"grid": GRID, **curves},
        "n_bootstrap": config.N_BOOTSTRAP,
    }


def _num(v) -> float:
    """JSON stores undefined values (e.g. a rate with no control users) as null."""
    return np.nan if v is None else float(v)


def figures(outcome: str, res: dict) -> None:
    plots.style()
    grid = np.asarray(res["curves"]["grid"])
    scale = 1000  # incremental outcomes per 1,000 users
    unit = "conversions" if outcome == "conversion" else "visits"

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    sel = res["selected_model"]
    order = ["random", "response"] + [m for m in UPLIFT_MODELS if m != sel] + [sel]
    for m in order:
        c = res["curves"][m]
        u = np.asarray(c["uplift"]) * scale
        if m == "random":
            ax.plot(grid, u, color=plots.MUTED, lw=1.5, ls="--", label=LABELS[m])
            continue
        lw = 2.4 if m in (sel, "response") else 1.4
        ax.plot(grid, u, color=plots.MODEL_COLORS[m], lw=lw, label=LABELS[m])
        if m in (sel, "response"):
            ax.fill_between(grid, np.asarray(c["uplift_lo"]) * scale,
                            np.asarray(c["uplift_hi"]) * scale,
                            color=plots.MODEL_COLORS[m], alpha=0.15, lw=0)
    ax.set_xlabel("Fraction of users targeted, highest predicted score first")
    ax.set_ylabel(f"Incremental {unit} per 1,000 users")
    ax.set_title(f"Uplift curves on the test set ({outcome}), 95% bootstrap bands")
    ax.legend(loc="lower right", ncol=2)
    plots.save(fig, config.FIGURES_DIR / f"uplift_curves_{outcome}.png")

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    names = [m for m in res["test_ranking_by_qini"]][::-1]
    vals = [res["test"][m]["qini_coefficient"] * scale for m in names]
    ci = np.array([res["test"][m]["qini_coefficient_ci"] for m in names]) * scale
    colors = [plots.MODEL_COLORS[m] for m in names]
    ax.barh(range(len(names)), vals, color=colors, height=0.6)
    ax.errorbar(vals, range(len(names)), xerr=[np.array(vals) - ci[:, 0], ci[:, 1] - np.array(vals)],
                fmt="none", ecolor=plots.INK_2, capsize=3, lw=1)
    ax.axvline(0, color=plots.AXIS, lw=1)
    ax.set_yticks(range(len(names)), [LABELS[m] for m in names])
    ax.set_xlabel(f"Qini coefficient (x1,000): area above random targeting")
    ax.set_title(f"Test-set Qini coefficient with 95% CI ({outcome})")
    plots.save(fig, config.FIGURES_DIR / f"qini_{outcome}.png")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, m in zip(axes, ("response", sel)):
        d = res["deciles"][m]
        up = np.array([_num(r["uplift"]) for r in d]) * scale
        se = np.array([_num(r["se"]) for r in d]) * scale
        ax.bar(range(1, 11), up, color=plots.MODEL_COLORS[m], width=0.75)
        ax.errorbar(range(1, 11), up, yerr=1.96 * se, fmt="none", ecolor=plots.INK_2, capsize=2, lw=1)
        ax.axhline(0, color=plots.AXIS, lw=1)
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("Score decile (1 = highest)")
        ax.set_title(LABELS[m])
    axes[0].set_ylabel(f"Uplift, extra {unit} per 1,000")
    fig.suptitle(f"Measured uplift by decile on the test set ({outcome})", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    plots.save(fig, config.FIGURES_DIR / f"deciles_{outcome}.png")

    # relative lift by decile of the response model: flat means the ad multiplies
    # each user's baseline rate by a constant, so "who converts" ranks uplift too
    d = res["deciles"]["response"]
    rel = np.array([_num(r["relative_lift"]) for r in d]) * 100
    se = np.array([_num(r["relative_lift_se"]) for r in d]) * 100
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.errorbar(range(1, 11), rel, yerr=1.96 * se, fmt="o", color=plots.SERIES[1], ms=6, capsize=3, lw=1.2)
    ax.axhline(0, color=plots.AXIS, lw=1)
    ax.set_xticks(range(1, 11))
    verb = "convert" if outcome == "conversion" else "visit"
    ax.set_xlabel(f"Decile of the response model's score (1 = most likely to {verb} when treated)")
    ax.set_ylabel("Relative lift (%)")
    ax.set_title(f"Relative lift in the {outcome} rate, treated vs control, by response decile")
    plots.save(fig, config.FIGURES_DIR / f"relative_lift_{outcome}.png")


def main(figures_only: bool = False) -> None:
    if figures_only:  # redraw from results/evaluation.json without touching the data
        out = read_json("evaluation.json")
        for outcome in config.OUTCOMES:
            figures(outcome, out[outcome])
        return
    con = data.connect(read_only=True)
    val, test = data.load_split(con, "val"), data.load_split(con, "test")
    con.close()
    e_test = np.load(config.ARTIFACTS_DIR / "propensity.npz")["test"]
    out = {}
    for outcome in config.OUTCOMES:
        out[outcome] = evaluate_outcome(outcome, val, test, e_test)
    write_json("evaluation.json", out)
    for outcome in config.OUTCOMES:
        figures(outcome, out[outcome])


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--figures-only", action="store_true")
    main(ap.parse_args().figures_only)

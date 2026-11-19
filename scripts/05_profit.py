"""Stage 5: profit of targeting policies, with a sensitivity analysis.

For each ranking the profit-maximising cut-off is chosen on the validation
split and applied unchanged to the test split. Confidence intervals come from
a grouped bootstrap of the test split with the threshold held fixed.
Writes results/profit.json, results/app_data.json and figures/profit_*.png.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from _common import read_json, write_json
from uplift import config, data, metrics, plots, profit
from uplift.learners import LABELS

log = logging.getLogger("profit")

OUTCOME = config.PRIMARY_OUTCOME
APP_GRID = np.linspace(0, 1, 501)


def bootstrap_policy(score, t, y, groups, threshold, cost, value, n_boot, seed):
    rng = np.random.default_rng(seed)
    n_groups = int(groups.max()) + 1
    sel, all_ = [], []
    for _ in range(n_boot):
        w = metrics.group_poisson_weights(groups, n_groups, rng)
        sel.append(profit.evaluate_policy(score, t, y, threshold, cost, value, weights=w)["profit_per_user"])
        all_.append(profit.blanket_policies(t, y, cost, value, weights=w)["treat_all"])
    return np.array(sel), np.array(all_)


def policy_report(name, s_val, s_test, val, test, cost, value, n_boot) -> dict:
    yv, yt = val.y[OUTCOME], test.y[OUTCOME]
    c_val = metrics.curve(s_val, val.t, yv)
    pol = profit.best_policy(s_val, c_val, cost, value)
    res = profit.evaluate_policy(s_test, test.t, yt, pol.threshold, cost, value)
    # the optimum read off the test curve itself: optimistic, shown for contrast only
    c_test = metrics.curve(s_test, test.t, yt)
    p_test = profit.profit_curve(c_test, cost, value)
    out = {
        "model": name,
        "val_fraction": pol.fraction,
        "threshold": pol.threshold,
        "test_fraction": res["fraction"],
        "test_profit_per_user": res["profit_per_user"],
        "test_incremental_per_user": res["incremental_per_user"],
        "in_sample_best_fraction_on_test": float(c_test.x[int(np.argmax(p_test))]),
        "in_sample_best_profit_on_test": float(p_test.max()),
    }
    if n_boot:
        sel, alls = bootstrap_policy(s_test, test.t, yt, test.groups, pol.threshold, cost, value,
                                     n_boot, config.SEED)
        out["test_profit_ci"] = list(metrics.percentile_ci(sel))
        out["gain_vs_treat_all_ci"] = list(metrics.percentile_ci(sel - alls))
        out["p_not_better_than_treat_all"] = float(np.mean(sel - alls <= 0))
    return out


def main(figures_only: bool = False) -> None:
    ev = read_json("evaluation.json")[OUTCOME]
    selected = ev["selected_model"]
    con = data.connect(read_only=True)
    val, test = data.load_split(con, "val"), data.load_split(con, "test")
    con.close()
    s = np.load(config.ARTIFACTS_DIR / f"scores_{OUTCOME}.npz")
    cost, value, P = config.COST_PER_TREATED_USER, config.VALUE_PER_CONVERSION, config.POPULATION
    yt = test.y[OUTCOME]

    blanket = profit.blanket_policies(test.t, yt, cost, value)
    if figures_only:  # redraw from results/profit.json
        pj = read_json("profit.json")
        figures(selected, s, val, test, pj["policies"], blanket, pj["sensitivity"])
        return
    rng = np.random.default_rng(config.SEED)
    n_groups = int(test.groups.max()) + 1
    treat_all_boot = [profit.blanket_policies(test.t, yt, cost, value,
                                              weights=metrics.group_poisson_weights(test.groups, n_groups, rng))["treat_all"]
                      for _ in range(config.N_BOOTSTRAP)]

    scenarios = {}
    for name, sc in config.SCENARIOS.items():
        res = {"cost": sc["cost"], "value": sc["value"],
               "treat_all_per_user": profit.blanket_policies(test.t, yt, sc["cost"], sc["value"])["treat_all"]}
        for m in ("response", selected, "random"):
            res[m] = policy_report(m, s[f"val_{m}"], s[f"test_{m}"], val, test, sc["cost"], sc["value"],
                                   config.N_BOOTSTRAP)
            log.info("%s / %s: %s", name, m, res[m])
        scenarios[name] = res
    base_case = {m: scenarios["base"][m] for m in ("response", selected, "random")}

    # sensitivity: rescale cost and value independently
    sens = []
    for fc in config.SENSITIVITY_FACTORS:
        for fv in config.SENSITIVITY_FACTORS:
            c, v = cost * fc, value * fv
            row = {"cost_factor": fc, "value_factor": fv, "cost": c, "value": v,
                   "treat_all": profit.blanket_policies(test.t, yt, c, v)["treat_all"]}
            for m in (selected, "response"):
                r = policy_report(m, s[f"val_{m}"], s[f"test_{m}"], val, test, c, v, n_boot=0)
                row[m] = {"fraction": r["test_fraction"], "profit_per_user": r["test_profit_per_user"]}
            row["best_blanket"] = max(row["treat_all"], 0.0)
            row["gain_vs_best_blanket"] = row[selected]["profit_per_user"] - row["best_blanket"]
            sens.append(row)

    ate = blanket["treat_all"] / value + cost / value  # recover the test ATE
    write_json("profit.json", {
        "outcome": OUTCOME,
        "selected_model": selected,
        "cost_per_treated_user": cost,
        "value_per_conversion": value,
        "population": P,
        "test_ate": ate,
        "break_even_cost_per_user": value * ate,
        "treat_all_per_user": blanket["treat_all"],
        "treat_all_ci": list(metrics.percentile_ci(np.array(treat_all_boot))),
        "treat_none_per_user": 0.0,
        "policies": base_case,
        "scenarios": scenarios,
        "sensitivity": sens,
        "n_bootstrap": config.N_BOOTSTRAP,
    })

    # data for the app: curves on a fine grid with bootstrap bands, the score
    # threshold at each point, and the feature profile of each decile
    boot = metrics.bootstrap({m: s[f"test_{m}"] for m in (selected, "response")}, test.t, yt,
                             test.groups, n_boot=100, seed=config.SEED, grid=APP_GRID)
    app = {"grid": APP_GRID, "selected_model": selected, "models": {}, "ate": ate,
           "default_cost": cost, "default_value": value}
    for m in (selected, "response"):
        c = metrics.curve(s[f"test_{m}"], test.t, yt)
        u, _ = metrics.on_grid(c, APP_GRID)
        lo, hi = np.quantile(boot["curves"][m], [0.05, 0.95], axis=0)
        st = np.sort(s[f"test_{m}"])[::-1]
        idx = np.clip((APP_GRID * len(st)).astype(int) - 1, 0, len(st) - 1)
        app["models"][m] = {"label": LABELS[m], "uplift": u, "uplift_lo90": lo, "uplift_hi90": hi,
                            "threshold": st[idx]}
    order = metrics.rank_order(s[f"test_{selected}"], seed=config.SEED)
    prof = []
    for d, idx in enumerate(np.array_split(order, 10), start=1):
        prof.append({"decile": d, **{f: float(test.X[idx, i].mean()) for i, f in enumerate(config.FEATURES)}})
    app["decile_profile"] = prof
    app["feature_means"] = {f: float(test.X[:, i].mean()) for i, f in enumerate(config.FEATURES)}
    write_json("app_data.json", app)

    figures(selected, s, val, test, base_case, blanket, sens)


def figures(selected, s, val, test, base_case, blanket, sens) -> None:
    plots.style()
    cost, value, P = config.COST_PER_TREATED_USER, config.VALUE_PER_CONVERSION, config.POPULATION
    yt = test.y[OUTCOME]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for m in ("response", selected):
        c = metrics.curve(s[f"test_{m}"], test.t, yt)
        ax.plot(c.x, profit.profit_curve(c, cost, value, P) / 1000, color=plots.MODEL_COLORS[m],
                lw=2.2, label=LABELS[m])
        r = base_case[m]
        ax.scatter([r["test_fraction"]], [r["test_profit_per_user"] * P / 1000], s=60, zorder=4,
                   color=plots.MODEL_COLORS[m], edgecolor=plots.SURFACE, linewidth=2)
    x = np.array([0, 1])
    ax.plot(x, x * blanket["treat_all"] * P / 1000, color=plots.MUTED, ls="--", lw=1.5,
            label="Random targeting")
    ax.axhline(0, color=plots.AXIS, lw=1)
    ax.scatter([1], [blanket["treat_all"] * P / 1000], s=50, color=plots.INK_2, zorder=4)
    ax.annotate("treat all", (1, blanket["treat_all"] * P / 1000), xytext=(-8, 8),
                textcoords="offset points", ha="right", fontsize=8, color=plots.INK_2)
    ax.set_xlabel("Fraction of users targeted")
    ax.set_ylabel("Incremental profit per 1M users (USD thousands)")
    ax.set_title(f"Profit curves on test (cost \\${cost}/user, \\${value:.0f}/conversion); "
                 "dots: cut-off chosen on validation", fontsize=10)
    ax.legend(loc="lower center")
    plots.save(fig, config.FIGURES_DIR / "profit_curves.png")

    fac = config.SENSITIVITY_FACTORS
    k = len(fac)
    frac = np.zeros((k, k))
    gain = np.zeros((k, k))
    for r in sens:
        i, j = fac.index(r["cost_factor"]), fac.index(r["value_factor"])
        frac[i, j] = r[selected]["fraction"]
        gain[i, j] = r["gain_vs_best_blanket"] * P / 1000
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    seq = LinearSegmentedColormap.from_list("seq", plots.SEQ)
    # gains: blue = targeting earns more, red = it earns less
    div = LinearSegmentedColormap.from_list("div", plots.DIVERGING[::-1])
    labels = [f"{f:g}x" for f in fac]
    for ax, mat, cmap, title, fmt, vlim in (
        (axes[0], frac, seq, "Share of users targeted", "{:.0%}", (0, 1)),
        (axes[1], gain, div, "Gain over best of treat-all / treat-none\n(USD thousands per 1M users)",
         "{:.1f}", (-np.abs(gain).max(), np.abs(gain).max())),
    ):
        im = ax.imshow(mat, cmap=cmap, vmin=vlim[0], vmax=vlim[1], origin="lower")
        for i in range(k):
            for j in range(k):
                v = float(mat[i, j]) if cmap is seq else round(float(mat[i, j]), 1) + 0.0  # no "-0.0"
                dark = (cmap is seq and v > 0.55) or (cmap is div and abs(v) > 0.6 * vlim[1])
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if dark else plots.INK)
        ax.set_xticks(range(k), labels)
        ax.set_yticks(range(k), labels)
        ax.set_xlabel(f"Value per conversion (1x = \\${value:.0f})")
        ax.set_ylabel(f"Cost per targeted user (1x = \\${cost})")
        ax.grid(False)
        ax.set_title(title, fontsize=10)
    fig.suptitle(f"Sensitivity of the {LABELS[selected]} policy (test set)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    plots.save(fig, config.FIGURES_DIR / "profit_sensitivity.png")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--figures-only", action="store_true")
    main(ap.parse_args().figures_only)

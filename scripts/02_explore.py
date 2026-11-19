"""Stage 2: explore the trial with SQL, check the randomisation, estimate the ATE.

Writes results/eda.json, results/randomisation.json and figures/eda_*.png.
Only the training and validation splits are used for the propensity check;
the test split is reserved for stage 4.
"""

import logging

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_auc_score

from _common import write_json
from uplift import causal, config, data, metrics, plots

log = logging.getLogger("explore")

N_PERMUTATIONS = 5
PROPENSITY_TRAIN_ROWS = 1_500_000
PROPENSITY_TEST_ROWS = 500_000


def sql_tables(con) -> dict:
    out = {
        "overview": data.query(con, "02_overview.sql"),
        "file_order": data.query(con, "03_file_order.sql"),
        "duplicates": data.query(con, "04_duplicates.sql"),
        "balance": data.query(con, "05_balance.sql"),
        "ate": data.query(con, "06_ate.sql"),
        "splits": data.query(con, "08_split_summary.sql"),
        "runs": data.query(con, "11_treatment_runs.sql"),
    }
    for outcome in config.OUTCOMES:
        out[f"segments_{outcome}"] = data.query(con, "09_segment_uplift.sql", outcome=outcome)
    return out


def bootstrap_ate(t: np.ndarray, y: np.ndarray, groups: np.ndarray, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    n_groups = int(groups.max()) + 1
    tf, yf = t.astype(np.float64), y.astype(np.float64)
    out = np.empty(n_boot)
    for b in range(n_boot):
        w = metrics.group_poisson_weights(groups, n_groups, rng)
        wt = w * tf
        wc = w - wt
        out[b] = (wt @ yf) / wt.sum() - (wc @ yf) / wc.sum()
    return out


def propensity_check(train: data.Split, val: data.Split) -> dict:
    """Can treatment be predicted from the features?

    Fit on a sample of the training split, score a sample of the validation
    split, and compare the AUC with AUCs obtained after shuffling the
    treatment labels (the null of perfect randomisation).
    """
    rng = np.random.default_rng(config.SEED)
    i = rng.choice(len(train), PROPENSITY_TRAIN_ROWS, replace=False)
    j = rng.choice(len(val), PROPENSITY_TEST_ROWS, replace=False)
    Xa, ta, Xb, tb = train.X[i], train.t[i], val.X[j], val.t[j]
    params = dict(n_estimators=300, learning_rate=0.05, num_leaves=63,
                  min_child_samples=500, verbose=-1, n_jobs=4)

    model = lgb.LGBMClassifier(random_state=config.SEED, **params).fit(Xa, ta)
    p = model.predict_proba(Xb)[:, 1]
    auc = roc_auc_score(tb, p)

    # bootstrap CI of the held-out AUC
    boot = []
    for _ in range(200):
        k = rng.integers(0, len(tb), len(tb))
        boot.append(roc_auc_score(tb[k], p[k]))

    null = []
    for r in range(N_PERMUTATIONS):
        perm = rng.permutation(ta)
        tb_perm = rng.permutation(tb)
        m = lgb.LGBMClassifier(random_state=config.SEED + r, **params).fit(Xa, perm)
        null.append(roc_auc_score(tb_perm, m.predict_proba(Xb)[:, 1]))
        log.info("permutation %d: AUC %.4f", r, null[-1])

    # does the small imbalance move the effect estimate? compare the plain
    # difference in means with an inverse-propensity-weighted estimate on the
    # held-out sample
    out = {}
    yb = {o: val.y[o][j].astype(float) for o in config.OUTCOMES}
    for o, y in yb.items():
        naive = y[tb == 1].mean() - y[tb == 0].mean()
        w1, w0 = tb / p, (1 - tb) / (1 - p)
        ipw = (w1 @ y) / w1.sum() - (w0 @ y) / w0.sum()
        out[o] = {"naive": naive, "ipw": ipw}

    return {
        "auc": auc,
        "auc_ci": metrics.percentile_ci(np.array(boot)),
        "null_aucs": null,
        "propensity_quantiles": dict(zip(["p01", "p10", "p50", "p90", "p99"],
                                         np.quantile(p, [0.01, 0.1, 0.5, 0.9, 0.99]))),
        "share_outside_80_90": float(np.mean((p < 0.80) | (p > 0.90))),
        "ate_naive_vs_ipw_holdout": out,
        "n_fit": PROPENSITY_TRAIN_ROWS,
        "n_eval": PROPENSITY_TEST_ROWS,
        "_p": p,
    }


def figures(tables: dict, prop: dict) -> None:
    plots.style()

    # runs of treated / control users along the file
    runs = tables["runs"]
    fig, ax = plt.subplots(figsize=(8, 3.0))
    for _, r in runs.iterrows():
        color = plots.SERIES[0] if r["treatment"] == 1 else plots.SERIES[1]
        ax.bar(r["first_row"] / 1e6, r["visit_rate"] * 100, width=r["n_users"] / 1e6, align="edge",
               color=color, edgecolor=plots.SURFACE, linewidth=1)
    ax.bar([0], [0], color=plots.SERIES[0], label="treated run")
    ax.bar([0], [0], color=plots.SERIES[1], label="control run")
    ax.set_xlim(0, runs["first_row"].iloc[-1] / 1e6 + runs["n_users"].iloc[-1] / 1e6)
    ax.set_xlabel("Position in the file (millions of rows)")
    ax.set_ylabel("Visit rate (%)")
    ax.set_title(f"The file is {len(runs)} sorted runs of treated or control users, "
                 "with very different base rates", fontsize=10)
    ax.legend(loc="upper left")
    plots.save(fig, config.FIGURES_DIR / "eda_file_order.png")

    # covariate balance
    bal = tables["balance"]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.axvspan(-0.1, 0.1, color=plots.GRID, alpha=0.6, lw=0)
    ax.axvline(0, color=plots.AXIS, lw=1)
    ax.scatter(bal["smd"], bal["feature"], s=36, color=plots.SERIES[0], zorder=3)
    ax.set_xlim(-0.12, 0.12)
    ax.invert_yaxis()
    ax.set_xlabel("Standardised mean difference, treated minus control")
    ax.set_title("Covariate balance (shaded band: |SMD| < 0.1)")
    plots.save(fig, config.FIGURES_DIR / "eda_balance.png")

    # randomisation test
    fig, ax = plt.subplots(figsize=(5.5, 2.4))
    ax.scatter(prop["null_aucs"], [0] * len(prop["null_aucs"]), s=40, color=plots.MUTED,
               label="shuffled treatment labels")
    lo, hi = prop["auc_ci"]
    ax.errorbar([prop["auc"]], [0], xerr=[[prop["auc"] - lo], [hi - prop["auc"]]], fmt="o",
                color=plots.SERIES[1], ms=7, capsize=3, label="real labels (95% CI)")
    ax.set_yticks([])
    ax.set_xlabel("Held-out AUC for predicting treatment from the 12 features")
    ax.set_title("Treatment is almost, but not perfectly, unpredictable")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.45), ncol=2)
    plots.save(fig, config.FIGURES_DIR / "eda_propensity_auc.png")

    # heterogeneity by feature quintile
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, outcome in zip(axes, config.OUTCOMES):
        seg = tables[f"segments_{outcome}"]
        grid = seg.pivot(index="feature", columns="quintile", values="uplift")
        grid = grid.loc[config.FEATURES] * 100
        v = np.nanmax(np.abs(grid.values))
        cmap = LinearSegmentedColormap.from_list("div", plots.DIVERGING)
        im = ax.imshow(grid.values, cmap=cmap, vmin=-v, vmax=v, aspect="auto")
        ax.set_xticks(range(5), [f"Q{q}" for q in range(1, 6)])
        ax.set_yticks(range(12), config.FEATURES)
        ax.grid(False)
        ax.set_title(f"{outcome.capitalize()} uplift (pp) by feature quintile")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    plots.save(fig, config.FIGURES_DIR / "eda_segment_uplift.png")


def main() -> None:
    con = data.connect(read_only=True)
    tables = sql_tables(con)
    full = con.sql("SELECT treatment, conversion, visit, group_key FROM criteo").df()
    con.close()

    groups = pd.factorize(full["group_key"])[0]
    t = full["treatment"].to_numpy(np.int8)
    ate_boot = {}
    for o in config.OUTCOMES:
        samples = bootstrap_ate(t, full[o].to_numpy(np.int8), groups, n_boot=200, seed=config.SEED)
        ate_boot[o] = {"ci_low": metrics.percentile_ci(samples)[0],
                       "ci_high": metrics.percentile_ci(samples)[1], "n_boot": 200}
    del full

    con = data.connect(read_only=True)
    train, val = data.load_split(con, "train"), data.load_split(con, "val")
    con.close()
    prop = propensity_check(train, val)

    # Doubly robust ATE on train + validation (the test split stays untouched)
    X = np.concatenate([train.X, val.X])
    tt = np.concatenate([train.t, val.t])
    ys = {o: np.concatenate([train.y[o], val.y[o]]) for o in config.OUTCOMES}
    g = np.concatenate([train.groups, val.groups + train.groups.max() + 1])
    del train, val
    robust = causal.aipw(X, tt, ys, groups=g, n_folds=2, seed=config.SEED)
    log.info("AIPW: %s", robust)

    ate = tables["ate"].set_index("outcome")
    write_json("eda.json", {
        "n_rows": int(tables["overview"]["n_users"].sum()),
        "overview": tables["overview"].to_dict(orient="records"),
        "file_order": tables["file_order"].to_dict(orient="records"),
        "duplicates": tables["duplicates"].to_dict(orient="records"),
        "rows_in_duplicate_groups": int(tables["duplicates"].query("group_size_capped > 1")["n_rows"].sum()),
        "n_distinct_feature_vectors": int(tables["duplicates"]["n_groups"].sum()),
        "balance": tables["balance"].to_dict(orient="records"),
        "max_abs_smd": float(tables["balance"]["smd"].abs().max()),
        "ate": {o: {**ate.loc[o].to_dict(), "bootstrap": ate_boot[o]} for o in config.OUTCOMES},
        "splits": tables["splits"].to_dict(orient="records"),
        "treatment_runs": tables["runs"].to_dict(orient="records"),
        "n_treatment_runs": int(len(tables["runs"])),
        "segments": {o: tables[f"segments_{o}"].to_dict(orient="records") for o in config.OUTCOMES},
    })
    write_json("randomisation.json", {**{k: v for k, v in prop.items() if not k.startswith("_")},
                                      "aipw_train_val": robust})
    figures(tables, prop)


if __name__ == "__main__":
    main()

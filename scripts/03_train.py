"""Stage 3: fit every learner on the training split, for each outcome.

Each LightGBM component stops early on its own loss on the validation split.
Scores for the validation and test splits are saved to data/artifacts/ so
that stage 4 can evaluate them without refitting. Fit times and tree counts
go to results/training.json.
"""

import argparse
import logging
import time

import numpy as np

from _common import write_json
from uplift import config, data, learners

log = logging.getLogger("train")


def n_trees(model) -> dict:
    out = {}
    for name, part in vars(model).items():
        if hasattr(part, "best_iteration_"):
            out[name] = int(part.best_iteration_ or part.n_estimators)
        elif isinstance(part, learners.TLearner):
            out.update({f"stage1_{k}": v for k, v in n_trees(part).items()})
    return out


def main(outcomes: list[str]) -> None:
    con = data.connect(read_only=True)
    train, val, test = (data.load_split(con, s) for s in ("train", "val", "test"))
    con.close()
    report = {}

    # Propensity model, used only for the robustness checks in stage 4: the
    # pooled file mixes several tests whose treatment shares differ, so the
    # chance of treatment is weakly predictable from the features.
    rng = np.random.default_rng(config.SEED)
    idx = rng.choice(len(train), size=min(3_000_000, len(train)), replace=False)
    start = time.perf_counter()
    prop = learners._fit(learners._classifier(config.SEED, n_estimators=1000), train.X[idx], train.t[idx],
                         eval_set=(val.X, val.t))
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(config.ARTIFACTS_DIR / "propensity.npz",
             val=learners._p1(prop, val.X), test=learners._p1(prop, test.X))
    report["propensity"] = {"fit_seconds": time.perf_counter() - start,
                            "trees": int(prop.best_iteration_ or prop.n_estimators),
                            "rows": len(idx)}
    log.info("propensity model: %s", report["propensity"])

    for outcome in outcomes:
        y_tr, y_va = train.y[outcome], val.y[outcome]
        scores = {}
        report[outcome] = {}
        fitted = {}
        rng = np.random.default_rng(config.SEED)
        scores["val_random"] = rng.random(len(val))
        scores["test_random"] = rng.random(len(test))
        for name, model in learners.make_learners().items():
            if name == "x_learner":
                model.stage1 = fitted["t_learner"]
            start = time.perf_counter()
            model.fit(train.X, train.t, y_tr, val=(val.X, val.t, y_va))
            secs = time.perf_counter() - start
            fitted[name] = model
            scores[f"val_{name}"] = model.predict(val.X)
            scores[f"test_{name}"] = model.predict(test.X)
            report[outcome][name] = {"fit_seconds": secs, "trees": n_trees(model)}
            log.info("%s / %s: %.0f s, trees %s", outcome, name, secs, report[outcome][name]["trees"])
        config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(config.ARTIFACTS_DIR / f"scores_{outcome}.npz", **scores)
    write_json("training.json", {
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "lightgbm_params": learners.LGB_PARAMS,
        "early_stopping_rounds": learners.EARLY_STOPPING,
        "causal_forest": {"max_rows": learners.CausalForest.max_rows,
                          "n_estimators": learners.CausalForest.n_estimators,
                          "min_samples_leaf": learners.CausalForest.min_samples_leaf},
        "models": report,
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcomes", nargs="+", default=config.OUTCOMES)
    main(ap.parse_args().outcomes)

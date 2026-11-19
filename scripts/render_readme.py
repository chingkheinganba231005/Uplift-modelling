"""Fill the README from results/*.json so that every number in it is reproducible.

  python scripts/render_readme.py          # write README.md from docs/README.template.md
  python scripts/render_readme.py --check  # fail if README.md is out of date
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

LABELS = {
    "random": "Random targeting",
    "response": "Response model (who converts / visits)",
    "s_learner": "S-learner",
    "t_learner": "T-learner",
    "x_learner": "X-learner",
    "class_transformation": "Class transformation",
    "causal_forest": "Causal forest",
}
ORDER = ["response", "s_learner", "t_learner", "x_learner", "class_transformation", "causal_forest", "random"]


def prose(label: str) -> str:
    """Label for use inside a sentence: 'Causal forest' -> 'causal forest', 'X-learner' unchanged."""
    return label if "-" in label.split()[0] else label[0].lower() + label[1:]


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def money(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.0f}"


def pct(x: float, d: int = 0) -> str:
    return f"{100 * x:.{d}f}%"


def k3(x: float, d: int = 2) -> str:
    """Per-user rates shown per 1,000 users."""
    return f"{1000 * x:.{d}f}"


def ci3(ci, d: int = 2) -> str:
    return f"[{1000 * ci[0]:.{d}f}, {1000 * ci[1]:.{d}f}]"


def md_table(header, rows) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def context() -> dict:
    eda, rnd, ev, pr, tr = (load(n) for n in
                            ("eda.json", "randomisation.json", "evaluation.json", "profit.json", "training.json"))
    P = pr["population"]
    c = {}
    a = eda["ate"]
    for o in ("conversion", "visit"):
        c[f"ate_{o}"] = k3(a[o]["ate"])
        c[f"ate_{o}_ci"] = ci3([a[o]["ci_low"], a[o]["ci_high"]])
        c[f"ate_{o}_boot_ci"] = ci3([a[o]["bootstrap"]["ci_low"], a[o]["bootstrap"]["ci_high"]])
        c[f"rel_lift_{o}"] = pct(a[o]["relative_lift"])
        c[f"rate_t_{o}"] = pct(a[o]["rate_treated"], 3)
        c[f"rate_c_{o}"] = pct(a[o]["rate_control"], 3)
        aw = rnd["aipw_train_val"][o]
        c[f"aipw_{o}"] = k3(aw["aipw"])
        c[f"aipw_{o}_ci"] = ci3(aw["aipw_ci"])
        c[f"dm_{o}"] = k3(aw["difference_in_means"])
        c[f"dm_{o}_ci"] = ci3(aw["dm_ci"])
        c[f"aipw_shrink_{o}"] = pct(1 - aw["aipw"] / aw["difference_in_means"])
    c["n_rows"] = f"{eda['n_rows']:,}"
    c["n_treated"] = f"{next(r['n_users'] for r in eda['overview'] if r['treatment'] == 1):,}"
    c["n_control"] = f"{next(r['n_users'] for r in eda['overview'] if r['treatment'] == 0):,}"
    c["max_smd"] = f"{eda['max_abs_smd']:.3f}"
    c["dup_rows"] = f"{eda['rows_in_duplicate_groups']:,}"
    c["n_vectors"] = f"{eda['n_distinct_feature_vectors']:,}"
    c["n_runs"] = str(eda["n_treatment_runs"])
    sp = {r["split"]: r for r in eda["splits"]}
    c["n_train"], c["n_val"], c["n_test"] = (f"{sp[s]['n_users']:,}" for s in ("train", "val", "test"))
    c["auc"] = f"{rnd['auc']:.4f}"
    c["n_perm"] = str(len(rnd["null_aucs"]))
    n_all = sum(r["n_users"] for r in eda["overview"])
    c["conv_rate"] = pct(sum(r["conversions"] for r in eda["overview"]) / n_all, 2)
    c["auc_ci"] = f"[{rnd['auc_ci'][0]:.4f}, {rnd['auc_ci'][1]:.4f}]"
    c["null_auc_range"] = f"{min(rnd['null_aucs']):.4f} to {max(rnd['null_aucs']):.4f}"
    q = rnd["propensity_quantiles"]
    c["prop_p01"], c["prop_p99"] = f"{q['p01']:.2f}", f"{q['p99']:.2f}"

    # model comparison
    rows = []
    for m in ORDER:
        cells = [LABELS[m]]
        for o in ("conversion", "visit"):
            t = ev[o]["test"][m]
            cells.append(f"{k3(t['qini_coefficient'])} {ci3(t['qini_coefficient_ci'])}")
        for o in ("conversion", "visit"):
            t = ev[o]["test"][m]
            cells.append("baseline" if m == "response" else
                         f"{1000 * t['qini_minus_response']:+.2f} {ci3(t['qini_minus_response_ci'])}")
        for o in ("conversion", "visit"):
            cells.append(k3(ev[o]["ipw_robustness"][m]["qini_coefficient"]))
        rows.append(cells)
    c["model_table"] = md_table(
        ["Ranking", "Qini, conversion", "Qini, visit", "vs response, conversion", "vs response, visit",
         "IPW Qini, conv.", "IPW Qini, visit"], rows)
    for o in ("conversion", "visit"):
        e = ev[o]
        sel = e["selected_model"]
        c[f"selected_{o}"] = prose(LABELS[sel])
        c[f"best_test_{o}"] = prose(LABELS[e["test_ranking_by_qini"][0]])
        c[f"sel_vs_resp_{o}"] = f"{1000 * e['test'][sel]['qini_minus_response']:+.2f}"
        c[f"sel_vs_resp_{o}_ci"] = ci3(e["test"][sel]["qini_minus_response_ci"])
        diff_ci = e["test"][sel]["qini_minus_response_ci"]
        c[f"sel_trails_resp_{o}"] = f"{-1000 * e['test'][sel]['qini_minus_response']:.2f}"
        c[f"sel_trails_resp_{o}_ci"] = ci3([-diff_ci[1], -diff_ci[0]])
        d = e["deciles"]["response"]
        c[f"resp_top_uplift_{o}"] = k3(d[0]["uplift"], 1)
        c[f"resp_rest_uplift_max_{o}"] = k3(max(r["uplift"] for r in d[1:]), 1)
        c[f"resp_top_rel_{o}"] = pct(d[0]["relative_lift"])
        # deciles have equal size and the same 85/15 split in expectation, so
        # averaging their rates gives the pooled rates of deciles 2-10
        rest_t = sum(r["rate_treated"] for r in d[1:]) / len(d[1:])
        rest_c = sum(r["rate_control"] for r in d[1:]) / len(d[1:])
        c[f"resp_rest_rel_{o}"] = pct(rest_t / rest_c - 1)
    ct = ev["conversion"]["deciles"]["class_transformation"]
    c["ct_bottom_treated"] = pct(ct[-1]["n_treated"] / ct[-1]["n"], 1)
    c["overall_treated"] = pct(sum(r["n_treated"] for r in ct) / sum(r["n"] for r in ct), 1)
    v = ev["visit"]["test"]
    c["visit_gain_pct"] = pct(v[ev["visit"]["selected_model"]]["qini_minus_response"] / v["response"]["qini_coefficient"])

    # money
    for name, sc in pr["scenarios"].items():
        sel = pr["selected_model"]
        c[f"{name}_cost"] = f"${sc['cost']:.3f}"
        c[f"{name}_treat_all"] = money(sc["treat_all_per_user"] * P)
        c[f"{name}_treat_all_loss"] = money(abs(sc["treat_all_per_user"] * P))
        for m, tag in (("response", "resp"), (sel, "sel")):
            r = sc[m]
            c[f"{name}_{tag}_frac"] = pct(r["test_fraction"])
            c[f"{name}_{tag}_profit"] = money(r["test_profit_per_user"] * P)
            c[f"{name}_{tag}_profit_ci"] = f"{money(r['test_profit_ci'][0] * P)} to {money(r['test_profit_ci'][1] * P)}"
            gain = r["test_profit_per_user"] - sc["treat_all_per_user"]
            c[f"{name}_{tag}_gain"] = money(gain * P)
            c[f"{name}_{tag}_gain_ci"] = (f"{money(r['gain_vs_treat_all_ci'][0] * P)} to "
                                          f"{money(r['gain_vs_treat_all_ci'][1] * P)}")
            c[f"{name}_{tag}_spend_saved"] = money((1 - r["test_fraction"]) * sc["cost"] * P)
            c[f"{name}_{tag}_share_incremental"] = pct(r["test_incremental_per_user"] / pr["test_ate"])
            c[f"{name}_{tag}_insample_profit"] = money(r["in_sample_best_profit_on_test"] * P)
    c["break_even"] = f"${pr['break_even_cost_per_user']:.3f}"
    c["cost"] = f"${pr['cost_per_treated_user']:.3f}"
    c["value"] = f"${pr['value_per_conversion']:.0f}"
    c["selected_profit"] = LABELS[pr["selected_model"]]
    c["selected_profit_prose"] = prose(LABELS[pr["selected_model"]])
    c["n_boot"] = str(ev["conversion"]["n_bootstrap"])
    fit = tr["models"]
    c["cf_rows"] = f"{tr['causal_forest']['max_rows']:,}"
    c["train_minutes"] = f"{sum(v['fit_seconds'] for o in ('conversion', 'visit') for v in fit[o].values()) / 60:.0f}"
    return c


def render(template: str, ctx: dict) -> str:
    missing = set(re.findall(r"\{\{(\w+)\}\}", template)) - set(ctx)
    if missing:
        raise KeyError(f"no value for: {sorted(missing)}")
    return re.sub(r"\{\{(\w+)\}\}", lambda m: str(ctx[m.group(1)]), template)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    text = render((ROOT / "docs" / "README.template.md").read_text(), context())
    readme = ROOT / "README.md"
    if ap.parse_args().check:
        if not readme.exists() or readme.read_text() != text:
            sys.exit("README.md is out of date: run python scripts/render_readme.py")
        print("README.md matches results")
        return
    readme.write_text(text)
    print("wrote README.md")


if __name__ == "__main__":
    main()

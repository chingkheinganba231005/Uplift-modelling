"""Targeting planner.

Set the campaign's costs and budget; the app picks the share of users to
target (highest predicted uplift first) that maximises expected incremental
profit, and shows the uncertainty band from the test-set bootstrap.

Run with:  streamlit run app/streamlit_app.py
It only reads results/app_data.json, so the raw dataset is not needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
APP_DATA = ROOT / "results" / "app_data.json"


@st.cache_data
def load() -> dict:
    return json.loads(APP_DATA.read_text())


def usd(x: float) -> str:
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def plan(grid, uplift, cost, value, population, budget):
    """Best fraction to target under a spend limit, and the profit curve."""
    prof = population * (value * uplift - cost * grid)
    max_frac = 1.0 if budget is None or cost == 0 else min(1.0, budget / (cost * population))
    ok = grid <= max_frac + 1e-12
    k = int(np.argmax(np.where(ok, prof, -np.inf)))
    return k, prof


def main() -> None:
    st.set_page_config(page_title="Uplift targeting planner", layout="wide")
    d = load()
    grid = np.array(d["grid"])
    sel = d["selected_model"]
    models = d["models"]

    st.title("Uplift targeting planner")
    st.caption("Criteo uplift trial, test split. Users are ranked by the selected model's predicted "
               "uplift; the curves are measured on held-out randomised data.")

    with st.sidebar:
        st.header("Campaign")
        population = st.number_input("Users you can reach", 10_000, 100_000_000, 1_000_000, step=100_000)
        cost = st.number_input("Cost per targeted user (USD)", 0.0, 1.0, float(d["default_cost"]),
                               step=0.001, format="%.4f")
        value = st.number_input("Value per conversion (USD)", 0.0, 1000.0, float(d["default_value"]), step=5.0)
        use_budget = st.checkbox("Limit spend", value=False)
        budget = st.number_input("Budget (USD)", 0.0, 1e8, 2_000.0, step=500.0) if use_budget else None
        model = st.radio("Ranking", [sel, "response"], format_func=lambda m: models[m]["label"])

    m = models[model]
    u, lo, hi = (np.array(m[k]) for k in ("uplift", "uplift_lo90", "uplift_hi90"))
    k, prof = plan(grid, u, cost, value, population, budget)
    frac = grid[k]
    n_target = int(round(frac * population))
    treat_all = population * (value * u[-1] - cost)
    random_same = frac * treat_all

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Target", f"{frac:.0%} of users", f"{n_target:,} users", delta_color="off")
    c2.metric("Expected incremental profit", usd(prof[k]),
              f"90% band {usd(population * (value * lo[k] - cost * frac))} to "
              f"{usd(population * (value * hi[k] - cost * frac))}", delta_color="off")
    c3.metric("Treat everyone", usd(treat_all))
    c4.metric("Random users, same number", usd(random_same))

    if k == 0:
        st.warning("At these costs no segment is expected to pay back: target nobody.")
    elif frac >= 0.999:
        st.write("Rule: target everyone.")
    elif model == sel:
        thr = m["threshold"][k]
        st.write(f"Rule: target users whose predicted conversion uplift is at least "
                 f"**{thr * 1000:.3f} extra conversions per 1,000 users**.")

    df = pd.DataFrame({
        "share targeted": grid,
        "profit": prof,
        "low": population * (value * lo - cost * grid),
        "high": population * (value * hi - cost * grid),
        "random": grid * treat_all,
    })
    base = alt.Chart(df).encode(x=alt.X("share targeted:Q", axis=alt.Axis(format="%")))
    band = base.mark_area(opacity=0.18, color="#2a78d6").encode(y="low:Q", y2="high:Q")
    line = base.mark_line(color="#2a78d6", strokeWidth=2).encode(
        y=alt.Y("profit:Q", title="Incremental profit (USD)"),
        tooltip=[alt.Tooltip("share targeted:Q", format=".0%"), alt.Tooltip("profit:Q", format=",.0f"),
                 alt.Tooltip("low:Q", format=",.0f"), alt.Tooltip("high:Q", format=",.0f")])
    rnd = base.mark_line(color="#898781", strokeDash=[5, 4], strokeWidth=1.5).encode(y="random:Q")
    pick = alt.Chart(pd.DataFrame({"x": [frac], "y": [prof[k]]})).mark_point(
        size=90, filled=True, color="#0b0b0b").encode(x="x:Q", y="y:Q")
    st.altair_chart((band + line + rnd + pick).properties(height=380), width="stretch")
    st.caption("Blue: profit when targeting the top share by predicted uplift, with its 90% bootstrap band. "
               "Grey dashes: the same number of users picked at random. The best point is read off the "
               "test-set curve, so it is slightly optimistic; the README reports the policy whose cut-off "
               "was chosen on the validation set.")

    if model == sel:
        st.subheader("Who gets targeted")
        prof_df = pd.DataFrame(d["decile_profile"]).set_index("decile")
        means = pd.Series(d["feature_means"])
        rel = prof_df.sub(means, axis=1)
        st.write("Average feature value in each decile of predicted uplift, minus the population "
                 "average. Features are anonymised by Criteo, so they describe the segment but "
                 "cannot be named.")
        st.dataframe(rel.style.format("{:+.2f}").background_gradient(cmap="RdBu_r", axis=None),
                     width="stretch")


if __name__ == "__main__":
    main()

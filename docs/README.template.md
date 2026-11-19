# Uplift Modelling: Who Should See the Ad?

Which users should an advertiser show an ad to, so that incremental profit
is as large as possible? Estimated on the Criteo randomised trial
({{n_rows}} users), evaluated once on {{n_test}} held-out users:

- **Effect of the ad:** conversion rises from {{rate_c_conversion}} to
  {{rate_t_conversion}}, +{{ate_conversion}} conversions per 1,000 users
  (95% CI {{ate_conversion_ci}}). A doubly robust estimate that corrects for
  the trial pooling several tests is {{aipw_shrink_conversion}} smaller.
- **Cheap campaign** ({{cost}} per targeted user, {{value}} per conversion):
  treating everyone earns {{base_treat_all}} per million users. Targeting the
  {{base_resp_frac}} most likely converters earns {{base_resp_gain}} more
  (95% CI {{base_resp_gain_ci}}) and cuts ad spend by {{base_resp_spend_saved}}.
- **Expensive campaign** ({{expensive_cost}} per user, just above the
  {{break_even}} break-even): treating everyone loses
  {{expensive_treat_all_loss}} per million users; targeting the top
  {{expensive_sel_frac}} by a causal forest earns **{{expensive_sel_profit}}**
  (95% CI {{expensive_sel_profit_ci}}).
- **Uplift vs "who converts":** for visits, the best uplift model ranks users
  {{visit_gain_pct}} better (Qini) than a response model, a significant gain;
  for conversions, which are rarer, the response model is as good as any
  uplift model, and the README explains why.

<table><tr>
<td><img src="figures/uplift_curves_visit.png" alt="Uplift curves on the test set for visits"></td>
<td><img src="figures/profit_sensitivity.png" alt="Sensitivity of the targeting policy to cost and value"></td>
</tr></table>

## Problem

An advertiser pays for every user it shows an ad to, but only earns from the
users the ad *changes*: people who convert because of it. Users who would
convert anyway, or never, are wasted spend. The quantity to rank users by is
the individual treatment effect (uplift), not the probability of converting.
This project estimates uplift with five model families on a large randomised
trial, evaluates them honestly on held-out users, and turns the rankings into
a targeting decision with an explicit cost and value.

## Data

The Criteo Uplift Prediction Dataset v2.1 (Diemert et al., 2018): one row per
user from incrementality tests in which a random part of the population was
held out from advertising. {{n_treated}} treated and {{n_control}} control
users, 12 anonymised features, `treatment`, `visit`, `conversion` and
`exposure`. The file is downloaded from the official Hugging Face mirror and
verified against its SHA-256 (`scripts/01_build_db.py`), then loaded into
DuckDB; all exploration is SQL in `sql/`.

What the SQL exploration found, and what it changed:

- **The file is sorted.** It is {{n_runs}} long runs of all-treated or
  all-control users with very different base rates (`sql/11_treatment_runs.sql`).
  A split by position would be meaningless, so the split is a hash.
- **{{dup_rows}} rows share an exact feature vector with another row** (only
  {{n_vectors}} distinct vectors). The split hashes the feature vector, not
  the row, so identical rows never sit on both sides: {{n_train}} train,
  {{n_val}} validation, {{n_test}} test users.
- **`exposure` is recorded after treatment** and is never used as a feature.

![The file is sorted into runs](figures/eda_file_order.png)

## Is it really randomised?

| Check | Result |
|---|---|
| Covariate balance, largest standardised mean difference | {{max_smd}} (threshold 0.1) |
| Held-out AUC of a classifier predicting treatment from the features | {{auc}}, 95% CI {{auc_ci}} |
| Same AUC after shuffling the treatment labels ({{n_perm}} permutations) | {{null_auc_range}} |
| Predicted treatment probability, 1st to 99th percentile | {{prop_p01}} to {{prop_p99}} |

Balance is excellent, but treatment is very slightly predictable: the AUC is
small and far outside the permutation null. That fits the run structure
above: the file pools several tests whose treatment shares differ, so the
features carry a little information about which test, and so about the
chance of treatment. Treatment is random *within* each test, not perfectly
across the pooled file. Two consequences are checked rather than assumed:

| Effect per 1,000 users (train + validation) | Difference in means | Cross-fitted AIPW |
|---|---|---|
| Conversion | {{dm_conversion}} {{dm_conversion_ci}} | {{aipw_conversion}} {{aipw_conversion_ci}} |
| Visit | {{dm_visit}} {{dm_visit_ci}} | {{aipw_visit}} {{aipw_visit_ci}} |

The doubly robust estimate (`src/uplift/causal.py`) adjusts for the
predictable part of treatment. The test-set model comparison is also
repeated with inverse-propensity weights (last two columns of the results
table). The headline ATE on the whole trial, from `sql/06_ate.sql`, is
{{ate_conversion}} extra conversions per 1,000 users (Wald CI
{{ate_conversion_ci}}; grouped bootstrap {{ate_conversion_boot_ci}}), a
relative lift of {{rel_lift_conversion}}, and {{ate_visit}} extra visits per
1,000 ({{ate_visit_ci}}).

## Method

| Model | Idea |
|---|---|
| Response model (baseline) | P(outcome \| x, treated): "who converts" |
| S-learner | One model of y on (x, t); uplift = f(x, 1) - f(x, 0) |
| T-learner | Separate models for treated and control users |
| X-learner | Impute individual effects from the other arm's model, regress them, combine with the propensity |
| Class transformation | One classifier on z = 1 for treated converters and control non-converters, arms reweighted for the 85/15 split |
| Causal forest | econml `CausalForestDML` with the design propensity, fitted on {{cf_rows}} training users |

All base learners are LightGBM with early stopping on the validation split
(`src/uplift/learners.py`). The model used for decisions is chosen on
validation only (largest Qini coefficient); the test split is scored once.

**Metrics** (`src/uplift/metrics.py`). Sort users by predicted uplift. The
uplift curve gives, for each targeted fraction, the extra outcomes per user
of the whole population; the Qini coefficient is the area between the Qini
curve and random targeting (0 = no better than random). 95% intervals come
from a paired Poisson bootstrap ({{n_boot}} replicates) that resamples
groups of identical feature vectors together.

## Results

Test set, Qini coefficient x 1,000 with 95% bootstrap intervals; "vs
response" is the paired difference from the response model.

{{model_table}}

**Visits: uplift is not the same as response.** The {{selected_visit}}
(chosen on validation) beats the response model by
{{sel_vs_resp_visit}} {{sel_vs_resp_visit_ci}}, {{visit_gain_pct}} of the
response model's Qini. The reason shows in the data: among the users the
response model ranks highest, people who visit anyway, the ad raises the visit
rate by only {{resp_top_rel_visit}}, against {{resp_rest_rel_visit}} for
everyone else. Those "sure things" are where the response model wastes
impressions.

**Conversions: here they nearly coincide.** The response model is the best
ranking on test; the {{selected_conversion}} chosen on validation trails it by
{{sel_trails_resp_conversion}} {{sel_trails_resp_conversion_ci}}, a small but
significant gap. Conversions are so
rare ({{conv_rate}} of users) that almost all the incremental ones come from the small group of
users who might convert at all: the response model's top decile gains
{{resp_top_uplift_conversion}} conversions per 1,000 users, and no other
decile more than {{resp_rest_uplift_max_conversion}}. Finding likely converters
and finding persuadable ones is then the same task, and estimating a
difference of two tiny probabilities only adds noise.

**Class transformation is the weakest model** on both outcomes, and on
conversion it is worse than random. Its reweighting assumes the chance of
treatment does not depend on x; in this pooled file it does slightly, and
with rare outcomes the transformed label mostly encodes treatment itself, so
the classifier partly learns the propensity. The test set shows it: the
decile it scores lowest for conversion is {{ct_bottom_treated}} treated,
against {{overall_treated}} overall.

**Robustness.** Under inverse-propensity weighting all Qini values shrink,
consistent with the AIPW estimate of a smaller effect, but the ordering of
the models is the same.

<table><tr>
<td><img src="figures/deciles_conversion.png" alt="Uplift by decile for conversions"></td>
<td><img src="figures/relative_lift_visit.png" alt="Relative lift by response decile for visits"></td>
</tr></table>

## Business impact

Profit of targeting a set of users, relative to targeting nobody, is
`value x incremental conversions - cost x users targeted`, measured on the
randomised test users. The targeting cut-off is chosen on validation and
applied unchanged to test: picking it on the test curve itself would report
the best point of a noisy curve. Figures are per million users.

| Policy | Base case: {{base_cost}} per user | Expensive: {{expensive_cost}} per user |
|---|---|---|
| Treat nobody | $0 | $0 |
| Treat everyone | {{base_treat_all}} | {{expensive_treat_all}} |
| Response model, cut-off from validation | {{base_resp_profit}} ({{base_resp_profit_ci}}), {{base_resp_frac}} targeted | {{expensive_resp_profit}} ({{expensive_resp_profit_ci}}), {{expensive_resp_frac}} targeted |
| {{selected_profit}}, cut-off from validation | {{base_sel_profit}} ({{base_sel_profit_ci}}), {{base_sel_frac}} targeted | {{expensive_sel_profit}} ({{expensive_sel_profit_ci}}), {{expensive_sel_frac}} targeted |

- **Base case.** Treating everyone is profitable because the break-even cost
  is {{break_even}} per user, far above {{base_cost}}. Targeting the top
  {{base_resp_frac}} by the response model keeps {{base_resp_share_incremental}}
  of the incremental conversions, saves {{base_resp_spend_saved}} of ad spend
  and earns {{base_resp_gain}} more than treating everyone (95% CI
  {{base_resp_gain_ci}}). The {{selected_profit_prose}} policy is not significantly
  different from treating everyone ({{base_sel_gain_ci}}).
- **Expensive campaign.** Above break-even, treating everyone loses money and
  the decision rests entirely on the ranking: the {{selected_profit_prose}} targets
  {{expensive_sel_frac}} of users and earns {{expensive_sel_profit}}.
- **Sensitivity** (heatmap above, `results/profit.json`): cost and value are
  each varied from 0.1x to 10x. Targeting adds the most when the cost per user
  is close to value x ATE; when value dominates, treat everyone; when cost
  dominates, target a few per cent or nobody.

An interactive version: `streamlit run app/streamlit_app.py` lets you set the
reachable population, cost, value and an optional budget, and shows the
best share to target, the expected incremental profit with a 90% band, and
the feature profile of each decile.

## Limitations

- Features are anonymised random projections, so the model cannot be
  explained in business terms; the app shows feature profiles only.
- The data are sub-sampled non-uniformly by Criteo for privacy, so absolute
  rates and money are illustrative; the ranking and the decision logic are
  what transfer.
- The pooled file is not perfectly randomised (see above). Model comparisons
  survive inverse-propensity weighting, but the doubly robust effect is
  {{aipw_shrink_conversion}} smaller for conversions and
  {{aipw_shrink_visit}} smaller for visits than the plain difference in
  means. The profit figures use the plain difference within each targeted
  segment, so they are likely optimistic by a similar proportion.
- Treatment is intention to treat: a treated user was eligible, not
  necessarily shown the ad (`exposure`). Costs are charged per targeted user.
- Costs and values are assumptions; the sensitivity analysis shows where
  the conclusions change.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
make install
make all      # download + DuckDB, SQL exploration, training (~{{train_minutes}} min on 4 cores), test evaluation, profit, README
make test     # pytest on a 20,000-row real sample in tests/data
make app
```

Each stage writes `results/*.json` and `figures/`; this README is generated
from those files by `scripts/render_readme.py`, and CI checks that it matches.
Seeds are fixed in `src/uplift/config.py` and dependencies are pinned.

## Data source and licence

Criteo AI Lab, *Criteo Uplift Prediction Dataset* v2.1, CC BY-NC-SA 4.0.
Diemert, Betlei, Renaudin and Amini, "A Large Scale Benchmark for Uplift
Modeling", AdKDD 2018. The 20,000-row sample in `tests/data` is
redistributed under the same licence.

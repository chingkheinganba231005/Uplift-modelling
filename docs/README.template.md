# Uplift Modelling on the Criteo Ad Trial

Showing an ad to someone who was going to buy anyway is wasted money. What
matters is the uplift: how much the ad changes a person's chance of
converting. In this project I estimate uplift for each user in Criteo's
randomised advertising trial ({{n_rows}} users), check the models on users
they never saw, and turn the best ranking into a targeting rule with an
explicit cost per user and value per conversion.

The main results (money and model comparisons are on the {{n_test}} held-out
test users):

- Across the whole trial, the ad raised conversion from {{rate_c_conversion}}
  to {{rate_t_conversion}}, i.e. {{ate_conversion}} extra conversions per 1,000
  users (95% CI {{ate_conversion_ci}}). After adjusting for the way the trial
  was pooled (see below) the effect is about {{aipw_shrink_conversion}} smaller.
- At {{cost}} per user and {{value}} per conversion, showing the ad to everyone
  earns {{base_treat_all}} per million users. Targeting only the
  {{base_resp_frac}} most likely converters earns {{base_resp_gain}} more
  (95% CI {{base_resp_gain_ci}}) and saves {{base_resp_spend_saved}} in ad spend.
- At {{expensive_cost}} per user, just above the {{break_even}} break-even,
  showing the ad to everyone loses {{expensive_treat_all_loss}} per million
  users, while targeting the top {{expensive_sel_frac}} with a causal forest
  earns {{expensive_sel_profit}} (95% CI {{expensive_sel_profit_ci}}).
- For visits, the best uplift model ranked users {{visit_gain_pct}} better than
  a plain "who will visit" model. For conversions no uplift model beat it.

<table><tr>
<td><img src="figures/uplift_curves_visit.png" alt="Uplift curves on the test set for visits"></td>
<td><img src="figures/profit_sensitivity.png" alt="Sensitivity of the targeting policy to cost and value"></td>
</tr></table>

## Data

The [Criteo Uplift Prediction Dataset](https://huggingface.co/datasets/criteo/criteo-uplift)
v2.1 (Diemert et al., 2018) has one row per user from incrementality tests,
where a random group was held out from advertising: {{n_treated}} treated and
{{n_control}} control users, 12 anonymised features, and whether each user
visited the site or converted. `scripts/01_build_db.py` downloads it, checks
its SHA-256 and loads it into DuckDB. All of the exploration is plain SQL in
`sql/`.

Three things came out of the SQL that changed how I set up the rest:

- The file is not shuffled. It is {{n_runs}} long blocks of all-treated or
  all-control users with very different visit rates
  (`sql/11_treatment_runs.sql`), so splitting by row position would be wrong.
  The split uses a hash instead.
- {{dup_rows}} rows share an exact feature vector with another row
  ({{n_vectors}} distinct vectors). I hash the feature vector rather than the
  row, so identical rows never end up in both training and test. Split sizes:
  {{n_train}} train, {{n_val}} validation, {{n_test}} test.
- `exposure` is recorded after treatment, so it is not used as a feature.

![Treated and control blocks in the raw file](figures/eda_file_order.png)

## Checking the randomisation

| Check | Result |
|---|---|
| Largest standardised mean difference between groups | {{max_smd}} (0.1 is the usual limit) |
| Held-out AUC of a classifier predicting treatment from the features | {{auc}}, 95% CI {{auc_ci}} |
| Same AUC with shuffled treatment labels ({{n_perm}} runs) | {{null_auc_range}} |
| Predicted chance of treatment, 1st to 99th percentile | {{prop_p01}} to {{prop_p99}} |

The groups are well balanced, but treatment is slightly predictable from the
features, well outside what shuffled labels give. That fits the block
structure: the file pools several tests that treated different shares of
users, and the features carry a little information about which test a user
came from. I compared the plain difference in means with a cross-fitted
doubly robust (AIPW) estimate, which corrects for this:

| Effect per 1,000 users (train + validation) | Difference in means | AIPW |
|---|---|---|
| Conversion | {{dm_conversion}} {{dm_conversion_ci}} | {{aipw_conversion}} {{aipw_conversion_ci}} |
| Visit | {{dm_visit}} {{dm_visit_ci}} | {{aipw_visit}} {{aipw_visit_ci}} |

The effect on the whole trial from `sql/06_ate.sql` is {{ate_conversion}}
extra conversions per 1,000 users (Wald CI {{ate_conversion_ci}}, grouped
bootstrap {{ate_conversion_boot_ci}}), a {{rel_lift_conversion}} relative
lift, and {{ate_visit}} extra visits per 1,000 ({{ate_visit_ci}}).

## Models

All models use LightGBM with early stopping on the validation split
(`src/uplift/learners.py`):

- a response model, P(convert | treated), as the baseline
- S-learner: one model with treatment as a feature
- T-learner: separate models for treated and control users
- X-learner: imputes each user's effect from the other group's model, then
  regresses on it
- class transformation, with the groups reweighted for the 85/15 split
- causal forest (econml `CausalForestDML`) on {{cf_rows}} training users,
  using the known treatment probability

The model used for decisions is picked on the validation split by Qini
coefficient. The test split is scored once at the end.

To compare rankings I use uplift and Qini curves: sort users by predicted
uplift and measure how many extra conversions the top x% would bring. The
Qini coefficient is the area between the model's curve and random targeting.
Intervals come from a paired bootstrap ({{n_boot}} resamples) that keeps rows
with identical features together.

## Results

Qini coefficient x 1,000 on the test set, with 95% bootstrap intervals. "vs
response" is the paired difference from the response model.

{{model_table}}

For visits, the {{selected_visit}} (chosen on validation) beats the
response model by {{sel_vs_resp_visit}} {{sel_vs_resp_visit_ci}}. The users
the response model ranks highest are mostly people who visit anyway: for
them the ad raises the visit rate by only {{resp_top_rel_visit}}, compared
with {{resp_rest_rel_visit}} for everyone else.

For conversions it goes the other way. The response model is the best
ranking on test and the {{selected_conversion}} trails it by
{{sel_trails_resp_conversion}} {{sel_trails_resp_conversion_ci}}. Only
{{conv_rate}} of users convert, and almost all of the extra conversions come
from the response model's top decile ({{resp_top_uplift_conversion}} per
1,000 users, against at most {{resp_rest_uplift_max_conversion}} in any
other decile). When the people who might convert and the people the ad can
persuade are the same small group, ranking by "who converts" works just as
well, and estimating the difference of two tiny probabilities only adds noise.

Class transformation is the weakest model and is worse than random on
conversions. It assumes the chance of treatment does not depend on the
features, which is not quite true here, and with so few conversions its
target mostly encodes treatment itself. The lowest-scored conversion decile
is {{ct_bottom_treated}} treated against {{overall_treated}} overall, so the
model has partly learned who was treated.

Re-weighting the test users by the inverse of their estimated treatment
probability shrinks all the Qini values (consistent with the smaller AIPW
effect) but keeps the models in the same order.

<table><tr>
<td><img src="figures/deciles_conversion.png" alt="Uplift by decile for conversions"></td>
<td><img src="figures/relative_lift_visit.png" alt="Relative lift by response decile for visits"></td>
</tr></table>

## From scores to a targeting decision

Profit relative to showing the ad to nobody is
`value x extra conversions - cost x users targeted`, measured on the
randomised test users. The cut-off is chosen on the validation split and
then applied to test unchanged. Picking it on the test curve would report
the best point of a noisy curve and overstate the result. Figures are per
million users.

| Policy | {{base_cost}} per user | {{expensive_cost}} per user |
|---|---|---|
| Nobody | $0 | $0 |
| Everyone | {{base_treat_all}} | {{expensive_treat_all}} |
| Response model, cut-off from validation | {{base_resp_profit}} ({{base_resp_profit_ci}}), {{base_resp_frac}} targeted | {{expensive_resp_profit}} ({{expensive_resp_profit_ci}}), {{expensive_resp_frac}} targeted |
| {{selected_profit}}, cut-off from validation | {{base_sel_profit}} ({{base_sel_profit_ci}}), {{base_sel_frac}} targeted | {{expensive_sel_profit}} ({{expensive_sel_profit_ci}}), {{expensive_sel_frac}} targeted |

At {{base_cost}} per user, showing the ad to everyone is already profitable
because the break-even cost is {{break_even}}. Targeting the top
{{base_resp_frac}} by the response model keeps {{base_resp_share_incremental}}
of the extra conversions, saves {{base_resp_spend_saved}} and earns
{{base_resp_gain}} more ({{base_resp_gain_ci}}). The {{selected_profit_prose}}
policy is not clearly different from showing the ad to everyone
({{base_sel_gain_ci}}). Once the cost passes break-even, showing the ad to
everyone loses money and the ranking is what makes the campaign profitable.

The heatmap at the top varies cost and value from 0.1x to 10x of the base
case. Targeting helps most when the cost per user is close to value x effect.

`streamlit run app/streamlit_app.py` opens a small planner: enter the number
of users, the cost, the value and an optional budget, and it shows how many
users to target, the expected profit with a 90% band, and the feature profile
of each decile.

## Limitations

- The features are anonymised, so the model cannot be explained in business
  terms.
- Criteo sub-sampled the data for privacy, so absolute rates and dollar
  figures are only illustrative. The ranking and the decision method are what
  carry over.
- The pooled file is not perfectly randomised. The model comparison holds
  up under re-weighting, but the AIPW effect is {{aipw_shrink_conversion}}
  smaller for conversions and {{aipw_shrink_visit}} smaller for visits. The
  profit figures use the plain difference within each targeted group, so
  they are probably optimistic by a similar amount.
- Treatment means eligible for the ad, not necessarily shown it
  (`exposure`). Costs are charged per targeted user.
- The costs and values are assumptions; the sensitivity grid shows where the
  conclusions change.

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
make install
make all      # download, DuckDB, SQL checks, training (about {{train_minutes}} min on 4 cores), evaluation, profit, README
make test     # tests on a 20,000-row real sample in tests/data
make app
```

Each step writes `results/*.json` and `figures/`. This README is generated
from those files by `scripts/render_readme.py`, and CI fails if it falls out
of date. Seeds are fixed in `src/uplift/config.py` and the dependencies are
pinned.

## Data source and licence

Criteo AI Lab, *Criteo Uplift Prediction Dataset* v2.1, CC BY-NC-SA 4.0.
Diemert, Betlei, Renaudin and Amini, "A Large Scale Benchmark for Uplift
Modeling", AdKDD 2018. The 20,000-row sample in `tests/data` is shared under
the same licence.

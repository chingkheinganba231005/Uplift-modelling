-- Average treatment effect (intention to treat) for each outcome, with a
-- normal-approximation 95% confidence interval for the difference in rates.
WITH long AS (
    UNPIVOT (SELECT treatment, visit, conversion FROM criteo)
    ON visit, conversion
    INTO NAME outcome VALUE y
),
rates AS (
    SELECT
        outcome,
        avg(y) FILTER (WHERE treatment = 1)            AS rate_treated,
        avg(y) FILTER (WHERE treatment = 0)            AS rate_control,
        count(*) FILTER (WHERE treatment = 1)          AS n_treated,
        count(*) FILTER (WHERE treatment = 0)          AS n_control
    FROM long
    GROUP BY outcome
)
SELECT
    outcome,
    n_treated,
    n_control,
    rate_treated,
    rate_control,
    rate_treated - rate_control AS ate,
    sqrt(rate_treated * (1 - rate_treated) / n_treated
         + rate_control * (1 - rate_control) / n_control) AS se,
    rate_treated - rate_control - 1.959964 * sqrt(rate_treated * (1 - rate_treated) / n_treated
         + rate_control * (1 - rate_control) / n_control) AS ci_low,
    rate_treated - rate_control + 1.959964 * sqrt(rate_treated * (1 - rate_treated) / n_treated
         + rate_control * (1 - rate_control) / n_control) AS ci_high,
    rate_treated / rate_control - 1 AS relative_lift
FROM rates
ORDER BY outcome;

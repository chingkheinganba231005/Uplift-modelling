-- Covariate balance: standardised mean difference (SMD) of each feature
-- between treated and control users. |SMD| < 0.1 is the usual threshold for
-- negligible imbalance.
WITH long AS (
    UNPIVOT criteo
    ON f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11
    INTO NAME feature VALUE value
),
by_arm AS (
    SELECT
        feature,
        avg(value) FILTER (WHERE treatment = 1)        AS mean_treated,
        avg(value) FILTER (WHERE treatment = 0)        AS mean_control,
        var_samp(value) FILTER (WHERE treatment = 1)   AS var_treated,
        var_samp(value) FILTER (WHERE treatment = 0)   AS var_control
    FROM long
    GROUP BY feature
)
SELECT
    feature,
    mean_treated,
    mean_control,
    (mean_treated - mean_control) / sqrt((var_treated + var_control) / 2) AS smd,
    sqrt(var_treated / var_control) AS sd_ratio
FROM by_arm
ORDER BY CAST(substr(feature, 2) AS INTEGER);

-- Heterogeneity before any model: treatment effect on the outcome within
-- quintiles of each feature, computed on the training split only.
WITH train AS (
    SELECT c.*
    FROM criteo c
    JOIN splits s USING (row_id)
    WHERE s.split = 'train'
),
long AS (
    UNPIVOT (SELECT treatment, {outcome} AS y, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11 FROM train)
    ON f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11
    INTO NAME feature VALUE value
),
binned AS (
    SELECT
        feature,
        ntile(5) OVER (PARTITION BY feature ORDER BY value) AS quintile,
        treatment,
        y
    FROM long
)
SELECT
    feature,
    quintile,
    count(*)                                   AS n_users,
    avg(y) FILTER (WHERE treatment = 1)        AS rate_treated,
    avg(y) FILTER (WHERE treatment = 0)        AS rate_control,
    avg(y) FILTER (WHERE treatment = 1) - avg(y) FILTER (WHERE treatment = 0) AS uplift
FROM binned
GROUP BY feature, quintile
ORDER BY CAST(substr(feature, 2) AS INTEGER), quintile;

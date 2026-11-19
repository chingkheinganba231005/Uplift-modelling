-- Runs of consecutive rows with the same treatment value, in file order.
-- A shuffled file would have millions of short runs. This file has a handful
-- of long ones whose outcome rates differ sharply, which suggests it is a
-- concatenation of several tests, each stored treated-then-control or
-- control-then-treated.
WITH flagged AS (
    SELECT
        row_id, treatment, visit, conversion,
        CASE WHEN treatment <> lag(treatment) OVER (ORDER BY row_id) THEN 1 ELSE 0 END AS changed
    FROM criteo
),
numbered AS (
    SELECT *, sum(changed) OVER (ORDER BY row_id) AS run
    FROM flagged
)
SELECT
    run,
    min(row_id)          AS first_row,
    count(*)             AS n_users,
    any_value(treatment) AS treatment,
    avg(visit)           AS visit_rate,
    avg(conversion)      AS conversion_rate
FROM numbered
GROUP BY run
ORDER BY run;

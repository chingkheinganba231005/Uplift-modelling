-- How many rows share an identical feature vector, and whether those groups
-- mix treated and control users.
WITH groups AS (
    SELECT
        group_key,
        count(*)       AS group_size,
        sum(treatment) AS n_treated
    FROM criteo
    GROUP BY group_key
)
SELECT
    least(group_size, 5)  AS group_size_capped,
    count(*)              AS n_groups,
    sum(group_size)       AS n_rows,
    avg(CASE WHEN n_treated > 0 AND n_treated < group_size THEN 1 ELSE 0 END) AS share_mixed_arms
FROM groups
GROUP BY group_size_capped
ORDER BY group_size_capped;

-- Check that the split preserved the treatment share and outcome rates.
SELECT
    s.split,
    count(*)            AS n_users,
    avg(c.treatment)    AS treatment_share,
    avg(c.visit)        AS visit_rate,
    avg(c.conversion)   AS conversion_rate,
    count(DISTINCT c.group_key) AS n_groups
FROM criteo c
JOIN splits s USING (row_id)
GROUP BY s.split
ORDER BY CASE s.split WHEN 'train' THEN 0 WHEN 'val' THEN 1 ELSE 2 END;

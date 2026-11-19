-- Treatment share along the file. A randomly ordered file would show about
-- 0.85 in every block; this file is sorted, so a positional split is invalid.
SELECT
    block,
    count(*)        AS n_users,
    avg(treatment)  AS treatment_share,
    avg(visit)      AS visit_rate,
    avg(conversion) AS conversion_rate
FROM (
    SELECT ntile(20) OVER (ORDER BY row_id) AS block, treatment, visit, conversion
    FROM criteo
)
GROUP BY block
ORDER BY block;

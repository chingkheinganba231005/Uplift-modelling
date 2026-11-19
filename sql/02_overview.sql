-- Size of each arm and raw outcome rates.
SELECT
    treatment,
    count(*)                 AS n_users,
    count(*) / sum(count(*)) OVER () AS share,
    sum(visit)               AS visits,
    sum(conversion)          AS conversions,
    avg(visit)               AS visit_rate,
    avg(conversion)          AS conversion_rate,
    avg(exposure)            AS exposure_rate
FROM criteo
GROUP BY treatment
ORDER BY treatment;

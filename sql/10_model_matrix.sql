-- Model matrix for one split. exposure is recorded after treatment and is
-- deliberately excluded: using it would leak the outcome of the campaign.
SELECT
    c.f0, c.f1, c.f2, c.f3, c.f4, c.f5, c.f6, c.f7, c.f8, c.f9, c.f10, c.f11,
    c.treatment,
    c.conversion,
    c.visit,
    c.group_key
FROM criteo c
JOIN splits s USING (row_id)
WHERE s.split = '{split}'
ORDER BY c.row_id;

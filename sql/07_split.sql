-- Train / validation / test split by feature-vector group, so that identical
-- rows never sit on both sides. The assignment is a deterministic hash of the
-- group key and a seed: it does not depend on file order or on DuckDB's
-- internal hash function.
CREATE OR REPLACE TABLE splits AS
SELECT
    row_id,
    CASE
        WHEN bucket < {train_pct} THEN 'train'
        WHEN bucket < {train_pct} + {val_pct} THEN 'val'
        ELSE 'test'
    END AS split
FROM (
    SELECT row_id, md5_number_lower(concat_ws(':', group_key, {seed})) % 100 AS bucket
    FROM criteo
);

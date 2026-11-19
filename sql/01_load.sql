-- Load the raw CSV into DuckDB.
-- row_id records the position in the file. The file is sorted in long blocks
-- of treated and control users (see 03_file_order.sql), so row_id is kept for
-- auditing only and is never used as a feature or for splitting.
-- group_key identifies rows with an identical feature vector; it is used to
-- keep duplicates on the same side of the train/validation/test split.
CREATE OR REPLACE TABLE criteo AS
SELECT
    (row_number() OVER ()) - 1 AS row_id,
    *,
    md5_number_lower(concat_ws(',', f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11)) AS group_key
FROM read_csv(
    '{csv_path}',
    header = true,
    columns = {{
        'f0': 'DOUBLE', 'f1': 'DOUBLE', 'f2': 'DOUBLE', 'f3': 'DOUBLE',
        'f4': 'DOUBLE', 'f5': 'DOUBLE', 'f6': 'DOUBLE', 'f7': 'DOUBLE',
        'f8': 'DOUBLE', 'f9': 'DOUBLE', 'f10': 'DOUBLE', 'f11': 'DOUBLE',
        'treatment': 'TINYINT', 'conversion': 'TINYINT',
        'visit': 'TINYINT', 'exposure': 'TINYINT'
    }}
);

"""Stage 1: download the dataset, verify it and load it into DuckDB."""

import logging

import _common  # noqa: F401
from uplift import config, data

log = logging.getLogger("build_db")


def main() -> None:
    data.download()
    if config.DB_PATH.exists():
        config.DB_PATH.unlink()
    con = data.connect()
    data.build_database(con)
    n = data.scalar(con, "SELECT count(*) FROM criteo")
    if n != config.CSV_ROWS:
        raise RuntimeError(f"expected {config.CSV_ROWS} rows, loaded {n}")
    log.info("loaded %d rows into %s", n, config.DB_PATH)
    for split in ("train", "val", "test"):
        cache = config.ARTIFACTS_DIR / f"matrix_{split}.npz"
        if cache.exists():
            cache.unlink()
        m = data.load_split(con, split)
        log.info("%s: %d rows", split, len(m))
    con.close()


if __name__ == "__main__":
    main()

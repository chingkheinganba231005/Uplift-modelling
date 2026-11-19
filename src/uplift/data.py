"""Download, load into DuckDB and read model matrices."""

from __future__ import annotations

import hashlib
import logging
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(dest: Path = config.CSV_PATH, url: str = config.CSV_URL,
             sha256: str = config.CSV_SHA256) -> Path:
    """Download the dataset once and verify its checksum."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sha256_of(dest) == sha256:
        log.info("dataset already present and verified: %s", dest)
        return dest
    tmp = dest.with_suffix(".part")
    log.info("downloading %s", url)
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    digest = sha256_of(tmp)
    if digest != sha256:
        tmp.unlink()
        raise RuntimeError(f"checksum mismatch: got {digest}, expected {sha256}")
    tmp.rename(dest)
    return dest


def read_sql(name: str, **params) -> str:
    """Read a query from sql/ and fill in its {placeholders}."""
    text = (config.SQL_DIR / name).read_text()
    return text.format(**params) if params else text


def connect(db_path: Path = config.DB_PATH, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(db_path), read_only=read_only)
    con.execute("SET threads = 4").fetchall()
    return con


def build_database(con: duckdb.DuckDBPyConnection, csv_path: Path = config.CSV_PATH,
                   seed: int = config.SEED) -> None:
    """Create the criteo and splits tables."""
    con.execute(read_sql("01_load.sql", csv_path=str(csv_path).replace("'", "''"))).fetchall()
    con.execute(read_sql("07_split.sql", train_pct=config.SPLIT_TRAIN_PCT,
                         val_pct=config.SPLIT_VAL_PCT, seed=int(seed))).fetchall()


def scalar(con: duckdb.DuckDBPyConnection, sql: str):
    """First value of a query, with the result fully consumed."""
    return con.execute(sql).fetchall()[0][0]


def query(con: duckdb.DuckDBPyConnection, name: str, **params) -> pd.DataFrame:
    return con.sql(read_sql(name, **params)).df()


@dataclass
class Split:
    X: np.ndarray          # float32, (n, 12)
    t: np.ndarray          # int8, treatment indicator
    y: dict[str, np.ndarray]  # outcome name -> int8 array
    groups: np.ndarray     # int64 codes of identical feature vectors (for the bootstrap)

    def __len__(self) -> int:
        return len(self.t)


def load_split(con: duckdb.DuckDBPyConnection, split: str) -> Split:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unknown split {split!r}")
    cache = config.ARTIFACTS_DIR / f"matrix_{split}.npz"
    if cache.exists():
        z = np.load(cache)
        return Split(z["X"], z["t"], {o: z[o] for o in config.OUTCOMES}, z["groups"])
    df = con.sql(read_sql("10_model_matrix.sql", split=split)).df()
    out = Split(
        X=df[config.FEATURES].to_numpy(np.float32),
        t=df["treatment"].to_numpy(np.int8),
        y={o: df[o].to_numpy(np.int8) for o in config.OUTCOMES},
        groups=pd.factorize(df["group_key"])[0].astype(np.int64),
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=out.X, t=out.t, groups=out.groups, **out.y)
    return out

import numpy as np
import pandas as pd
import pytest

from conftest import SAMPLE_CSV
from uplift import config, data


@pytest.fixture(scope="module")
def con(tmp_path_factory):
    db = tmp_path_factory.mktemp("db") / "sample.duckdb"
    c = data.connect(db)
    data.build_database(c, csv_path=SAMPLE_CSV, seed=config.SEED)
    yield c
    c.close()


@pytest.fixture(scope="module")
def raw():
    return pd.read_csv(SAMPLE_CSV)


def test_sha256(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    assert data.sha256_of(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_load_keeps_every_row(con, raw):
    assert data.scalar(con, "SELECT count(*) FROM criteo") == len(raw)
    assert data.scalar(con, "SELECT count(*) FROM splits") == len(raw)


def test_ate_sql_matches_pandas(con, raw):
    ate = data.query(con, "06_ate.sql").set_index("outcome")
    for outcome in ("visit", "conversion"):
        expected = raw.loc[raw.treatment == 1, outcome].mean() - raw.loc[raw.treatment == 0, outcome].mean()
        assert ate.loc[outcome, "ate"] == pytest.approx(expected)
        assert ate.loc[outcome, "ci_low"] < ate.loc[outcome, "ate"] < ate.loc[outcome, "ci_high"]


def test_balance_sql_matches_pandas(con, raw):
    bal = data.query(con, "05_balance.sql").set_index("feature")
    t, c = raw[raw.treatment == 1], raw[raw.treatment == 0]
    for f in ("f0", "f7"):
        smd = (t[f].mean() - c[f].mean()) / np.sqrt((t[f].var() + c[f].var()) / 2)
        assert bal.loc[f, "smd"] == pytest.approx(smd)


def test_split_proportions_and_groups(con):
    s = data.query(con, "08_split_summary.sql").set_index("split")
    share = s["n_users"] / s["n_users"].sum()
    assert share["train"] == pytest.approx(0.6, abs=0.02)
    assert share["val"] == pytest.approx(0.2, abs=0.02)
    # identical feature vectors never straddle two splits
    straddling = data.scalar(con, """
        SELECT count(*) FROM (
            SELECT c.group_key FROM criteo c JOIN splits s USING (row_id)
            GROUP BY c.group_key HAVING count(DISTINCT s.split) > 1)
    """)
    assert straddling == 0


def test_split_is_deterministic(tmp_path):
    a = data.connect(tmp_path / "a.duckdb")
    data.build_database(a, csv_path=SAMPLE_CSV, seed=1)
    b = data.connect(tmp_path / "b.duckdb")
    data.build_database(b, csv_path=SAMPLE_CSV, seed=1)
    qa = a.sql("SELECT row_id, split FROM splits ORDER BY row_id").df()
    qb = b.sql("SELECT row_id, split FROM splits ORDER BY row_id").df()
    pd.testing.assert_frame_equal(qa, qb)


def test_model_matrix_excludes_exposure(con, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    sql = data.read_sql("10_model_matrix.sql", split="train")
    assert "exposure" not in sql.split("--")[-1].split("FROM")[0]
    m = data.load_split(con, "train")
    assert m.X.shape[1] == 12 and m.X.dtype == np.float32
    assert set(np.unique(m.t)) <= {0, 1}
    cached = data.load_split(con, "train")  # second call reads the cache
    np.testing.assert_array_equal(m.X, cached.X)

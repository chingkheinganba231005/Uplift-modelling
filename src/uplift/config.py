"""Paths, seeds and business assumptions shared by every stage."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SQL_DIR = ROOT / "sql"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
ARTIFACTS_DIR = DATA_DIR / "artifacts"

CSV_NAME = "criteo-research-uplift-v2.1.csv.gz"
CSV_PATH = RAW_DIR / CSV_NAME
CSV_URL = (
    "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/" + CSV_NAME
)
# SHA-256 of the file as published on the Hugging Face mirror (Git LFS object id).
CSV_SHA256 = "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"
CSV_ROWS = 13_979_592

DB_PATH = DATA_DIR / "criteo.duckdb"

FEATURES = [f"f{i}" for i in range(12)]
OUTCOMES = ["conversion", "visit"]
PRIMARY_OUTCOME = "conversion"

SEED = 20240612
# Share of feature-vector groups (not rows) assigned to each split, in percent.
SPLIT_TRAIN_PCT = 60
SPLIT_VAL_PCT = 20  # remainder is test

# Treatment share in the trial. Used as the propensity score for the X-learner
# and the class-transformation model (the randomisation check justifies this).
PROPENSITY = 0.85

N_BOOTSTRAP = 300

# Business assumptions for the base case (USD). The sensitivity analysis
# multiplies each of them by factors between 0.1 and 10.
COST_PER_TREATED_USER = 0.005
VALUE_PER_CONVERSION = 40.0
SENSITIVITY_FACTORS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
# Population used to express profit in money terms.
POPULATION = 1_000_000

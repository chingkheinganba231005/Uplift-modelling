import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SAMPLE_CSV = Path(__file__).parent / "data" / "criteo_sample.csv.gz"

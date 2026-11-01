"""Shared constants and helpers for the Flight Delay Intelligence experiment pipeline."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys_path_fix = str(ROOT)
import sys
if sys_path_fix not in sys.path:
    sys.path.insert(0, sys_path_fix)

DATA_CSV = ROOT / "flight_data.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
CHART_DIR = ARTIFACTS / "ppt_charts"
METRICS_DIR = ARTIFACTS / "metrics"
MODELS_DIR = ROOT / "models"
WORK_DIR = ARTIFACTS / "work"          # intermediate experiment files (git-ignored)
for d in (PROCESSED_DIR, ARTIFACTS, CHART_DIR, METRICS_DIR, MODELS_DIR, WORK_DIR):
    d.mkdir(parents=True, exist_ok=True)

MODEL_TABLE = PROCESSED_DIR / "model_table.parquet"

SEED = 42
TEST_START = pd.Timestamp("2022-09-26")     # preserved from the original project
FIT_FRACTION = 0.78                         # original: 9,150 fit / 11,744 train = 77.9 %
DELAY_THRESHOLD = 15.0

# Cap applied to the cost-heavy sklearn estimators (documented in the report)
SLOW_MODEL_CAP = 300_000
TUNE_CAP = 400_000


def save_json(obj, path):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if isinstance(o, (pd.Timestamp,)):
            return str(o)
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=conv)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

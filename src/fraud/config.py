import os
import random
from pathlib import Path

import numpy as np
import torch

# paths are resolved relative to the project root so the code behaves identically
# whether it is run as a script, from a notebook, or by the test runner
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_FILE = RAW_DATA_DIR / "creditcard.csv"

KAGGLE_DATASET_PATH = "mlg-ulb/creditcardfraud"
os.environ["KAGGLE_CACHE_HOME"] = str(RAW_DATA_DIR.absolute())

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create output directories on import so downstream code can write without checks
for _d in (PROCESSED_DATA_DIR, FIGURES_DIR, METRICS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42

SPLIT_STRATEGIES = ("stratified-random", "temporal")

COST_FN = 100.0  # cost of a missed fraud (false negative)
COST_FP = 50.0    # cost of a false alarm (false positive)

COST_RATIOS = (2, 5, 10, 50, 100, 1000)

def set_seed(seed: int = SEED, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)

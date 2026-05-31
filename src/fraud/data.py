from pathlib import Path

#import numpy as np
import pandas as pd
#from sklearn.preprocessing import StandardScaler

from fraud.config import RAW_DATA_FILE

V_COLUMNS = [f"V{i}" for i in range(1, 29)]
DAY_SECONDS = 24 * 60 * 60


def load_raw(path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download dataset, rename it to creditcard.csv and place it in "
            f"{path.parent}/."
        )
    return pd.read_csv(path)



from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from fraud.config import KAGGLE_DATASET_PATH, RAW_DATA_FILE

V_COLUMNS = [f"V{i}" for i in range(1, 29)]
DAY_SECONDS = 24 * 60 * 60


def ensure_dataset() -> None:
    if not RAW_DATA_FILE.exists():
        print(f"No dataset found in {RAW_DATA_FILE}")
        print("Downloading automatically via Kaggle CLI...")

        RAW_DATA_FILE.parent.mkdir(exist_ok=True)
        path = kagglehub.dataset_download(KAGGLE_DATASET_PATH)

        csv_file = next(Path(path).glob("*.csv"))
        csv_file.rename(RAW_DATA_FILE)
        print(f"Dataset stored at {RAW_DATA_FILE}")


def load_raw(path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download dataset, rename it to creditcard.csv and place it in "
            f"{path.parent}/."
        )
    return pd.read_csv(path)


def build_torch_data_loader(
    X: np.ndarray, batch_size: int = 32, shuffle: bool = False
) -> DataLoader:
    X_dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    return DataLoader(X_dataset, batch_size=batch_size, shuffle=shuffle)

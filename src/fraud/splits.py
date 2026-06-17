from abc import ABC, abstractmethod

import pandas as pd
from sklearn.model_selection import train_test_split

from fraud.config import SEED


class DataSplitter(ABC):
    """
    Common interface for three-way (train / val / test) partitioning

    A configurable splitter lets us compare resampling criteria and report how
    robust the conclusions are to the partition. Every splitter
    returns (df_train, df_val, df_test); df_train still contains both classes,
    and the caller keeps legitimate-only rows for autoencoder training.
    Validation is used for thresholds/early stopping and test only for the final
    evaluation
    """

    name: str = "DataSplitter"

    @abstractmethod
    def split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Partition ``df`` into (train, val, test)."""


class StratifiedRandomSplitter(DataSplitter):
    """
    Random split stratified by the class label.

    Stratification forces validation and test to keep the real fraud proportion
    (~0.17%), so precision is not artificially inflated. ``seed`` is fixed
    for reproducibility
    """

    name = "stratified-random"

    def __init__(
        self,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        label_col: str = "Class",
        seed: int = SEED,
    ):
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.label_col = label_col
        self.seed = seed

    def split(self, df):
        temp_frac = self.val_frac + self.test_frac
        df_train, df_temp = train_test_split(
            df,
            test_size=temp_frac,
            stratify=df[self.label_col],
            random_state=self.seed,
        )
        # second cut splits the held-out part into val/test, again stratified
        rel_test = self.test_frac / temp_frac
        df_val, df_test = train_test_split(
            df_temp,
            test_size=rel_test,
            stratify=df_temp[self.label_col],
            random_state=self.seed,
        )
        return df_train, df_val, df_test


class TemporalSplitter(DataSplitter):
    """
    Time-ordered split: earliest transactions train, latest test.

    Mimics a realistic deployment (train on the past, predict the future) and
    exposes any non-stationarity in fraud patterns. The fraud proportion in each
    block is the natural rate of that time window, not forced to ~0.17%. This is
    the actual temporal protocol (no resampling); we report the realized rates
    """

    name = "temporal"

    def __init__(
        self,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        time_col: str = "Time",
    ):
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.time_col = time_col

    def split(self, df):
        df_sorted = df.sort_values(self.time_col, kind="stable")
        n = len(df_sorted)
        n_test = int(n * self.test_frac)
        n_val = int(n * self.val_frac)
        n_train = n - n_val - n_test

        df_train = df_sorted.iloc[:n_train]
        df_val = df_sorted.iloc[n_train : n_train + n_val]
        df_test = df_sorted.iloc[n_train + n_val :]
        return df_train, df_val, df_test


_SPLITTERS = {
    StratifiedRandomSplitter.name: StratifiedRandomSplitter,
    TemporalSplitter.name: TemporalSplitter,
}


def get_splitter(name: str, **kwargs) -> DataSplitter:
    if name not in _SPLITTERS:
        raise ValueError(
            f"Unknown split strategy {name!r}. Available: {list(_SPLITTERS)}"
        )
    return _SPLITTERS[name](**kwargs)

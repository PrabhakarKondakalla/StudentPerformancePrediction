from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.common import ensure_directory


@dataclass
class DataIngestionConfig:
    raw_data_path: str
    train_data_path: str
    test_data_path: str
    test_size: float
    random_state: int


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def run(self) -> tuple[str, str, str]:
        raw_path = Path(self.config.raw_data_path)
        if not raw_path.exists():
            raise FileNotFoundError(f"Dataset not found at {raw_path}")

        dataframe = pd.read_csv(raw_path)
        train_df, test_df = train_test_split(
            dataframe,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )

        train_path = Path(self.config.train_data_path)
        test_path = Path(self.config.test_data_path)
        ensure_directory(train_path.parent)
        ensure_directory(test_path.parent)

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        return str(raw_path), str(train_path), str(test_path)

from pathlib import Path
from typing import Any

import pandas as pd

from src.domain.interfaces.dataset_repository import DatasetRepository
from src.infrastructure.config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.infrastructure.config.settings import logger


class FileSystemDatasetRepository(DatasetRepository):
    def __init__(self, raw_dir: Path = RAW_DATA_DIR, processed_dir: Path = PROCESSED_DATA_DIR):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def save_raw(self, library: str, name: str, data: Any) -> None:
        path = self.raw_dir / f"{library}_{name}.csv"
        logger.info(f"Saving raw dataset to {path}")
        data.to_csv(path, index=True)

    def save_raw_combined(self, data: Any) -> None:
        path = self.raw_dir / "dataset.csv"
        logger.info(f"Saving combined raw dataset to {path}")
        data.to_csv(path, index=True)

    def save_processed(self, data: Any) -> None:
        path = self.processed_dir / "dataset.csv"
        logger.info(f"Saving processed dataset to {path}")
        data.to_csv(path, index=True)

    def load_processed(self) -> pd.DataFrame:
        path = self.processed_dir / "dataset.csv"
        logger.info(f"Loading processed dataset from {path}")
        return pd.read_csv(path, index_col=0)

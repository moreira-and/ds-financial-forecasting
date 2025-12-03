import datetime as dt
from typing import Dict

import numpy as np
import pandas as pd

from src.domain.interfaces.dataset_repository import DatasetRepository
from src.domain.interfaces.dataset_service import DatasetService
from src.domain.value_objects.dataset_request import DatasetRequest
from src.infrastructure.adapters.dataset.calendar_enricher import enrich_calendar
from src.infrastructure.adapters.dataset.loading_strategies import (
    DatasetMultiLoader,
    YfinanceLoadingStrategy,
    BcbLoadingStrategy,
    DataReaderLoadingStrategy,
)
from src.infrastructure.config.settings import logger


class MarketDatasetService(DatasetService):
    def __init__(self, repository: DatasetRepository):
        self.repository = repository

    def load_and_prepare(self, request: DatasetRequest) -> pd.DataFrame:
        end_date = dt.datetime.now().date()
        start_date = end_date - dt.timedelta(days=request.years * 365)
        logger.info(f"Requesting information between {start_date} and {end_date}")

        raw_data = self._load_raw_data(start_date, end_date)
        combined = self._persist_and_combine(raw_data)
        logger.success("Raw data successfully loaded...")

        processed = self._process_dataset(combined, request.asset, request.asset_focus)
        processed = enrich_calendar(processed)
        self.repository.save_processed(processed)
        logger.success("Clean data successfully loaded...")
        return processed

    def _load_raw_data(self, start_date, end_date) -> Dict[str, Dict[str, pd.DataFrame]]:
        loaders = DatasetMultiLoader(
            [
                YfinanceLoadingStrategy(start_date, end_date),
                BcbLoadingStrategy(start_date, end_date),
                DataReaderLoadingStrategy(start_date, end_date),
            ]
        )
        return loaders.load()

    def _persist_and_combine(self, raw_data: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
        df_raw = pd.DataFrame()
        for lib, datasets in raw_data.items():
            for name, df in datasets.items():
                self.repository.save_raw(lib, name, df)
                df.columns = ["_".join(map(str, col)).strip() if isinstance(col, tuple) else col for col in df.columns]
                df_raw = pd.concat([df_raw, df], axis=1)
                logger.info(f"Saved {lib}_{name} dataset")
        self.repository.save_raw_combined(df_raw)
        return df_raw

    def _process_dataset(self, df_raw: pd.DataFrame, asset: str, asset_focus: str) -> pd.DataFrame:
        target_cols = [col for col in df_raw.columns if asset in col]
        target_col = [col for col in target_cols if asset_focus in col]
        df_raw = df_raw.dropna(subset=target_col).sort_index()
        df_raw = df_raw.ffill().bfill()
        df_raw = df_raw.pct_change(periods=1, fill_method=None)
        df_raw = df_raw.replace([np.inf, -np.inf], 0).fillna(0)
        return df_raw

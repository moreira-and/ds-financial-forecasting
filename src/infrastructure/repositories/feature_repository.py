from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np

from src.domain.interfaces.feature_repository import FeatureRepository
from src.infrastructure.config.settings import logger, PROCESSED_DATA_DIR


class FileSystemFeatureRepository(FeatureRepository):
    def __init__(self, train_dir: Path = PROCESSED_DATA_DIR, test_dir: Path = None):
        self.train_dir = train_dir
        self.test_dir = test_dir or train_dir
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def save_training_features(self, X_train: Any, y_train: Any) -> None:
        logger.success(f"Saving train features in {self.train_dir}...")
        np.save(self.train_dir / "X_train.npy", X_train)
        np.save(self.train_dir / "y_train.npy", y_train)

    def save_test_features(self, X_test: Any, y_test: Any) -> None:
        logger.success(f"Saving test features in {self.train_dir} and {self.test_dir}...")
        np.save(self.train_dir / "X_test.npy", X_test)
        np.save(self.test_dir / "y_test.npy", y_test)

    def save_preprocessor(self, preprocessor: Any) -> None:
        logger.info("Saving preprocessor...")
        with open(self.train_dir / "preprocessor.pkl", "wb") as f:
            cloudpickle.dump(preprocessor, f)

    def save_postprocessor(self, postprocessor: Any) -> None:
        logger.info("Saving postprocessor...")
        with open(self.train_dir / "postprocessor.pkl", "wb") as f:
            cloudpickle.dump(postprocessor, f)

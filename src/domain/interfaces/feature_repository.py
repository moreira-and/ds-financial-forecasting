from abc import ABC, abstractmethod
from typing import Any


class FeatureRepository(ABC):
    @abstractmethod
    def save_training_features(self, X_train: Any, y_train: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_test_features(self, X_test: Any, y_test: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_preprocessor(self, preprocessor: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_postprocessor(self, postprocessor: Any) -> None:
        raise NotImplementedError


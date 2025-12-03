from abc import ABC, abstractmethod
from typing import Any

from src.domain.value_objects.predict_request import PredictRequest


class PredictionService(ABC):
    @abstractmethod
    def predict(self, request: PredictRequest) -> Any:
        """Run inference using stored artifacts."""
        raise NotImplementedError

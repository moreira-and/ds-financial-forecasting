from abc import ABC, abstractmethod
from typing import Any

from src.domain.value_objects.train_request import TrainRequest


class TrainingService(ABC):
    @abstractmethod
    def train(self, request: TrainRequest) -> Any:
        """Train a model and persist artifacts."""
        raise NotImplementedError

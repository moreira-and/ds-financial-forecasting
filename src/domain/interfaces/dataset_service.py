from abc import ABC, abstractmethod
from typing import Any

from src.domain.value_objects.dataset_request import DatasetRequest


class DatasetService(ABC):
    @abstractmethod
    def load_and_prepare(self, request: DatasetRequest) -> Any:
        """Load, transform, and persist dataset artifacts."""
        raise NotImplementedError

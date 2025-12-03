from abc import ABC, abstractmethod
from typing import Any

from src.domain.value_objects.feature_request import FeatureRequest


class FeatureService(ABC):
    @abstractmethod
    def generate(self, request: FeatureRequest) -> Any:
        """Generate features from a processed dataset."""
        raise NotImplementedError

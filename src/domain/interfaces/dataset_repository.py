from abc import ABC, abstractmethod
from typing import Any, Dict


class DatasetRepository(ABC):
    @abstractmethod
    def save_raw(self, library: str, name: str, data: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_raw_combined(self, data: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_processed(self, data: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_processed(self) -> Any:
        raise NotImplementedError


from src.domain.interfaces.dataset_service import DatasetService
from src.domain.value_objects.dataset_request import DatasetRequest


class PrepareDatasetUseCase:
    def __init__(self, dataset_service: DatasetService):
        self._dataset_service = dataset_service

    def execute(self, request: DatasetRequest):
        return self._dataset_service.load_and_prepare(request)

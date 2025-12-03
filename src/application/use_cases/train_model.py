from src.domain.interfaces.training_service import TrainingService
from src.domain.value_objects.train_request import TrainRequest


class TrainModelUseCase:
    def __init__(self, training_service: TrainingService):
        self._training_service = training_service

    def execute(self, request: TrainRequest):
        return self._training_service.train(request)

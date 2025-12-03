from src.domain.interfaces.prediction_service import PredictionService
from src.domain.value_objects.predict_request import PredictRequest


class PredictModelUseCase:
    def __init__(self, prediction_service: PredictionService):
        self._prediction_service = prediction_service

    def execute(self, request: PredictRequest):
        return self._prediction_service.predict(request)

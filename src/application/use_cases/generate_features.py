from src.domain.interfaces.feature_service import FeatureService
from src.domain.value_objects.feature_request import FeatureRequest


class GenerateFeaturesUseCase:
    def __init__(self, feature_service: FeatureService):
        self._feature_service = feature_service

    def execute(self, request: FeatureRequest):
        return self._feature_service.generate(request)

import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.domain.interfaces.feature_repository import FeatureRepository
from src.domain.interfaces.feature_service import FeatureService
from src.domain.value_objects.feature_request import FeatureRequest
from src.infrastructure.adapters.features.prepare_data_template import DefaultRnnPrepareDataTemplate
from src.infrastructure.adapters.features.splitter_strategy import SequentialSplitter
from src.infrastructure.adapters.features.transform_strategy import DefaultRnnTransformStrategy
from src.infrastructure.adapters.features.generator_strategy import DefaultRnnGenerator
from src.infrastructure.config.settings import logger


class RnnFeatureService(FeatureService):
    def __init__(
        self,
        repository: FeatureRepository,
        transformer=None,
        splitter=None,
        generator=None,
    ):
        self.repository = repository
        self.transformer = transformer or DefaultRnnTransformStrategy()
        self.splitter = splitter
        self.generator = generator

    def generate(self, request: FeatureRequest):
        start_time = time.time()
        logger.info("Generating features from dataset...")

        prepare_data_template = DefaultRnnPrepareDataTemplate(
            dataset=pd.read_csv(request.dataset_path, index_col=0).sort_index(),
            targets=request.targets,
            splitter=self.splitter or SequentialSplitter(train_size_ratio=request.train_size_ratio),
            transformer=self.transformer,
            generator=self.generator or DefaultRnnGenerator(batch_size=request.batch_size, sequence_length=request.sequence_length),
        )

        prepare_data_template.prepare_data()
        X_train, X_test, y_train, y_test = prepare_data_template.get_data()

        self.repository.save_training_features(X_train, y_train)
        self.repository.save_test_features(X_test, y_test)

        preprocessor = prepare_data_template.get_preprocessor()
        self.repository.save_preprocessor(preprocessor)

        postprocessor = prepare_data_template.get_postprocessor()
        self.repository.save_postprocessor(postprocessor)

        elapsed_time = time.time() - start_time
        logger.info(f"Total time taken: {elapsed_time:.2f} seconds")

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

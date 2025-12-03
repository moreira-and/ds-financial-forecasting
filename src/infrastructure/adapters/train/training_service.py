import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.domain.interfaces.training_service import TrainingService
from src.domain.value_objects.train_request import TrainRequest
from src.infrastructure.adapters.log.log_strategy import KerasExperimentMlFlowLogger
from src.infrastructure.adapters.train.callbacks_strategy import RegressionCallbacksStrategy
from src.infrastructure.adapters.train.compile_strategy import RegressionCompileStrategy
from src.infrastructure.adapters.train.model_builder import RegressionRobustModelBuilder, LoadKerasModelBuilder
from src.infrastructure.adapters.train.model_template import ModelKerasPipeline
from src.infrastructure.adapters.train.train_strategy import RegressionTrainStrategy
from src.infrastructure.config.settings import logger, MODELS_DIR, PROCESSED_DATA_DIR


class KerasTrainingService(TrainingService):
    def __init__(self, models_dir: Path = MODELS_DIR, processed_dir: Path = PROCESSED_DATA_DIR):
        self.models_dir = models_dir
        self.processed_dir = processed_dir

    def train(self, request: TrainRequest):
        start_time = time.time()
        logger.info("Loading training dataset...")

        X_train = np.load(request.X_path)
        y_train = np.load(request.y_path)

        input_shape = X_train.shape[1:]
        output_shape = y_train.shape[1:]
        logger.info(f"Input shape: {input_shape}, Output shape: {output_shape}")

        model_builder = self._select_builder(request.model_path, input_shape, output_shape)
        compiler = RegressionCompileStrategy()
        trainer = RegressionTrainStrategy(batch_size=request.batch_size, epochs=request.epochs, validation_len=request.validation_len, callbacks=RegressionCallbacksStrategy.get())

        template = ModelKerasPipeline(model_builder=model_builder, compiler=compiler, trainer=trainer)

        logger.info("Training model...")
        model, history = template.run(X_train, y_train)
        logger.info("Model training complete.")

        model_name = f"{request.model_name}.keras"
        logger.info(f"Saving '{model_name}' in '{self.models_dir}'...")
        model.save(self.models_dir / model_name)

        elapsed_time = time.time() - start_time
        logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")

        logger.info("Logging experiment into mlflow.")
        ml_logger = KerasExperimentMlFlowLogger(model=model, history=history, validation_len=request.validation_len, batch_size=request.batch_size, elapsed_time=elapsed_time)
        ml_logger.run(run_name="training_run", experiment_name=request.experiment_name, model_name="regression-pipeline", purpose_tag="regression-pipeline")

        logger.success("Experiment logged successfully.")
        return model, history

    def _select_builder(self, model_path: Optional[Path], input_shape, output_shape):
        if model_path:
            logger.info(f"Loading model from {model_path}")
            return LoadKerasModelBuilder(model_path=model_path)
        logger.info("Building new model...")
        return RegressionRobustModelBuilder(input_shape=input_shape, output_shape=output_shape)

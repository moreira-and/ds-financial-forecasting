from pathlib import Path

import cloudpickle
import keras
import pandas as pd

from src.domain.interfaces.prediction_service import PredictionService
from src.domain.value_objects.predict_request import PredictRequest
from src.infrastructure.adapters.train.metric_strategy import smape, rmse, r2_score
from src.infrastructure.config.settings import logger


class KerasPredictionService(PredictionService):
    def predict(self, request: PredictRequest):
        logger.info("Performing inference for model...")

        model = keras.models.load_model(request.model_path, custom_objects={"smape": smape, "rmse": rmse, "r2_score": r2_score})

        length = model.input_shape[1]

        df = pd.read_csv(request.input_path, index_col=0, parse_dates=True)
        logger.info(f"Input data shape: {df.tail(length).shape}")

        with open(request.preprocessor_path, "rb") as f:
            preprocessor = cloudpickle.load(f)

        X_processed = preprocessor.transform(df.tail(length + 1))

        predictions = model.predict(X_processed)

        with open(request.postprocessor_path, "rb") as f:
            postprocessor = cloudpickle.load(f)

        df_predicted = postprocessor.inverse_transform(predictions)
        df_predicted["type"] = "Predicted"

        last_index = df.index[-1]
        new_index = last_index + pd.Timedelta(days=1)
        df_predicted.index = [new_index]

        df["type"] = "True"
        df_report = pd.concat([df, df_predicted])
        df_report = df_report.ffill()

        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        df_report.to_csv(request.output_path, index=True)

        logger.success("Inference complete.")
        return df_report

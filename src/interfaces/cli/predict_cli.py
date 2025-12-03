from pathlib import Path
import typer

from src.application.use_cases.predict_model import PredictModelUseCase
from src.domain.value_objects.predict_request import PredictRequest
from src.infrastructure.adapters.predict.prediction_service import KerasPredictionService
from src.infrastructure.config.settings import MODELS_DIR, PROCESSED_DATA_DIR, PREDICTED_DATA_DIR


app = typer.Typer()


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    preprocessor_path: Path = PROCESSED_DATA_DIR / "preprocessor.pkl",
    model_path: Path = MODELS_DIR / "default_model.keras",
    postprocessor_path: Path = PROCESSED_DATA_DIR / "postprocessor.pkl",
    output_path: Path = PREDICTED_DATA_DIR / "dataset_report.csv",
):
    service = KerasPredictionService()
    use_case = PredictModelUseCase(prediction_service=service)
    use_case.execute(
        PredictRequest(
            input_path=input_path,
            preprocessor_path=preprocessor_path,
            model_path=model_path,
            postprocessor_path=postprocessor_path,
            output_path=output_path,
        )
    )


if __name__ == "__main__":
    app()

from pathlib import Path
import typer

from src.application.use_cases.train_model import TrainModelUseCase
from src.domain.value_objects.train_request import TrainRequest
from src.infrastructure.adapters.train.training_service import KerasTrainingService
from src.infrastructure.config.settings import MODELS_DIR, PROCESSED_DATA_DIR


app = typer.Typer()


@app.command()
def main(
    X_path: Path = PROCESSED_DATA_DIR / "X_train.npy",
    y_path: Path = PROCESSED_DATA_DIR / "y_train.npy",
    epochs: int = 256,
    validation_len: int = 64,
    batch_size: int = 32,
    experiment_name: str = "default_experiment",
    model_name: str = "default_model",
    model_path: Path = None,
    optimizer: str = None,
    loss: str = None,
    metrics: str = None,
):
    service = KerasTrainingService()
    use_case = TrainModelUseCase(training_service=service)

    use_case.execute(
        TrainRequest(
            X_path=X_path,
            y_path=y_path,
            epochs=epochs,
            validation_len=validation_len,
            batch_size=batch_size,
            experiment_name=experiment_name,
            model_name=model_name,
            model_path=model_path,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )
    )


if __name__ == "__main__":
    app()

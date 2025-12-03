import time
from pathlib import Path
from typing import List

import typer

from src.application.use_cases.generate_features import GenerateFeaturesUseCase
from src.domain.value_objects.feature_request import FeatureRequest
from src.infrastructure.adapters.features.feature_service import RnnFeatureService
from src.infrastructure.config.settings import logger, PROCESSED_DATA_DIR
from src.infrastructure.repositories.feature_repository import FileSystemFeatureRepository


app = typer.Typer()


@app.command()
def main(
    dataset_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    train_dir: Path = PROCESSED_DATA_DIR,
    test_dir: Path = PROCESSED_DATA_DIR,
    targets: List[str] = typer.Option(["^BVSP"]),
    train_size_ratio: float = 0.95,
    batch_size: int = 1,
    sequence_length: int = 32,
):
    start_time = time.time()
    logger.info("Generating features from dataset...")

    service = RnnFeatureService(repository=FileSystemFeatureRepository(train_dir=train_dir, test_dir=test_dir))
    use_case = GenerateFeaturesUseCase(feature_service=service)

    use_case.execute(
        FeatureRequest(
            dataset_path=dataset_path,
            train_dir=train_dir,
            test_dir=test_dir,
            targets=targets,
            train_size_ratio=train_size_ratio,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
    )

    elapsed_time = time.time() - start_time
    logger.info(f"Total time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    app()

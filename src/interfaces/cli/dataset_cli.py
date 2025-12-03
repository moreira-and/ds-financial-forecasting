import time

import typer

from src.application.use_cases.prepare_dataset import PrepareDatasetUseCase
from src.domain.value_objects.dataset_request import DatasetRequest
from src.infrastructure.adapters.dataset.dataset_service import MarketDatasetService
from src.infrastructure.config.settings import logger
from src.infrastructure.repositories.dataset_repository import FileSystemDatasetRepository


app = typer.Typer()


@app.command()
def main(asset: str = "^BVSP", asset_focus: str = "Close", years: int = 10):
    start_time = time.time()
    logger.info("Starting raw data loading...")

    service = MarketDatasetService(repository=FileSystemDatasetRepository())
    use_case = PrepareDatasetUseCase(dataset_service=service)
    use_case.execute(DatasetRequest(asset=asset, asset_focus=asset_focus, years=years))

    elapsed_time = time.time() - start_time
    logger.info(f"Total time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    app()

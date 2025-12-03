from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class TrainRequest:
    X_path: Path
    y_path: Path
    epochs: int
    validation_len: int
    batch_size: int
    experiment_name: str
    model_name: str
    model_path: Optional[Path] = None
    optimizer: Optional[str] = None
    loss: Optional[str] = None
    metrics: Optional[str] = None


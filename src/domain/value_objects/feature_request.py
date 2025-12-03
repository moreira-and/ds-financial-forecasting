from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class FeatureRequest:
    dataset_path: Path
    train_dir: Path
    test_dir: Path
    targets: List[str]
    train_size_ratio: float
    batch_size: int
    sequence_length: int


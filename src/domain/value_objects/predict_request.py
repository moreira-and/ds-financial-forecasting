from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PredictRequest:
    input_path: Path
    preprocessor_path: Path
    model_path: Path
    postprocessor_path: Path
    output_path: Path


from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigLoader:
    def __init__(self, config_path: str):
        self.path = Path(config_path)
        self.config = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            raise FileNotFoundError(f"Config not found: {self.path}")
        with open(self.path, "r") as f:
            return yaml.safe_load(f)

    def get(self, *keys: str, default=None) -> Any:
        data: Any = self.config
        for key in keys:
            if isinstance(data, dict):
                data = data.get(key, default if key == keys[-1] else {})
            else:
                return default
        return data


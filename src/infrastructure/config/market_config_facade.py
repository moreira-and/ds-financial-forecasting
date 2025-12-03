from pathlib import Path
from typing import Dict, Optional

from src.infrastructure.config.config_loader import ConfigLoader
from src.infrastructure.config.settings import CONFIG_DIR


class MarketConfigFacade:
    def __init__(self, config: Optional[Dict] = None, path: Path = CONFIG_DIR / "dataset.yaml"):
        self._config = config or ConfigLoader(str(path)).config

    @property
    def tickers(self) -> Dict[str, str]:
        return self._config.get("yfinance", {}).get("tickers_code", {})

    @property
    def sgs_codes(self) -> Dict[str, int]:
        return self._config.get("bcb", {}).get("sgs_code", {})

    @property
    def datareader_codes(self) -> Dict[str, str]:
        return self._config.get("DataReader", {}).get("reader_code", {})

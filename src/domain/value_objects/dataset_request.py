from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetRequest:
    asset: str = "^BVSP"
    asset_focus: str = "Close"
    years: int = 10


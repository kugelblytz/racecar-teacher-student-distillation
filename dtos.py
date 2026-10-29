from typing import Dict, List, Optional

from pydantic import BaseModel


class RaceCarPredictRequestDto(BaseModel):
    did_crash: bool
    elapsed_ticks: int
    distance: float
    velocity: Dict[str, float]
    sensors: Dict[str, Optional[float]]


class RaceCarPredictResponseDto(BaseModel):
    actions: List[str]

from pydantic import BaseModel, Field
from typing import Dict


class SegmentAnthropometry(BaseModel):
    segment_com_length: float
    segment_com_percentage: float

class AnthropometricData(BaseModel):
    anthropometric_data: Dict[str, SegmentAnthropometry] = Field(...)
    ## to do: can add validation to check that segments exist in the segment_connections



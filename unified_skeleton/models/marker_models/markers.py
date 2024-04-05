
from pydantic import BaseModel
from typing import List

class Markers(BaseModel):
    markers: List[str]


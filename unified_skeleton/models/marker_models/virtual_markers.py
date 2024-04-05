
from pydantic import BaseModel, validator
from typing import Dict, List

class VirtualMarkers(BaseModel):
    virtual_markers: Dict[str, Dict[str, List]]

    @validator('virtual_markers', each_item=True)
    def validate_virtual_marker(cls, virtual_marker):
        marker_names = virtual_marker.get('marker_names', [])
        marker_weights = virtual_marker.get('marker_weights', [])

        if len(marker_names) != len(marker_weights):
            raise ValueError(f'The number of marker names must match the number of marker weights for {virtual_marker}. Currently there are {len(marker_names)} names and {len(marker_weights)} weights.')

        if not isinstance(marker_names, list) or not all(isinstance(name, str) for name in marker_names):
            raise ValueError(f'Marker names must be a list of strings for {marker_names}.')

        if not isinstance(marker_weights, list) or not all(isinstance(weight, (int, float)) for weight in marker_weights):
            raise ValueError(f'Marker weights must be a list of numbers for {virtual_marker}.')

        weight_sum = sum(marker_weights)
        if not 0.99 <= weight_sum <= 1.01:  # Allowing a tiny bit of floating-point leniency
            raise ValueError(f'Marker weights must sum to approximately 1 for {virtual_marker} Current sum is {weight_sum}.')

        return virtual_marker

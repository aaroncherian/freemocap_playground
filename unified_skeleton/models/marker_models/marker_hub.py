from pydantic import BaseModel, root_validator, Field
from typing import Dict, List, Optional

from .markers import Markers
from .virtual_markers import VirtualMarkers


class MarkerHub(BaseModel):
    original_marker_names: Markers
    virtual_markers: Optional[VirtualMarkers] = None
    _all_markers: List[str] = Field(default_factory=list)

    @root_validator
    def copy_markers_to_all(cls, values):
        """Copy markers to _all_markers at initialization."""
        original_marker_names = values.get('original_marker_names')
        if original_marker_names:
            # Directly initializing _all_markers with a copy of marker_names.markers
            values['_all_markers'] = original_marker_names.markers.copy()
        return values

    def add_virtual_markers(self, virtual_markers_dict: Dict[str, Dict[str, List]]):
        """Add virtual markers and update _all_markers."""
        virtual_markers_model = VirtualMarkers(virtual_markers=virtual_markers_dict)
        self.virtual_markers = virtual_markers_model
        for virtual_marker_name in virtual_markers_model.virtual_markers.keys():
            if virtual_marker_name not in self._all_markers:
                self._all_markers.append(virtual_marker_name)

    @property
    def all_markers(self) -> List[str]:
        """Publicly expose the combined list of markers."""
        return self._all_markers

    @classmethod
    def create(cls, marker_list: List[str]):
        """Class method to create an instance with initial marker names."""
        return cls(original_marker_names=Markers(markers=marker_list))


from typing import Dict, List, Optional
from pydantic import BaseModel
import numpy as np

from .marker_models.marker_hub import MarkerHub
from .joint_hierarchy import JointHierarchy
from .anthropometric_data import AnthropometricData
from .segments import Segment, Segments

class Skeleton(BaseModel):
    markers: MarkerHub
    segments: Optional[Dict[str, Dict[str, str]]] = None
    marker_data: Dict[str, np.ndarray] = {}  
    virtual_marker_data: Dict[str, np.ndarray] = {}
    joint_hierarchy: Optional[Dict[str, List[str]]] = None
    anthropometric_data: Optional[Dict[str, Dict[str, float]]] = None

    class Config:
        arbitrary_types_allowed = True

    def add_segments(self, segment_connections: Dict[str, Dict[str, str]]):
        """
        Adds segment connection data to the skeleton model.

        Parameters:
        - segment_connections: A dictionary where each key is a segment name and its value is a dictionary
          with information about that segment (e.g., 'proximal', 'distal' marker names).
        """
        segments_model = Segments(
        markers=self.markers, 
        segment_connections={name: Segment(**segment) for name, segment in segment_connections.items()}
    )
        self.segments = segments_model.segment_connections

    def add_joint_hierarchy(self, joint_hierarchy: Dict[str, List[str]]):
        """
        Adds joint hierarchy data to the skeleton model.

        Parameters:
        - joint_hierarchy: A dictionary with joint names as keys and lists of connected marker names as values.
        """

        joint_hierarchy_model = JointHierarchy(markers=self.markers, joint_hierarchy=joint_hierarchy)
        self.joint_hierarchy = joint_hierarchy_model.joint_hierarchy

    def add_anthropometric_data(self, anthropometric_data: Dict[str, Dict[str, float]]):
        """
        Adds anthropometric data to the skeleton model.

        Parameters:
        - anthropometric_data: A dictionary containing segment mass percentages.
        """
        anthropometric_data_model = AnthropometricData(anthropometric_data=anthropometric_data)
        self.anthropometric_data = anthropometric_data_model.anthropometric_data

    def integrate_freemocap_3d_data(self, freemocap_3d_data:np.ndarray):
        num_markers_in_data = freemocap_3d_data.shape[1]
        original_marker_names_list = self.markers.original_marker_names.markers
        num_markers_in_model = len(original_marker_names_list)
        
        # if num_markers_in_data != num_markers_in_model:
        #     raise ValueError(
        #         f"The number of markers in the 3D data ({num_markers_in_data}) does not match "
        #         f"the number of markers in the model ({num_markers_in_model})."
        #     )
    
        for i, marker_name in enumerate(original_marker_names_list):
            self.marker_data[marker_name] = freemocap_3d_data[:, i, :]

    def calculate_virtual_markers(self):
        # Check if actual marker data is present
        if not self.marker_data:
            raise ValueError("3d marker data must be integrated before calculating virtual markers. Run `integrate_freemocap_3d_data()` first.")

        # Iterate over the virtual markers and calculate their positions
        for vm_name, vm_info in self.markers.virtual_markers.virtual_markers.items():
            # Initialize an array to hold the computed positions of the virtual marker
            vm_positions = np.zeros((self.marker_data[next(iter(self.marker_data))].shape[0], 3))
            for marker_name, weight in zip(vm_info['marker_names'], vm_info['marker_weights']):
                vm_positions += self.marker_data[marker_name] * weight
            self.virtual_marker_data[vm_name] = vm_positions
        
        self.marker_data.update(self.virtual_marker_data)
    
    def get_segment_markers(self, segment_name: str) -> Dict[str, np.ndarray]:
        """Returns a dictionary with the positions of the proximal and distal markers for a segment."""
        segment = self.segments.get(segment_name)
        if not segment:
            raise ValueError(f"Segment '{segment_name}' is not defined in the skeleton.")

        proximal_marker = self.trajectories.get(segment.proximal)
        distal_marker = self.trajectories.get(segment.distal)

        return {
            'proximal': proximal_marker,
            'distal': distal_marker
        }
    @property
    def trajectories(self):
        return self.marker_data
from typing import Dict, List
from pydantic import BaseModel, root_validator

from marker_models.marker_hub import MarkerHub


class JointHierarchy(BaseModel):
    markers: MarkerHub
    joint_hierarchy: Dict[str, List[str]]

    @root_validator
    def check_that_all_markers_exist(cls, values):
        marker_names = values.get('markers').all_markers
        # virtual_markers = values.get('virtual_markers').virtual_markers
        joint_hierarchy = values.get('joint_hierarchy')

        # virtual_marker_names = set(virtual_markers.keys())

        for joint_name, joint_connections in joint_hierarchy.items():
            if joint_name not in marker_names:
                raise ValueError(f'The joint {joint_name} is not in the list of markers or virtual markers.')
            for connected_marker in joint_connections:
                if connected_marker not in marker_names:
                    raise ValueError(f'The connected marker {connected_marker} for {joint_name} is not in the list of markers or virtual markers.')

        return values
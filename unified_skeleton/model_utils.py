from typing import List, Dict
from models.marker_models.marker_hub import MarkerHub

from models.segments import Segments, Segment
from models.skeleton import Skeleton

def create_marker_hub(marker_list: List[str], virtual_markers: Dict[str, Dict[str, List]] = None) -> MarkerHub:
    """
    Creates a MarkerHub instance from a list of actual marker names and optional virtual markers.

    Parameters:
    - marker_list: A list of strings representing the names of actual markers.
    - virtual_markers: A dictionary defining virtual markers and their related data. Each key is a virtual marker name,
      and its value is another dictionary with 'marker_names' and 'marker_weights' as keys.

    Returns:
    - An instance of MarkerHub populated with actual and, if provided, virtual markers.
    """
    # Create the MarkerHub instance using the provided list of actual markers.
    marker_hub = MarkerHub.create(marker_list=marker_list)
    # If virtual markers are provided, add them to the MarkerHub instance.
    if virtual_markers:
        marker_hub.add_virtual_markers(virtual_markers)
    return marker_hub

def create_skeleton_model(
    actual_markers: List[str], 
    segment_connections: Dict[str, Dict[str, str]], 
    virtual_markers: Dict[str, Dict[str, List]] = None
) -> Skeleton:
    """
    Creates a Skeleton model that includes both actual and optionally virtual markers, as well as segment connections.
    
    Parameters:
    - actual_markers: A list of strings representing the names of actual markers.
    - segment_connections: A dictionary where each key is a segment name and its value is a dictionary
      with information about that segment (e.g., 'proximal', 'distal' marker names).
    - virtual_markers: Optional; a dictionary with information necessary to compute virtual markers.

    Returns:
    - An instance of the Skeleton class that represents the complete skeletal model including markers and segments.
    """
    # Create the MarkerHub which serves as the source of truth for marker data within the Skeleton.
    # Virtual markers are optional and only added if provided.
    marker_hub = create_marker_hub(
        marker_list=actual_markers,
        virtual_markers=virtual_markers
    )
    # Construct the Segments object which includes segment connection information.
    segments = Segments(
        markers=marker_hub, 
        segment_connections={name: Segment(**segment) for name, segment in segment_connections.items()}
    )
    # Create and return the Skeleton instance that brings together the markers and segments into a single model.
    return Skeleton(markers=marker_hub, segments=segments)
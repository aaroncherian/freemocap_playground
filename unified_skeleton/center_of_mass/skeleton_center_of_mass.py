from typing import Dict
import numpy as np
from models.skeleton import Skeleton

from .calculate_segment_center_of_mass import calculate_all_segments_com
from .calculate_total_body_center_of_mass import calculate_total_body_center_of_mass


def get_all_segment_markers(skeleton: Skeleton) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Retrieves the proximal and distal marker positions for all segments within a skeleton.

    Parameters:
    - skeleton: The Skeleton instance containing segment information and marker data.

    Returns:
    - A dictionary where each key is a segment name, and the value is another dictionary
      with 'proximal' and 'distal' keys containing the respective marker positions as numpy arrays.
    """

    # Initialize a dictionary to store the positions of segment markers
    segment_positions = {}

    # Iterate over each segment defined in the skeleton
    for segment_name in skeleton.segments.segment_connections.keys():
        # Retrieve the proximal and distal marker positions for the current segment
        # The get_segment_markers method of the Skeleton class is used for this retrieval
        # which is expected to return a dictionary with 'proximal' and 'distal' marker positions
        segment_positions[segment_name] = skeleton.get_segment_markers(segment_name)
    
    # Return the dictionary containing all the segment marker positions
    return segment_positions

def calculate_center_of_mass_from_skeleton(skeleton: Skeleton) -> np.ndarray:
    """
    Calculates the center of mass of the total body based on segment center of mass positions and anthropometric data.
    
    Parameters:
    - skeleton: The Skeleton instance containing marker data and segment information.
    - anthropometric_data: A dictionary containing segment mass percentages
    """
    segment_3d_positions = get_all_segment_markers(skeleton)
    segment_com_data = calculate_all_segments_com(segment_3d_positions, skeleton.anthropometric_data)
    total_body_com = calculate_total_body_center_of_mass(segment_com_data, skeleton.anthropometric_data)

    return total_body_com

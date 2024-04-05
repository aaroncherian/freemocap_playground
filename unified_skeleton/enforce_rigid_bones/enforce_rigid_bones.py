from typing import Dict, List
import numpy as np
from copy import deepcopy

def enforce_rigid_bones(
    marker_data: Dict[str, np.ndarray], 
    segment_connections: Dict[str, Dict[str, str]], 
    bone_lengths_and_statistics: Dict[str, Dict[str, float]], 
    joint_hierarchy: Dict[str, List[str]]
) -> Dict[str, np.ndarray]:
    """
    Enforces rigid bones by adjusting the distal joints of each segment to match the median length.
    
    Parameters:
    - marker_data: The original marker positions.
    - segment_connections: Information about how segments (bones) are connected.
    - bone_lengths_and_statistics: The desired bone lengths and statistics for each segment.
    - joint_hierarchy: The hierarchy of joints, indicating parent-child relationships.

    Returns:
    - A dictionary of adjusted marker positions.
    """
    rigid_marker_data = deepcopy(marker_data)

    for segment_name, stats in bone_lengths_and_statistics.items():
        desired_length = stats['median']
        lengths = stats['lengths']
        
        segment = segment_connections[segment_name]
        proximal_marker, distal_marker = segment.proximal, segment.distal
        
        for frame_index, current_length in enumerate(lengths):
            if current_length != desired_length:
                proximal_position = marker_data[proximal_marker][frame_index]
                distal_position = marker_data[distal_marker][frame_index]
                direction = distal_position - proximal_position
                direction /= np.linalg.norm(direction)  # Normalize to unit vector
                adjustment = (desired_length - current_length) * direction
                
                rigid_marker_data[distal_marker][frame_index] += adjustment
                
                adjust_children(distal_marker, frame_index, adjustment, rigid_marker_data, joint_hierarchy)
    
    return rigid_marker_data

def adjust_children(
    parent_marker: str, 
    frame_index: int, 
    adjustment: np.ndarray, 
    marker_data: Dict[str, np.ndarray], 
    joint_hierarchy: Dict[str, List[str]]
):
    """
    Recursively adjusts the positions of child markers based on the adjustment of the parent marker.
    """
    if parent_marker in joint_hierarchy:
        for child_marker in joint_hierarchy[parent_marker]:
            marker_data[child_marker][frame_index] += adjustment
            adjust_children(child_marker, frame_index, adjustment, marker_data, joint_hierarchy)

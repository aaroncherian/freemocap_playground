from typing import Dict
import numpy as np


def calculate_total_body_center_of_mass(segment_center_of_mass_data: Dict[str, np.ndarray], anthropometric_data: Dict[str, Dict[str, float]]) -> np.ndarray:
    """
    Calculates the total body center of mass for each frame based on segment COM positions and anthropometric data.
    
    Parameters:
    - segment_com_data: A dictionary with segment names as keys and COM positions as values for each frame.
    - anthropometric_data: A dictionary containing segment mass percentages.
    
    Returns:
    - A numpy array containing the position of the total body center of mass for each frame.
    """
    # Assume all segments have the same number of frames
    num_frames = next(iter(segment_center_of_mass_data.values())).shape[0]
    total_body_com = np.zeros((num_frames, 3))

    for segment_name, segment_info in anthropometric_data.items():
        # Retrieve the COM position for the current segment
        segment_com = segment_center_of_mass_data.get(segment_name)
        # Retrieve the mass percentage for the current segment
        segment_mass_percentage = segment_info.segment_com_percentage

        # Add the weighted segment COM to the total COM for each frame
        total_body_com += segment_com * segment_mass_percentage


    return total_body_com
import numpy as np
from typing import Dict

def calculate_bone_lengths_and_statistics(
    marker_data: Dict[str, np.ndarray], 
    segment_connections: Dict[str, Dict[str, str]]
) -> Dict[str, Dict[str, float]]:
    """
    Calculates bone lengths for each frame and their statistics (median and standard deviation)
    based on marker positions and segment connections.

    Parameters:
    - marker_data: A dictionary containing marker trajectories with marker names as keys and 
      3D positions as values (numpy arrays).
    - segment_connections: A dictionary defining the segments (bones) with segment names as keys
      and dictionaries with 'proximal' and 'distal' markers as values.

    Returns:
    - A dictionary with segment names as keys and dictionaries with lengths, median lengths,
      and standard deviations as values.
    """
    bone_statistics = {}

    for segment_name, segment in segment_connections.items():
        # Retrieve the 3D positions for the proximal and distal markers of the segment
        proximal_pos = marker_data[segment.proximal]
        distal_pos = marker_data[segment.distal]

        # Calculate the lengths of the bone for each frame using Euclidean distance
        lengths = np.linalg.norm(distal_pos - proximal_pos, axis=1)
        # Filter out NaN values, which can occur if marker data is missing
        valid_lengths = lengths[~np.isnan(lengths)]

        # Calculate the median and standard deviation from the valid lengths
        median_length = np.median(valid_lengths)
        stdev_length = np.std(valid_lengths)

        # Store the calculated lengths and statistics in the bone_statistics dictionary
        bone_statistics[segment_name] = {
            'lengths': lengths,  # The raw lengths for each frame
            'median': median_length,  # The median length of the bone
            'stdev': stdev_length  # The standard deviation of the bone lengths
        }

    return bone_statistics
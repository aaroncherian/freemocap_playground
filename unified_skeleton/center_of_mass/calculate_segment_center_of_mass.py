from typing import Dict
import numpy as np


def calculate_all_segments_com(
    segment_positions: Dict[str, Dict[str, np.ndarray]], 
    anthropometric_data: Dict[str, Dict[str, float]]
) -> Dict[str, np.ndarray]:
    """
    Calculates the center of mass (COM) for each segment based on provided segment positions 
    and anthropometric data.

    Parameters:
    - segment_positions: A dictionary where each key is a segment name, and the value is another 
      dictionary with 'proximal' and 'distal' keys containing the respective marker positions 
      as numpy arrays.
    - anthropometric_data: A dictionary containing anthropometric information for each segment.
      Each key is a segment name with a value that is a dictionary, which includes the 
      'segment_com_length' key representing the percentage distance from the proximal marker 
      to the segment's COM.

    Returns:
    - A dictionary where each key is a segment name, and the value is the calculated COM 
      position as a numpy array.
    """

    # Initialize a dictionary to store the calculated COM data for each segment
    segment_com_data = {}

    # Iterate through each segment, using the provided anthropometric data
    for segment_name, segment_info in anthropometric_data.items():
        # Retrieve the proximal and distal positions for the current segment from the segment_positions dictionary
        proximal = segment_positions[segment_name]["proximal"]
        distal = segment_positions[segment_name]["distal"]

        # Retrieve the COM length percentage from the anthropometric data
        com_length = segment_info.segment_com_length

        # Calculate the COM position for the segment based on the proximal and distal positions,
        # and the COM length percentage
        segment_com = proximal + (distal - proximal) * com_length

        # Store the calculated COM position in the segment_com_data dictionary
        segment_com_data[segment_name] = segment_com

    # Return the dictionary containing all calculated segment COM positions
    return segment_com_data



from skellymodels.experimental.model_redo.managers.human import Human
from skellymodels.experimental.model_redo.tracker_info.model_info import ModelInfo
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


path_to_recording = Path(r'D:\2025_03_05_atc_testing\freemocap_data\2025-03-05T16_06_17_gmt-5_atc_trial_3')
# path_to_output_data = path_to_recording / 'output_data'/ 'mediapipe_rigid_bones_3d.npy'
path_to_output_data = path_to_recording/ 'saved_data' /'npy'/ 'body_frame_name_xyz.npy'

model_info = ModelInfo(config_path = Path(__file__).parent.parent/'tracker_info'/'mediapipe_just_body.yaml')
human = Human.from_numpy_array(
    name = 'human', 
    model_info=model_info,
    tracked_points_numpy_array=np.load(path_to_output_data)
)

import numpy as np

def check_all_rigid_segments(segment_data, tolerance=0.001):
    """
    Checks whether all rigid body segments maintain constant length over time.
    
    Parameters:
        segment_data (dict): Dictionary where each key is a segment name, 
                             and each value contains 'proximal' and 'distal' arrays.
        tolerance (float): Allowed variation in segment length (default 1mm, or 0.001m).
    
    Returns:
        dict: A dictionary containing residual errors for each segment.
    """
    results = {}

    for segment_name, segment in segment_data.items():
        proximal = np.array(segment['proximal'])  # Shape: (n_frames, 3)
        distal = np.array(segment['distal'])      # Shape: (n_frames, 3)

        if proximal.shape != distal.shape:
            raise ValueError(f"Mismatch in proximal/distal shape for segment {segment_name}")

        # Compute Euclidean distance for each frame
        segment_lengths = np.linalg.norm(proximal - distal, axis=1)

        # The expected length is the length in the first frame
        expected_length = segment_lengths[0]

        # Compute the deviation from the expected length
        deviations = np.abs(segment_lengths - expected_length)
        max_change = np.max(deviations)
        std_deviation = np.std(deviations)

        # Check if any deviation exceeds the tolerance
        violates_rigidity = np.any(deviations > tolerance)

        results[segment_name] = {
            "expected_length": expected_length,
            "max_change": max_change,
            "std_deviation": std_deviation,
            "violates_rigidity": violates_rigidity
        }

    return results

# Example usage:
results = check_all_rigid_segments(human.body.trajectories['3d_xyz'].segment_data)
print(results)

f = 2
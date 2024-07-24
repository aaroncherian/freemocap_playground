from pathlib import Path
import numpy as np

def filter_by_confidence(raw_data: np.ndarray, confidence_threshold: float) -> np.ndarray:
    """
    Set the X and Y coordinates to NaN if the confidence level is less than or equal to the threshold.
    
    Parameters:
        raw_data (np.ndarray): The raw data array of shape (camera, frame, marker, XYC).
        confidence_threshold (float): The confidence threshold below which the X and Y coordinates will be set to NaN.
    
    Returns:
        np.ndarray: The modified raw data array with X and Y set to NaN where the confidence level is less than or equal to the threshold.
    """
    # Find the indices where confidence is less than or equal to the threshold
    low_confidence_mask = raw_data[..., 2] <= confidence_threshold
    
    # Set X and Y to NaN where confidence is low
    raw_data[..., :2][low_confidence_mask] = np.nan
    
    return raw_data

path_to_recording_session = Path(r'D:\steen_pantsOn_gait_3_cameras_high_net_res')
confidence_threshold = .7


path_to_raw_data = path_to_recording_session / 'output_data' / 'raw_data'/ 'openpose_2dData_numCams_numFrames_numTrackedPoints_pixelXY.npy'

raw_data = np.load(path_to_raw_data)

filtered_data = filter_by_confidence(raw_data, confidence_threshold)

np.save(path_to_raw_data, filtered_data)

f = 2
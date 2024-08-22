from pathlib import Path
import numpy as np
from freemocap_anipose import CameraGroup
from scatter_3d import plot_3d_scatter


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
    raw_data_copy = raw_data.copy()
    low_confidence_mask = raw_data_copy[..., 2] <= confidence_threshold
    
    # Set X and Y to NaN where confidence is low
    raw_data_copy[..., :2][low_confidence_mask] = np.nan
    
    return raw_data_copy

confidence_threshold = .5
path_to_recording_session = Path(r'D:\mdn_treadmill_for_testing')
# path_to_recording_session = Path(r'D:\steen_pantsOn_gait')
path_to_calibration_toml = path_to_recording_session/ 'sesh_2023-05-17_12_49_06_calibration_3_camera_calibration.toml'

path_to_raw_data = path_to_recording_session / 'output_data' / 'raw_data'/ 'openpose_2dData_numCams_numFrames_numTrackedPoints_pixelXY.npy'
raw_2d_data = np.load(path_to_raw_data)[:,:,0:25,:]


number_of_cameras = raw_2d_data.shape[0]
number_of_frames = raw_2d_data.shape[1]
number_of_tracked_points = raw_2d_data.shape[2]

filtered_2d_data = filter_by_confidence(raw_2d_data, confidence_threshold)[:,:,:,:2]
filtered_2d_data_flat = filtered_2d_data.reshape(number_of_cameras, -1, 2)

calibration_object = CameraGroup.load(path_to_calibration_toml)
filtered_3d_data_flat = calibration_object.triangulate(filtered_2d_data_flat, progress=True)
filtered_3d_data = filtered_3d_data_flat.reshape(number_of_frames, number_of_tracked_points, 3)

original_3d_data = np.load(path_to_recording_session / 'output_data' / 'raw_data' / 'openpose_3dData_numFrames_numTrackedPoints_spatialXYZ.npy')[:,0:25,:]
processed_3d_data = np.load(path_to_recording_session/'output_data'/'openpose_body_3d_xyz.npy')
# plot_3d_scatter({'original': original_3d_data, 'filtered': filtered_3d_data})
plot_3d_scatter({'processed': processed_3d_data, 'original': original_3d_data, })

# np.save(path_to_raw_data, filtered_data)

f = 2
from pathlib import Path
import numpy as np

recording_folder_path = Path(r'D:\steen_pantsOn_gait_3_cameras_high_net_res')
# recording_folder_path = Path(r'D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_11_55_05_TF01_flexion_neg_5_6_trial_1')
output_data_folder_path = recording_folder_path / 'output_data'
# data_3d_path = output_data_folder_path / 'openpose_body_3d_xyz.npy'
data_3d_path = output_data_folder_path / 'raw_data'/'openpose_3dData_numFrames_numTrackedPoints_spatialXYZ.npy'

data = np.load(data_3d_path)

f
import socket
from pathlib import Path

import numpy as np
import pandas as pd


from fmc_to_trc_utils.mediapipe_skeleton_builder import mediapipe_indices, slice_mediapipe_data
from fmc_to_trc_utils import good_frame_finder, skeleton_y_up_alignment, create_trc



freemocap_data_folder_path = Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3')

sessionID = 'sesh_2023-05-17_13_48_44_MDN_treadmill_2'
data_array_folder = 'output_data'
array_name = 'mediapipe_body_3d_xyz.npy'

data_array_folder_path = freemocap_data_folder_path / sessionID / data_array_folder
skel3d_data = np.load(data_array_folder_path / array_name)

session_info = {'sessionID':sessionID,'skeleton_type': 'mediapipe'}

good_frame = good_frame_finder.find_good_frame(skel3d_data,mediapipe_indices, .3,debug = True)
y_up_skel_data = skeleton_y_up_alignment.align_skeleton_with_origin(session_info, data_array_folder_path, skel3d_data, mediapipe_indices, good_frame, debug = False)

skel_body_points = slice_mediapipe_data(y_up_skel_data,len(mediapipe_indices))
skel_3d_flat = create_trc.flatten_mediapipe_data(skel_body_points)
skel_3d_flat_dataframe = pd.DataFrame(skel_3d_flat)

create_trc.create_trajectory_trc(skel_3d_flat_dataframe,mediapipe_indices, 30, data_array_folder_path)
f = 2
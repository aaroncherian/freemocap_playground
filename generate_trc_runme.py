import socket
from pathlib import Path

import numpy as np
import pandas as pd

from fmc_validation_toolbox.mediapipe_skeleton_builder import mediapipe_indices, slice_mediapipe_data
from fmc_validation_toolbox import good_frame_finder, skeleton_y_up_alignment, create_trc  


this_computer_name = socket.gethostname()

if this_computer_name == 'DESKTOP-F5LCT4Q':
    #freemocap_validation_data_path = Path(r"C:\Users\aaron\Documents\HumonLab\Spring2022\ValidationStudy\FreeMocap_Data")
    #freemocap_data_folder_path = Path(r'D:\freemocap2022\FreeMocap_Data')
    freemocap_data_folder_path = Path(r'D:\ValidationStudy2022\FreeMocap_Data')
else:
    freemocap_data_folder_path = Path(r'C:\Users\Aaron\Documents\sessions\FreeMocap_Data')

#sessionID = 'sesh_2022-05-12_15_13_02'  
#sessionID = 'sesh_2022-06-28_12_55_34'

sessionID = 'sesh_2022-05-24_16_10_46_JSM_T1_WalkRun'
data_array_folder = 'DataArrays'
array_name = 'mediaPipeSkel_3d_filtered.npy'

data_array_folder_path = freemocap_data_folder_path / sessionID / data_array_folder
skel3d_data = np.load(data_array_folder_path / array_name)

session_info = {'sessionID':sessionID,'skeleton_type': 'mediapipe'}

good_frame = good_frame_finder.find_good_frame(session_info,skel3d_data, initial_velocity_guess=.2, debug = False)
y_up_skel_data = skeleton_y_up_alignment.align_skeleton_with_origin(session_info, data_array_folder_path, skel3d_data, mediapipe_indices, good_frame, debug = True)

skel_body_points = slice_mediapipe_data(y_up_skel_data,len(mediapipe_indices))
skel_3d_flat = create_trc.flatten_mediapipe_data(skel_body_points)
skel_3d_flat_dataframe = pd.DataFrame(skel_3d_flat)

create_trc.create_trajectory_trc(skel_3d_flat_dataframe,mediapipe_indices, 30, data_array_folder_path)
f = 2
import socket
from pathlib import Path

import numpy as np
import pandas as pd

from fmc_validation_toolbox.mediapipe_skeleton_builder import mediapipe_indices, slice_mediapipe_data
from fmc_validation_toolbox import good_frame_finder, skeleton_y_up_alignment, create_trc,skeleton_filtering, skeleton_interpolation  



this_computer_name = socket.gethostname()

if this_computer_name == 'DESKTOP-F5LCT4Q':
    #freemocap_validation_data_path = Path(r"C:\Users\aaron\Documents\HumonLab\Spring2022\ValidationStudy\FreeMocap_Data")
    #freemocap_data_folder_path = Path(r'D:\freemocap2022\FreeMocap_Data')
    freemocap_data_folder_path = Path(r'D:\ValidationStudy2022\FreeMocap_Data')
else:
    freemocap_data_folder_path = Path(r'C:\Users\Aaron\Documents\sessions\FreeMocap_Data')

#sessionID = 'sesh_2022-05-12_15_13_02'  
#sessionID = 'sesh_2022-06-28_12_55_34'

sessionID = 'sesh_2022-05-24_15_55_40_JSM_T1_BOS'
data_array_folder = 'DataArrays'
array_name = 'mediaPipeSkel_3d.npy'

session_info = {'sessionID':sessionID,'skeleton_type': 'mediapipe'}

data_array_folder_path = freemocap_data_folder_path / sessionID / data_array_folder
skel3d_raw_data = np.load(data_array_folder_path / array_name)

## Filtering and saving a filtered npy file
sampling_rate = 30
cutoff = 7
order = 4

skel_3d_interpolated = skeleton_interpolation.interpolate_skeleton(skel3d_raw_data)
skel_3d_filtered = skeleton_filtering.filter_skeleton(skel_3d_interpolated,cutoff,sampling_rate,order)
np.save(data_array_folder_path/'mediaPipeSkel_3d_filtered.npy', skel_3d_filtered)

## Aligning data to be y-up
good_frame = good_frame_finder.find_good_frame(session_info,skel_3d_filtered, initial_velocity_guess=.2, debug = False)
y_up_skel_data = skeleton_y_up_alignment.align_skeleton_with_origin(session_info, data_array_folder_path, skel_3d_filtered, mediapipe_indices, good_frame, debug = True)

## Generating a trc file
skel_body_points = slice_mediapipe_data(y_up_skel_data,len(mediapipe_indices))
skel_3d_flat = create_trc.flatten_mediapipe_data(skel_body_points)
skel_3d_flat_dataframe = pd.DataFrame(skel_3d_flat)

create_trc.create_trajectory_trc(skel_3d_flat_dataframe,mediapipe_indices, 30, data_array_folder_path)
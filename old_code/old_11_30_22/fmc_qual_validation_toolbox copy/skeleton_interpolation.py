
import socket

from pathlib import Path

import numpy as np      
import pandas as pd
from rich.progress import track


def interpolate_skeleton(skeleton_3d_data):
    """ Takes in a 3d skeleton numpy array from freemocap and interpolates missing NaN values"""
    num_frames = skeleton_3d_data.shape[0]
    num_markers = skeleton_3d_data.shape[1]

    skel_3d_interpolated = np.empty((num_frames, num_markers, 3))

    for marker in track(range(num_markers)):
        this_marker_skel3d_data = skeleton_3d_data[:,marker,:]
        df = pd.DataFrame(this_marker_skel3d_data)
        df2 = df.interpolate(method = 'linear',axis = 0) #use pandas interpolation methods to fill in missing data
        this_marker_interpolated_skel3d_array = np.array(df2)
        #replace the remaining NaN values (the ones that often happen at the start of the recording)
        this_marker_interpolated_skel3d_array = np.where(np.isfinite(this_marker_interpolated_skel3d_array), this_marker_interpolated_skel3d_array, np.nanmean(this_marker_interpolated_skel3d_array))
        
        skel_3d_interpolated[:,marker,:] = this_marker_interpolated_skel3d_array
        
    return skel_3d_interpolated





if __name__ == '__main__':
    
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

    data_array_folder_path = freemocap_data_folder_path / sessionID / data_array_folder



    skel3d_data = np.load(data_array_folder_path / array_name)
    skel_3d_interpolated = interpolate_skeleton(skel3d_data)


    f= 2




import numpy as np      
import pandas as pd
from rich.progress import track


def interpolate_freemocap_data(freemocap_marker_data:np.ndarray) -> np.ndarray:
    """ Takes in a 3d skeleton numpy array from freemocap and interpolates missing NaN values"""
    num_frames = freemocap_marker_data.shape[0]
    num_markers = freemocap_marker_data.shape[1]

    freemocap_interpolated_data = np.empty((num_frames, num_markers, 3))

    for marker in track(range(num_markers), description= 'Interpolating Data'):
        this_marker_skel3d_data = freemocap_marker_data[:,marker,:]
        df = pd.DataFrame(this_marker_skel3d_data)
        df2 = df.interpolate(method = 'linear',axis = 0) #use pandas interpolation methods to fill in missing data
        this_marker_interpolated_skel3d_array = np.array(df2)
        #replace the remaining NaN values (the ones that often happen at the start of the recording)
        this_marker_interpolated_skel3d_array = np.where(np.isfinite(this_marker_interpolated_skel3d_array), this_marker_interpolated_skel3d_array, np.nanmean(this_marker_interpolated_skel3d_array))
        
        freemocap_interpolated_data[:,marker,:] = this_marker_interpolated_skel3d_array
        
    return freemocap_interpolated_data





if __name__ == '__main__':
    
    from pathlib import Path
    

    freemocap_data_folder_path = Path(r'D:\ValidationStudy2022\FreeMocap_Data')


    sessionID = 'sesh_2022-05-24_15_55_40_JSM_T1_BOS'
    data_array_folder = 'DataArrays'
    array_name = 'mediaPipeSkel_3d.npy'

    data_array_folder_path = freemocap_data_folder_path / sessionID / data_array_folder



    freemocap_marker_data = np.load(data_array_folder_path / array_name)
    freemocap_marker_data_interpolated = interpolate_freemocap_data(freemocap_marker_data)


    f= 2


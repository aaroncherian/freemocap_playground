
import numpy as np

from freemocap_utils.mediapipe_skeleton_builder import mediapipe_indices
from freemocap_utils.qualisys_indices import qualisys_indices

import pandas as pd

from pathlib import Path

import seaborn as sns
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def calculate_rmse_dataframe(qualisys_data:np.ndarray, freemocap_data:np.ndarray):

    rmse_dataframe = pd.DataFrame()

    dimension_list = ['x','y','z']

    for marker_name in mediapipe_indices:
        if marker_name in qualisys_indices: #only run rmse if the marker is in both qualisys and freemocap
            mediapipe_marker_index = mediapipe_indices.index(marker_name)
            qualisys_marker_index = qualisys_indices.index(marker_name)

            rmse_list = []
            
            for dimension_count,dimension in enumerate(dimension_list):
                rmse = calculate_rmse(qualisys_data[:,qualisys_marker_index,dimension_count],freemocap_data[:,mediapipe_marker_index,dimension_count])
                
                rmse_list.append(rmse) #add rmse per dimension to a list 

            rmse_dict = {'marker': marker_name, 'rmse': rmse_list, 'dimension': dimension_list}
            rmse_dataframe = pd.concat([pd.DataFrame(rmse_dict),rmse_dataframe])
    
    return rmse_dataframe




if __name__ == "__main__":
    
    path_to_freemocap_session_folder = Path(r'D:\ValidationStudy_numCams\FreeMoCap_Data\sesh_2022-05-24_16_10_46_JSM_T1_WalkRun')
    freemocap_data = np.load(path_to_freemocap_session_folder/'DataArrays'/'mediaPipeSkel_3d_origin_aligned.npy')



    path_to_qualisys_session_folder = Path(r"D:\ValidationStudy2022\FreeMocap_Data\qualisys_sesh_2022-05-24_16_02_53_JSM_T1_WalkRun")
    qualisys_data = np.load(path_to_qualisys_session_folder/'DataArrays'/'qualisys_marker_data_29Hz.npy')



    freemocap_sliced_data = freemocap_data[1162:6621,:,:]

    frames_to_use = [0,freemocap_sliced_data.shape[0]]
    rmse_dataframe = calculate_rmse_dataframe(qualisys_data=qualisys_data, freemocap_data=freemocap_sliced_data, frames_to_use=frames_to_use)

    f = 2 


            



